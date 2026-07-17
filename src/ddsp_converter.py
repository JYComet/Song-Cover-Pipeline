"""
DDSP-SVC Timbre Conversion Module

Wraps the DDSP-SVC 6.3 (RectifiedFlow) inference pipeline for singing voice
conversion. Converts the timbre of input vocals to a target speaker while
preserving pitch, rhythm, and expression.

Key parameters (mapped to user-facing terminology):
  - infer_step  (推理轮 / rounds): Number of ODE sampling steps (default 100)
  - t_start     (浮动 / float): ODE start time, 0=pure diffusion, 1=pure DDSP (default 0.4)
  - method:     ODE solver method ("euler" or "rk4")
  - pitch_extractor: F0 extraction method ("rmvpe" default)
"""

import os
import sys
import time
import logging
import hashlib
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class DDSPConverter:
    """
    Singing voice conversion using DDSP-SVC RectifiedFlow model.

    Wraps the DDSP-SVC 6.3 inference pipeline. The model converts the
    timbre of input vocals while preserving the original pitch contour,
    rhythm, and expression.

    Parameters
    ----------
    ddsp_project_dir : str or Path
        Root directory of the DDSP-SVC 6.3 project containing:
        main_reflow.py, reflow/, ddsp/, nsf_hifigan/, encoder/, slicer.py, pretrain/
    model_ckpt : str or Path
        Path to the model checkpoint (.pt file), relative to ddsp_project_dir
        or absolute. E.g. "exp/ria/model_20000.pt".
    device : str
        Device for inference ("cuda:0", "cpu").
    infer_step : int
        Number of RectifiedFlow ODE sampling steps. More steps = better
        quality but slower. Default 100.
    t_start : float
        ODE start time (0.0 to 1.0). 0.0 = full diffusion refinement,
        1.0 = pure DDSP (no refinement). Default 0.4 provides a good
        balance of quality and timbre preservation.
    method : str
        ODE solver: "euler" (faster) or "rk4" (more accurate).
    pitch_extractor : str
        F0 extraction method: "rmvpe", "fcpe", "parselmouth", "dio",
        "harvest", or "crepe".
    key : int or float
        Pitch shift in semitones. 0 = original pitch.
    vocal_register_shift : int or float
        Vocal register shift in semitones (only for PC-type vocoder).
    formant_shift : int or float
        Formant shift in semitones (only for pitch-augmented model).
    f0_min : float
        Minimum F0 in Hz.
    f0_max : float
        Maximum F0 in Hz.
    threshold : float
        Response threshold in dB. Frames below this level are silenced.
    spk_id : int
        Speaker ID for multi-speaker models. Default 1.
    """

    def __init__(
        self,
        ddsp_project_dir: str | Path,
        model_ckpt: str | Path,
        device: str = "cuda:0",
        infer_step: int = 100,
        t_start: float = 0.4,
        method: str = "euler",
        pitch_extractor: str = "rmvpe",
        key: float = 0.0,
        vocal_register_shift: float = 0.0,
        formant_shift: float = 0.0,
        f0_min: float = 65.0,
        f0_max: float = 800.0,
        threshold: float = -60.0,
        spk_id: int = 1,
    ):
        # Store original working directory — all relative paths are
        # resolved against this (the project root), not ddsp_project_dir.
        self._original_cwd = os.getcwd()
        self.ddsp_project_dir = Path(ddsp_project_dir)
        if not self.ddsp_project_dir.is_absolute():
            self.ddsp_project_dir = Path(self._original_cwd) / self.ddsp_project_dir
        self.model_ckpt_rel = model_ckpt  # may be relative
        self.device = device
        self.infer_step = infer_step
        self.t_start = t_start
        self.method = method
        self.pitch_extractor = pitch_extractor
        self.key = key
        self.vocal_register_shift = vocal_register_shift
        self.formant_shift = formant_shift
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.threshold = threshold
        self.spk_id = spk_id

        # Resolved at load time
        self._model = None
        self._vocoder = None
        self._args = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """
        Load the DDSP-SVC model, vocoder, and encoders.

        Changes the working directory to the DDSP project root so that
        relative paths in the config (pretrain/..., encoder/...) resolve
        correctly. Restores the original directory after loading.
        """
        if self._loaded:
            logger.info("DDSP model already loaded, skipping.")
            return

        # Resolve model checkpoint path.
        # Relative paths are resolved against the original working directory
        # (project root), NOT against ddsp_project_dir — model checkpoints
        # live in models/DDSP/, separately from the DDSP source in vendor/ddsp/.
        model_ckpt_path = Path(self.model_ckpt_rel)
        if not model_ckpt_path.is_absolute():
            model_ckpt_path = Path(self._original_cwd) / model_ckpt_path

        if not model_ckpt_path.is_file():
            raise FileNotFoundError(
                f"DDSP model checkpoint not found: {model_ckpt_path}"
            )

        logger.info("DDSP project dir: %s", self.ddsp_project_dir)
        logger.info("DDSP model checkpoint: %s", model_ckpt_path)

        # Inject DDSP project root directory into sys.path
        # DO NOT add subdirectories (reflow/, ddsp/, etc.) individually —
        # they must be importable as sub-packages under the root.  The reflow
        # directory lacks __init__.py so Python treats it as a namespace
        # package only when discovered via its parent directory.
        ddsp_str = str(self.ddsp_project_dir)
        if ddsp_str not in sys.path:
            sys.path.insert(0, ddsp_str)

        # We need to temporarily chdir to the DDSP project directory
        # because the code uses relative paths to load pretrained models:
        #   - 'pretrain/rmvpe/model.pt'
        #   - 'pretrain/contentvec/pytorch_model.bin'
        #   - vocoder ckpt from config
        original_cwd = os.getcwd()
        os.chdir(str(self.ddsp_project_dir))

        try:
            import torch
            from reflow.vocoder import load_model_vocoder

            logger.info("Loading DDSP model + vocoder...")
            t0 = time.time()

            self._model, self._vocoder, self._args = load_model_vocoder(
                str(model_ckpt_path), device=self.device
            )

            elapsed = time.time() - t0
            logger.info("DDSP model loaded in %.1fs", elapsed)

            # Validate config compatibility
            sampling_rate = self._args.data.sampling_rate
            block_size = self._args.data.block_size
            encoder = self._args.data.encoder
            logger.info(
                "DDSP config: sampling_rate=%d, block_size=%d, encoder=%s, "
                "n_spk=%d, vocoder_type=%s",
                sampling_rate, block_size, encoder,
                self._args.model.n_spk, self._args.vocoder.type,
            )

            # Log inference parameters
            actual_method = self.method
            if actual_method == "auto":
                actual_method = self._args.infer.method
            actual_step = self.infer_step
            if actual_step < 0:
                actual_step = self._args.infer.infer_step

            logger.info(
                "Inference params: infer_step=%d, t_start=%.2f, method=%s, "
                "pitch_extractor=%s, key=%.1f",
                actual_step, self.t_start, actual_method,
                self.pitch_extractor, self.key,
            )

            self._loaded = True
            logger.info("GPU memory allocated: %.1f MB", _gpu_memory_mb(self.device))

        finally:
            os.chdir(original_cwd)

    def convert(
        self,
        input_path: str | Path,
        output_path: str | Path,
    ) -> Path:
        """
        Convert the timbre of vocals in an audio file.

        Parameters
        ----------
        input_path : Path
            Path to input audio file (should be clean, dry vocals).
        output_path : Path
            Path to write the timbre-converted output.

        Returns
        -------
        Path
            Path to the output file.

        Raises
        ------
        RuntimeError
            If model has not been loaded.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        input_path = Path(input_path).resolve()
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Converting: %s -> %s", input_path.name, output_path.name)
        t0 = time.time()

        # We must run inference from within the DDSP project directory
        # because the code uses relative imports and paths.
        # IMPORTANT: resolve paths to absolute BEFORE chdir, otherwise
        # relative paths break after the directory change.
        original_cwd = os.getcwd()
        abs_input = str(input_path)
        abs_output = str(output_path)
        os.chdir(str(self.ddsp_project_dir))

        try:
            self._run_inference(abs_input, abs_output)
        finally:
            os.chdir(original_cwd)

        elapsed = time.time() - t0
        if output_path.is_file():
            import soundfile as sf
            info = sf.info(str(output_path))
            logger.info(
                "Conversion completed in %.1fs, output: %.1fs",
                elapsed, info.duration,
            )
        else:
            logger.error("Output file was not created: %s", output_path)

        return output_path

    def unload_model(self) -> None:
        """Release GPU memory held by the model, vocoder, and encoders."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._vocoder is not None:
            del self._vocoder
            self._vocoder = None
        self._args = None
        self._loaded = False

        import torch
        torch.cuda.empty_cache()
        logger.info("DDSP model unloaded, GPU memory freed.")

    # ------------------------------------------------------------------
    # Internal: inference pipeline (mirrors main_reflow.py logic)
    # ------------------------------------------------------------------

    def _run_inference(self, input_path: str, output_path: str) -> None:
        """
        Execute the full DDSP-SVC inference pipeline.

        This replicates the flow from main_reflow.py:
          1. Load audio
          2. Extract / load cached F0
          3. Apply key shift to F0
          4. Extract volume envelope + threshold mask
          5. Encode units (ContentVec)
          6. Slice audio into segments
          7. For each segment: model forward -> vocoder decode
          8. Cross-fade segments and write output

        Note: input_path and output_path are absolute path strings
              (already resolved before chdir to DDSP project dir).
        """
        from pathlib import Path as _Path
        input_file = _Path(input_path)
        output_file = _Path(output_path)
        import torch
        import librosa
        import soundfile as sf
        from ddsp.vocoder import F0_Extractor, Volume_Extractor, Units_Encoder
        from ddsp.core import upsample
        from slicer import Slicer
        from tqdm import tqdm

        # ---- Resolve inference parameters ----
        infer_step = self.infer_step
        if infer_step < 0:
            infer_step = self._args.infer.infer_step

        method = self.method
        if method == "auto":
            method = self._args.infer.method

        t_start = self.t_start
        if t_start < 0:
            if self._args.model.t_start is not None:
                t_start = float(self._args.model.t_start)
            else:
                t_start = 0.0
        if (
            hasattr(self._args.model, "t_start")
            and self._args.model.t_start is not None
            and t_start < self._args.model.t_start
        ):
            t_start = self._args.model.t_start

        # ---- Load input audio ----
        audio, sample_rate = librosa.load(input_path, sr=None)
        if audio.ndim > 1:
            audio = librosa.to_mono(audio)
        logger.info("Input audio: duration=%.1fs, sr=%d", len(audio) / sample_rate, sample_rate)

        hop_size = (
            self._args.data.block_size
            * sample_rate
            / self._args.data.sampling_rate
        )
        win_size = (
            self._args.data.volume_smooth_size
            * sample_rate
            / self._args.data.sampling_rate
        )

        # ---- Extract (or load cached) F0 ----
        md5_hash = hashlib.md5(input_file.read_bytes()).hexdigest()
        cache_dir = self.ddsp_project_dir / "cache"
        cache_file = (
            cache_dir
            / f"{self.pitch_extractor}_{hop_size}_{self.f0_min}_{self.f0_max}_{md5_hash}.npy"
        )

        if cache_file.exists():
            logger.info("Loading cached F0: %s", cache_file.name)
            f0 = np.load(str(cache_file), allow_pickle=False)
        else:
            logger.info("Extracting F0 using %s...", self.pitch_extractor)
            pitch_extractor = F0_Extractor(
                self.pitch_extractor,
                sample_rate,
                hop_size,
                float(self.f0_min),
                float(self.f0_max),
            )
            f0 = pitch_extractor.extract(audio, uv_interp=True, device=self.device)
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(str(cache_file), f0, allow_pickle=False)
            logger.info("F0 cached: %s", cache_file.name)

        f0_tensor = (
            torch.from_numpy(f0).float().to(self.device).unsqueeze(-1).unsqueeze(0)
        )

        # ---- Apply key shift ----
        f0_tensor = f0_tensor * (2 ** (float(self.key) / 12))

        # ---- Formant shift ----
        formant_shift_tensor = torch.from_numpy(
            np.array([[float(self.formant_shift)]])
        ).float().to(self.device)

        # ---- Vocal register factor ----
        if getattr(self._vocoder.vocoder.h, 'pc_aug', False):
            vocal_register_factor = 2 ** (float(self.vocal_register_shift) / 12)
        else:
            if float(self.vocal_register_shift) != 0:
                logger.warning(
                    "Vocal register shift not supported for current vocoder, ignored."
                )
            vocal_register_factor = 1.0

        # ---- Extract volume ----
        logger.info("Extracting volume envelope...")
        volume_extractor = Volume_Extractor(hop_size, win_size)
        volume = volume_extractor.extract(audio)
        mask = (volume > 10 ** (float(self.threshold) / 20)).astype("float")
        mask_tensor = (
            torch.from_numpy(mask).float().to(self.device).unsqueeze(-1).unsqueeze(0)
        )
        mask_tensor = upsample(mask_tensor, self._args.data.block_size).squeeze(-1)
        volume_tensor = (
            torch.from_numpy(volume).float().to(self.device).unsqueeze(-1).unsqueeze(0)
        )

        # ---- Load units encoder ----
        if self._args.data.encoder == "cnhubertsoftfish":
            cnhubertsoft_gate = self._args.data.cnhubertsoft_gate
        else:
            cnhubertsoft_gate = 10
        units_encoder = Units_Encoder(
            self._args.data.encoder,
            self._args.data.encoder_ckpt,
            self._args.data.encoder_sample_rate,
            self._args.data.encoder_hop_size,
            cnhubertsoft_gate=cnhubertsoft_gate,
            device=self.device,
        )

        # Speaker ID
        spk_id_tensor = torch.LongTensor(
            np.array([[int(self.spk_id)]])
        ).to(self.device)

        logger.info(
            "Inference: infer_step=%d, t_start=%.2f, method=%s",
            infer_step, t_start, method,
        )

        # ---- Slice and infer ----
        result = np.zeros(0)
        current_length = 0
        segments = _split(audio, sample_rate, hop_size)
        logger.info("Audio sliced into %d segments", len(segments))

        with torch.no_grad():
            for segment in tqdm(segments, desc="DDSP inference", unit="seg"):
                start_frame = segment[0]
                seg_input = (
                    torch.from_numpy(segment[1]).float().unsqueeze(0).to(self.device)
                )

                # Encode units
                seg_units = units_encoder.encode(seg_input, sample_rate, hop_size)

                # Slice F0 and volume to match segment
                seg_f0 = f0_tensor[
                    :, start_frame : start_frame + seg_units.size(1), :
                ]
                seg_volume = volume_tensor[
                    :, start_frame : start_frame + seg_units.size(1), :
                ]

                # Model forward
                seg_mel = self._model(
                    seg_units,
                    seg_f0 / vocal_register_factor,
                    seg_volume,
                    spk_id=spk_id_tensor,
                    spk_mix_dict=None,
                    aug_shift=formant_shift_tensor,
                    vocoder=self._vocoder,
                    infer_step=infer_step,
                    method=method,
                    t_start=t_start,
                )

                # Vocoder decode
                seg_output = self._vocoder.infer(seg_mel, seg_f0)

                # Apply volume mask
                seg_output *= mask_tensor[
                    :,
                    start_frame
                    * self._args.data.block_size : (
                        start_frame + seg_units.size(1)
                    )
                    * self._args.data.block_size,
                ]

                seg_output = seg_output.squeeze().cpu().numpy()

                # Cross-fade stitching
                silent_length = (
                    round(start_frame * self._args.data.block_size) - current_length
                )
                if silent_length >= 0:
                    result = np.append(result, np.zeros(silent_length))
                    result = np.append(result, seg_output)
                else:
                    result = _cross_fade(result, seg_output, current_length + silent_length)

                current_length = (
                    current_length + silent_length + len(seg_output)
                )

        # ---- Write output ----
        sf.write(output_path, result, self._args.data.sampling_rate)
        logger.info("Output written: %s (%.1fs)", output_file.name, len(result) / self._args.data.sampling_rate)


# ------------------------------------------------------------------
# Internal helpers (mirror functions from main_reflow.py)
# ------------------------------------------------------------------

def _split(audio: np.ndarray, sample_rate: float, hop_size: float,
           db_thresh: float = -40, min_len: int = 5000) -> list:
    """
    Slice audio into segments at silence boundaries.

    Returns list of (start_frame, audio_segment) tuples.
    """
    class _Slicer:
        def __init__(self, sr, threshold, min_length):
            self.sr = sr
            self.threshold = 10 ** (threshold / 20.0)
            self.hop_size = round(sr * 20 / 1000)
            self.win_size = min(round(sr * 300 / 1000), 4 * self.hop_size)
            self.min_length = round(sr * min_length / 1000 / self.hop_size)
            self.min_interval = round(sr * 300 / 1000 / self.hop_size)
            self.max_sil_kept = round(sr * 5000 / 1000 / self.hop_size)

        def slice(self, waveform):
            import librosa
            samples = waveform if waveform.ndim == 1 else librosa.to_mono(waveform.T) if waveform.ndim == 2 and waveform.shape[0] <= 2 else librosa.to_mono(waveform)
            if isinstance(samples, np.ndarray) and samples.ndim > 1:
                samples = samples.flatten()
            if len(samples) <= self.min_length:
                return {"0": {"slice": False, "split_time": f"0,{len(waveform)}"}}
            rms_list = librosa.feature.rms(y=samples, frame_length=self.win_size, hop_length=self.hop_size).squeeze(0)
            sil_tags = []
            silence_start = None
            clip_start = 0
            for i, rms in enumerate(rms_list):
                if rms < self.threshold:
                    if silence_start is None:
                        silence_start = i
                    continue
                if silence_start is None:
                    continue
                is_leading_silence = silence_start == 0 and i > self.max_sil_kept
                need_slice_middle = i - silence_start >= self.min_interval and i - clip_start >= self.min_length
                if not is_leading_silence and not need_slice_middle:
                    silence_start = None
                    continue
                if i - silence_start <= self.max_sil_kept:
                    pos = rms_list[silence_start: i + 1].argmin() + silence_start
                    if silence_start == 0:
                        sil_tags.append((0, pos))
                    else:
                        sil_tags.append((pos, pos))
                    clip_start = pos
                elif i - silence_start <= self.max_sil_kept * 2:
                    pos = rms_list[i - self.max_sil_kept: silence_start + self.max_sil_kept + 1].argmin() + i - self.max_sil_kept
                    pos_l = rms_list[silence_start: silence_start + self.max_sil_kept + 1].argmin() + silence_start
                    pos_r = rms_list[i - self.max_sil_kept: i + 1].argmin() + i - self.max_sil_kept
                    if silence_start == 0:
                        sil_tags.append((0, pos_r))
                        clip_start = pos_r
                    else:
                        sil_tags.append((min(pos_l, pos), max(pos_r, pos)))
                        clip_start = max(pos_r, pos)
                else:
                    pos_l = rms_list[silence_start: silence_start + self.max_sil_kept + 1].argmin() + silence_start
                    pos_r = rms_list[i - self.max_sil_kept: i + 1].argmin() + i - self.max_sil_kept
                    if silence_start == 0:
                        sil_tags.append((0, pos_r))
                    else:
                        sil_tags.append((pos_l, pos_r))
                    clip_start = pos_r
                silence_start = None
            total_frames = rms_list.shape[0]
            if silence_start is not None and total_frames - silence_start >= self.min_interval:
                silence_end = min(total_frames, silence_start + self.max_sil_kept)
                pos = rms_list[silence_start: silence_end + 1].argmin() + silence_start
                sil_tags.append((pos, total_frames + 1))
            if len(sil_tags) == 0:
                return {"0": {"slice": False, "split_time": f"0,{len(waveform)}"}}
            else:
                chunks = []
                if sil_tags[0][0]:
                    chunks.append({"slice": False, "split_time": f"0,{min(len(waveform), sil_tags[0][0] * self.hop_size)}"})
                for i in range(len(sil_tags)):
                    if i:
                        chunks.append({"slice": False, "split_time": f"{sil_tags[i-1][1] * self.hop_size},{min(len(waveform), sil_tags[i][0] * self.hop_size)}"})
                    chunks.append({"slice": True, "split_time": f"{sil_tags[i][0] * self.hop_size},{min(len(waveform), sil_tags[i][1] * self.hop_size)}"})
                if sil_tags[-1][1] * self.hop_size < len(waveform):
                    chunks.append({"slice": False, "split_time": f"{sil_tags[-1][1] * self.hop_size},{len(waveform)}"})
                return {str(i): c for i, c in enumerate(chunks)}

    slicer = _Slicer(
        sr=sample_rate, threshold=db_thresh, min_length=min_len
    )
    chunks = dict(slicer.slice(audio))
    result = []
    for k, v in chunks.items():
        tag = v["split_time"].split(",")
        if tag[0] != tag[1]:
            start_frame = int(int(tag[0]) // hop_size)
            end_frame = int(int(tag[1]) // hop_size)
            if end_frame > start_frame:
                result.append((
                    start_frame,
                    audio[int(start_frame * hop_size) : int(end_frame * hop_size)],
                ))
    return result


def _cross_fade(a: np.ndarray, b: np.ndarray, idx: int) -> np.ndarray:
    """Cross-fade two audio segments for seamless stitching."""
    result = np.zeros(idx + len(b))
    fade_len = len(a) - idx
    np.copyto(dst=result[:idx], src=a[:idx])
    k = np.linspace(0, 1.0, num=fade_len, endpoint=True)
    result[idx: len(a)] = (1 - k) * a[idx:] + k * b[:fade_len]
    np.copyto(dst=result[len(a):], src=b[fade_len:])
    return result


def _gpu_memory_mb(device: str) -> float:
    """Return allocated GPU memory in MB."""
    import torch
    if device.startswith("cuda"):
        return torch.cuda.memory_allocated(device) / (1024 * 1024)
    return 0.0
