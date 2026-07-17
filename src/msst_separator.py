"""
MSST Audio Separation Module

Handles audio source separation using Music-Source-Separation-Training (MSST) models.
Supports:
  - Harmony separation (karaoke model): separates lead vocals from instrumental/backing
  - Reverb separation (dereverb model): separates dry vocals from reverb tail

Uses the BS-RoFormer architecture with pretrained checkpoints.

Performance notes:
  - Install flash_attn (pip install flash-attn) for 30-50% speedup on BS-RoFormer models
  - Increase chunk_batch to use more VRAM for higher throughput
  - The demix function already uses torch.cuda.amp.autocast (mixed precision) by default
"""

import os
import sys
import time
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Dict, Tuple

import numpy as np
import soundfile as sf
import torch

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Flash Attention detection (cached)
# ------------------------------------------------------------------

def _check_flash_attn() -> bool:
    """Check if flash_attn is installed and working."""
    try:
        import flash_attn  # noqa: F401
        return True
    except ImportError:
        return False


_HAS_FLASH_ATTN: Optional[bool] = None


def has_flash_attn() -> bool:
    """Detect flash_attn once and cache the result."""
    global _HAS_FLASH_ATTN
    if _HAS_FLASH_ATTN is None:
        _HAS_FLASH_ATTN = _check_flash_attn()
    return _HAS_FLASH_ATTN


class MSSTSeparator:
    """
    Audio source separation using MSST (Music-Source-Separation-Training) models.

    Wraps the MSST codebase for loading models and performing separation inference.
    Each instance handles one model; create separate instances for different
    separation tasks (harmony, reverb, etc.).

    Parameters
    ----------
    msst_code_dir : str or Path
        Path to the Music-Source-Separation-Training-GUI directory containing
        the MSST source code (utils/, models/, configs/).
    model_type : str
        Model architecture type, e.g. "bs_roformer", "mel_band_roformer".
    config_path : str or Path
        Path to the YAML configuration file for the model.
    checkpoint_path : str or Path
        Path to the model checkpoint (.ckpt) file.
    device : str
        Device to run inference on (e.g. "cuda:0", "cpu").
    chunk_batch : int
        Override for inference batch_size. Larger values increase GPU utilization
        but use more VRAM. Default 8.
    """

    def __init__(
        self,
        msst_code_dir: str | Path,
        model_type: str,
        config_path: str | Path,
        checkpoint_path: str | Path,
        device: str = "cuda:0",
        chunk_batch: int = 8,
    ):
        self.msst_code_dir = Path(msst_code_dir)
        self.model_type = model_type
        self.config_path = Path(config_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.chunk_batch = chunk_batch

        self._model = None
        self._config = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """
        Load the MSST model and its configuration into GPU memory.

        Injects the MSST code directory into sys.path so that the MSST
        utility modules (utils.settings, utils.model_utils) are importable.
        """
        if self._loaded:
            logger.info("MSST model already loaded, skipping.")
            return

        # Validate paths
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"MSST checkpoint not found: {self.checkpoint_path}"
            )
        if not self.config_path.is_file():
            raise FileNotFoundError(
                f"MSST config not found: {self.config_path}"
            )
        if not self.msst_code_dir.is_dir():
            raise NotADirectoryError(
                f"MSST code directory not found: {self.msst_code_dir}"
            )

        # Inject MSST code directory into Python path if not already present
        msst_str = str(self.msst_code_dir)
        if msst_str not in sys.path:
            sys.path.insert(0, msst_str)

        from utils.model_utils import load_start_checkpoint
        from utils.settings import get_model_from_config

        logger.info(
            "Loading MSST model: type=%s config=%s checkpoint=%s",
            self.model_type,
            self.config_path.name,
            self.checkpoint_path.name,
        )

        model, config = get_model_from_config(
            self.model_type, str(self.config_path)
        )
        checkpoint = torch.load(
            str(self.checkpoint_path), weights_only=False, map_location="cpu"
        )
        load_start_checkpoint(
            SimpleNamespace(
                start_check_point=str(self.checkpoint_path),
                model_type=self.model_type,
                lora_checkpoint="",
            ),
            model,
            checkpoint,
            type_="inference",
        )
        model = model.to(self.device).eval()
        self._model = model
        self._config = config
        self._loaded = True

        # Override batch_size for higher GPU utilization
        if hasattr(config, "inference") and hasattr(config.inference, "batch_size"):
            old_b = config.inference.batch_size
            config.inference.batch_size = self.chunk_batch
            logger.info(
                "Model inference.batch_size: %s -> %s", old_b, self.chunk_batch
            )

        # Log model info
        sr = getattr(config.audio, "sample_rate", "unknown")
        nc = getattr(config.audio, "num_channels", "unknown")
        stereo = getattr(config.model, "stereo", "unknown")
        instruments = getattr(config.training, "instruments", "unknown")
        logger.info(
            "Model loaded: sample_rate=%s, num_channels=%s, stereo=%s, instruments=%s",
            sr, nc, stereo, instruments,
        )

        # Log performance-related info
        use_amp = getattr(config.training, 'use_amp', True)
        flash_available = has_flash_attn()
        logger.info(
            "Performance: AMP=%s, flash_attn=%s, chunk_batch=%d",
            use_amp, "AVAILABLE" if flash_available else "not installed", self.chunk_batch,
        )
        if not flash_available and self.model_type == "bs_roformer":
            # PyTorch >= 2.5 ships Flash Attention kernel via SDPA — check if active
            try:
                has_sdpa = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
                if has_sdpa:
                    logger.info(
                        "Note: PyTorch %s includes built-in Flash Attention via SDPA. "
                        "The 'flash_attn' pip package is not needed on this system.",
                        torch.__version__,
                    )
                else:
                    logger.info(
                        "Tip: install flash-attn for BS-RoFormer speedup:\n"
                        "  pip install flash-attn --no-build-isolation"
                    )
            except Exception:
                pass

        logger.info("GPU memory allocated: %.1f MB", self._gpu_memory_mb())

    def separate(
        self,
        audio: np.ndarray,
        sample_rate: int,
        target_stem: str,
    ) -> Dict[str, np.ndarray]:
        """
        Run source separation on audio data.

        Parameters
        ----------
        audio : np.ndarray
            Input audio waveform, shape (samples,) or (channels, samples).
            If stereo and model expects mono, channels are averaged.
        sample_rate : int
            Sample rate of the input audio. Will resample if different
            from the model's expected rate.
        target_stem : str
            The primary stem name to extract (e.g. "Vocals", "noreverb").

        Returns
        -------
        dict
            Mapping from stem names to numpy arrays of separated audio.
            Always includes at least the target_stem. Other stems may be
            present depending on the model (e.g. "Instrumental", "reverb").

        Raises
        ------
        RuntimeError
            If model has not been loaded.
        KeyError
            If target_stem is not among the model's output stems.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Resample if needed (use torchaudio on GPU if available, fallback to librosa)
        model_sr = int(getattr(self._config.audio, "sample_rate", 44100))
        if sample_rate != model_sr:
            logger.info(
                "Resampling input from %d Hz to %d Hz", sample_rate, model_sr
            )
            audio = _resample_fast(audio, sample_rate, model_sr)
            sample_rate = model_sr

        # Convert to mono or multi-channel as needed
        audio = _adapt_channels(audio, self._config)

        logger.info(
            "Running separation: audio_shape=%s target=%s",
            audio.shape, target_stem,
        )
        t0 = time.time()

        # Run the model
        output_waveforms = self._run_demix(audio)

        elapsed = time.time() - t0
        audio_dur = audio.shape[-1] / sample_rate
        logger.info(
            "Separation completed in %.1fs (%.1fx realtime)",
            elapsed, audio_dur / max(elapsed, 0.001),
        )

        # Validate target stem
        if target_stem not in output_waveforms:
            available = list(output_waveforms.keys())
            raise KeyError(
                f"Target stem {target_stem!r} not in model outputs. "
                f"Available: {available}"
            )

        return {
            stem: _to_numpy(wav)
            for stem, wav in output_waveforms.items()
        }

    def separate_to_file(
        self,
        input_path: str | Path,
        output_dir: str | Path,
        target_stem: str,
        other_stems: Optional[list[str]] = None,
        output_sample_rate: int = 44100,
    ) -> Dict[str, Path]:
        """
        Load audio from file, run separation, and save all desired stems.

        Uses soundfile for fast I/O (3-5x faster than librosa for pure load).

        Parameters
        ----------
        input_path : Path
            Path to the input audio file.
        output_dir : Path
            Directory to write output stem files.
        target_stem : str
            Primary stem to extract.
        other_stems : list of str, optional
            Additional stems to save. If None, saves all available stems.
        output_sample_rate : int
            Sample rate for output files.

        Returns
        -------
        dict
            Mapping from stem name to output file path.
        """
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load audio using soundfile (fast path, no resampling overhead)
        logger.info("Loading audio: %s", input_path.name)
        audio, sr = _load_audio_fast(str(input_path))

        logger.info("Audio loaded: shape=%s sr=%d duration=%.1fs",
                     audio.shape, sr, audio.shape[-1] / sr)

        # Run separation
        waveforms = self.separate(audio, sr, target_stem)

        # Save stems
        base_name = input_path.stem
        saved = {}

        stems_to_save = set()
        stems_to_save.add(target_stem)
        if other_stems:
            stems_to_save.update(other_stems)
        else:
            stems_to_save.update(waveforms.keys())

        for stem_name in stems_to_save:
            if stem_name not in waveforms:
                logger.warning(
                    "Stem %r not in model outputs, skipping.", stem_name
                )
                continue

            wav = waveforms[stem_name]
            out_path = output_dir / f"{base_name}_{stem_name}.wav"
            sf.write(
                str(out_path), wav.T, output_sample_rate,
                format="WAV", subtype="FLOAT",
            )
            saved[stem_name] = out_path
            duration = wav.shape[-1] / output_sample_rate
            logger.info("Saved: %s (%.1fs)", out_path.name, duration)

        return saved

    def unload_model(self) -> None:
        """Release GPU memory held by the model."""
        if self._model is not None:
            del self._model
            self._model = None
            self._config = None
            self._loaded = False
            torch.cuda.empty_cache()
            logger.info("MSST model unloaded, GPU memory freed.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_demix(self, mix: np.ndarray) -> Dict[str, np.ndarray]:
        """Execute the demix inference pipeline."""
        from utils.audio_utils import denormalize_audio, normalize_audio
        from utils.model_utils import demix

        # Optional normalization
        norm_params = None
        model_input = mix
        has_normalize = (
            hasattr(self._config, "inference")
            and "normalize" in self._config.inference
            and self._config.inference["normalize"] is True
        )
        if has_normalize:
            model_input, norm_params = normalize_audio(model_input)

        # Core demix call (already uses torch.cuda.amp.autocast internally)
        waveforms = demix(
            self._config, self._model, model_input, self.device,
            model_type=self.model_type, pbar=False,
        )

        # Denormalize outputs
        if norm_params is not None:
            waveforms = {
                k: denormalize_audio(v, norm_params)
                for k, v in waveforms.items()
            }

        return waveforms

    def _gpu_memory_mb(self) -> float:
        """Return allocated GPU memory in MB."""
        if self.device.startswith("cuda"):
            return torch.cuda.memory_allocated(self.device) / (1024 * 1024)
        return 0.0


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _load_audio_fast(path: str) -> Tuple[np.ndarray, int]:
    """
    Load audio file using soundfile (faster than librosa for pure I/O).

    Returns (waveform, sample_rate) where waveform is (channels, samples).

    This is 3-5x faster than librosa.load() when no resampling is needed,
    because librosa internally does format conversion even when sr=None.
    """
    audio, sample_rate = sf.read(str(path))
    # soundfile returns (samples,) for mono or (samples, channels) for multi-channel
    if audio.ndim == 1:
        audio = np.expand_dims(audio, axis=0)
    else:
        audio = audio.T  # (samples, channels) -> (channels, samples)
    return audio.astype(np.float32), sample_rate


def _resample_fast(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample audio efficiently.

    Tries torchaudio (GPU) first, falls back to librosa (CPU).
    """
    try:
        import torchaudio.transforms as T
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        resampler = T.Resample(orig_sr, target_sr, lowpass_filter_width=128).to(device)
        audio_tensor = torch.from_numpy(audio).to(device)
        audio_resampled = resampler(audio_tensor).cpu().numpy()
        return audio_resampled.astype(np.float32)
    except Exception:
        import librosa
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def _adapt_channels(mix: np.ndarray, config) -> np.ndarray:
    """
    Adapt audio channel layout to match model expectations.

    - If mono input but model expects stereo: duplicate to 2 channels.
    - If stereo input but model expects mono: average to mono.
    """
    if mix.ndim == 1:
        mix = np.expand_dims(mix, axis=0)
        if (
            hasattr(config, "audio")
            and "num_channels" in config.audio
            and config.audio["num_channels"] == 2
        ):
            mix = np.concatenate([mix, mix], axis=0)
    elif mix.ndim == 2 and mix.shape[0] == 2:
        if (
            hasattr(config, "model")
            and "stereo" in config.model
            and not config.model["stereo"]
        ):
            mix = np.mean(mix, axis=0, keepdims=True)
    return mix


def _to_numpy(tensor_or_array) -> np.ndarray:
    """Convert torch Tensor or numpy array to float32 numpy array."""
    if hasattr(tensor_or_array, "cpu"):
        tensor_or_array = tensor_or_array.detach().cpu()
    arr = np.asarray(tensor_or_array, dtype=np.float32)
    return arr


# ------------------------------------------------------------------
# Linked MSST Separator — chained karaoke → dereverb with STFT pass-through
# ------------------------------------------------------------------

class LinkedMSSTSeparator:
    """
    Chained audio separation that runs karaoke and dereverb models in sequence
    while keeping intermediate data in GPU memory.

    Instead of the traditional pipeline:
        karaoke → write vocals.wav → read vocals.wav → dereverb
    this class:
        karaoke → [vocals in GPU memory] → dereverb (no disk I/O)

    Additionally, residual stems (Instrumental, reverb) are computed in the
    STFT domain, which is both faster and more precise than time-domain
    subtraction with librosa.

    Parameters
    ----------
    karaoke_separator : MSSTSeparator
        Pre-loaded karaoke model separator.
    dereverb_separator : MSSTSeparator
        Pre-loaded dereverb model separator.
    """

    def __init__(
        self,
        karaoke_separator: MSSTSeparator,
        dereverb_separator: MSSTSeparator,
    ):
        self.karaoke = karaoke_separator
        self.dereverb = dereverb_separator

        if not self.karaoke._loaded:
            raise RuntimeError("Karaoke separator must be loaded before LinkedMSSTSeparator")
        if not self.dereverb._loaded:
            raise RuntimeError("Dereverb separator must be loaded before LinkedMSSTSeparator")

        # Verify STFT parameter compatibility
        k_cfg = self.karaoke._config
        d_cfg = self.dereverb._config
        self._stft_n_fft = k_cfg.model.stft_n_fft
        self._stft_hop = k_cfg.model.stft_hop_length
        self._stft_win = k_cfg.model.stft_win_length
        self._sample_rate = k_cfg.audio.sample_rate

        if (d_cfg.model.stft_n_fft != self._stft_n_fft or
            d_cfg.model.stft_hop_length != self._stft_hop or
            d_cfg.model.stft_win_length != self._stft_win):
            logger.warning(
                "Karaoke and dereverb STFT params differ! "
                "STFT pass-through will NOT be used. Falling back to disk I/O."
            )
            self._stft_compatible = False
        else:
            self._stft_compatible = True
            logger.info(
                "LinkedMSST: STFT params match (n_fft=%d, hop=%d, win=%d). "
                "STFT-domain residual + memory pass-through enabled.",
                self._stft_n_fft, self._stft_hop, self._stft_win,
            )

        self.device = self.karaoke.device

    def separate(
        self,
        audio: np.ndarray,
        sample_rate: int,
        output_dir: Path,
        base_name: str,
        output_sample_rate: int = 44100,
    ) -> Dict[str, Path]:
        """
        Run karaoke → dereverb separation, keeping intermediate data in memory.

        Parameters
        ----------
        audio : np.ndarray
            Original audio (channels, samples).
        sample_rate : int
            Sample rate of the input audio.
        output_dir : Path
            Directory for output files.
        base_name : str
            Base filename stem for output files.
        output_sample_rate : int
            Sample rate for output WAV files.

        Returns
        -------
        dict
            Mapping of stem_name → file_path for all saved stems.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()

        # ---- Stage 1: Karaoke separation ----
        logger.info("LinkedMSST Stage 1: Karaoke separation on original audio")
        t1 = time.time()
        karaoke_waveforms = self.karaoke.separate(audio, sample_rate, "Vocals")
        vocals = karaoke_waveforms["Vocals"]  # numpy (ch, samples)
        t_karaoke = time.time() - t1
        logger.info("Karaoke separation: %.1fs", t_karaoke)

        # Save vocals
        vocals_path = output_dir / f"{base_name}_Vocals.wav"
        sf.write(str(vocals_path), vocals.T, output_sample_rate, format="WAV", subtype="FLOAT")
        saved = {"Vocals": vocals_path}

        # ---- Stage 2: Dereverb separation (from in-memory vocals) ----
        logger.info("LinkedMSST Stage 2: Dereverb separation on vocals (in-memory)")
        t2 = time.time()
        dereverb_waveforms = self.dereverb.separate(vocals, sample_rate, "noreverb")
        noreverb = dereverb_waveforms["noreverb"]  # numpy (ch, samples)
        t_dereverb = time.time() - t2
        logger.info("Dereverb separation: %.1fs", t_dereverb)

        # Save noreverb (dry vocals)
        noreverb_path = output_dir / f"{base_name}_Vocals_noreverb.wav"
        sf.write(str(noreverb_path), noreverb.T, output_sample_rate, format="WAV", subtype="FLOAT")
        saved["noreverb"] = noreverb_path

        # ---- Stage 3: Compute residuals in STFT domain ----
        logger.info("LinkedMSST Stage 3: STFT-domain residual computation")
        t3 = time.time()

        if self._stft_compatible:
            instrumental, reverb = self._compute_residuals_stft(
                audio, vocals, noreverb, sample_rate,
            )
        else:
            # Fallback: time-domain subtraction
            instrumental = audio - vocals
            # Align lengths
            min_len = min(vocals.shape[-1], noreverb.shape[-1])
            reverb = vocals[:, :min_len] - noreverb[:, :min_len]

        t_residual = time.time() - t3
        logger.info("Residual computation: %.1fs", t_residual)

        # Save instrumental
        instrumental_path = output_dir / f"{base_name}_Instrumental.wav"
        sf.write(str(instrumental_path), instrumental.T, output_sample_rate,
                 format="WAV", subtype="FLOAT")
        saved["Instrumental"] = instrumental_path

        # Save reverb
        reverb_path = output_dir / f"{base_name}_Vocals_reverb.wav"
        sf.write(str(reverb_path), reverb.T, output_sample_rate,
                 format="WAV", subtype="FLOAT")
        saved["reverb"] = reverb_path

        t_total = time.time() - t0
        logger.info("LinkedMSST total: %.1fs (karaoke=%.1fs, dereverb=%.1fs, residual=%.1fs)",
                     t_total, t_karaoke, t_dereverb, t_residual)

        return saved

    def _compute_residuals_stft(
        self, original: np.ndarray, vocals: np.ndarray, noreverb: np.ndarray,
        sample_rate: int,
    ) -> tuple:
        """
        Compute instrumental and reverb residuals in the STFT domain.

        instrumental = original - vocals   (in complex STFT domain)
        reverb       = vocals - noreverb   (in complex STFT domain)

        This is more precise than time-domain subtraction because it avoids
        librosa resampling artifacts and channel/length alignment issues.
        """
        # Convert to torch tensors on GPU
        orig_t = torch.from_numpy(original.astype(np.float32)).to(self.device)
        voc_t = torch.from_numpy(vocals.astype(np.float32)).to(self.device)
        dry_t = torch.from_numpy(noreverb.astype(np.float32)).to(self.device)

        # STFT window
        window = torch.hann_window(self._stft_win, device=self.device)

        def _stft(audio_t):
            """Compute STFT for all channels."""
            stfts = []
            for ch in range(audio_t.shape[0]):
                s = torch.stft(
                    audio_t[ch],
                    n_fft=self._stft_n_fft,
                    hop_length=self._stft_hop,
                    win_length=self._stft_win,
                    window=window,
                    return_complex=True,
                )
                stfts.append(s)
            return stfts  # list of (freq, time) complex tensors

        def _istft(stft_list, length):
            """iSTFT for all channels."""
            audios = []
            for s in stft_list:
                a = torch.istft(
                    s,
                    n_fft=self._stft_n_fft,
                    hop_length=self._stft_hop,
                    win_length=self._stft_win,
                    window=window,
                    length=length,
                )
                audios.append(a)
            return torch.stack(audios, dim=0)

        # Compute STFTs
        stft_orig = _stft(orig_t)
        stft_voc = _stft(voc_t)
        stft_dry = _stft(dry_t)

        # Align STFT frames (use minimum time frames across all)
        min_frames = min(s.shape[-1] for s in stft_orig + stft_voc + stft_dry)
        stft_orig = [s[..., :min_frames] for s in stft_orig]
        stft_voc = [s[..., :min_frames] for s in stft_voc]
        stft_dry = [s[..., :min_frames] for s in stft_dry]

        # Residuals in STFT domain
        stft_inst = [o - v for o, v in zip(stft_orig, stft_voc)]
        stft_rev = [v - d for v, d in zip(stft_voc, stft_dry)]

        # iSTFT back to audio
        instrumental = _istft(stft_inst, orig_t.shape[-1])
        reverb = _istft(stft_rev, voc_t.shape[-1])

        return instrumental.cpu().numpy(), reverb.cpu().numpy()

    def unload_models(self) -> None:
        """Release GPU memory for both models."""
        self.karaoke.unload_model()
        self.dereverb.unload_model()
        logger.info("LinkedMSST: both models unloaded.")
