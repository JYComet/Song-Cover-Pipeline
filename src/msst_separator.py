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
            logger.info(
                "Tip: install flash_attn for 30-50%% speedup on BS-RoFormer models:\n"
                "  pip install flash-attn --no-build-isolation"
            )

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
