"""
Audio Mixer Module

Handles multi-track audio mixing for the final cover song assembly.
Mixes the timbre-converted vocals with the original instrumental (BGM+harmony)
and reverb tail to produce the final cover song.

Features:
  - Channel unification (mono → stereo expansion)
  - Length alignment (truncation or zero-padding)
  - Per-track gain adjustment (dB)
  - Peak normalization of the final mix
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


class AudioMixer:
    """
    Multi-track audio mixer for cover song assembly.

    Mixes multiple audio tracks with configurable gain levels.
    Handles channel unification (mono→stereo), length alignment,
    and output normalization.

    Parameters
    ----------
    sample_rate : int
        Target sample rate for mixing. All tracks are resampled to this rate.
    vocal_gain_db : float
        Gain for the vocal track in dB. 0.0 = unity gain.
    instrumental_gain_db : float
        Gain for the instrumental/backing track in dB.
    reverb_gain_db : float
        Gain for the reverb track in dB.
    normalize_output : bool
        If True, apply peak normalization to prevent clipping.
    output_format : str
        Output audio format ("wav" or "flac").
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        vocal_gain_db: float = 0.0,
        instrumental_gain_db: float = 0.0,
        reverb_gain_db: float = -3.0,
        normalize_output: bool = True,
        output_format: str = "wav",
    ):
        self.sample_rate = sample_rate
        self.vocal_gain_db = vocal_gain_db
        self.instrumental_gain_db = instrumental_gain_db
        self.reverb_gain_db = reverb_gain_db
        self.normalize_output = normalize_output
        self.output_format = output_format

    def mix(
        self,
        vocal_path: str | Path,
        instrumental_path: str | Path,
        reverb_path: str | Path,
        output_path: str | Path,
    ) -> Path:
        """
        Mix vocals, instrumental, and reverb tracks into final cover song.

        Parameters
        ----------
        vocal_path : Path
            Path to the new timbre-converted vocal track.
        instrumental_path : Path
            Path to the instrumental/backing track (from harmony separation).
        reverb_path : Path
            Path to the reverb tail track (from reverb separation).
        output_path : Path
            Path to write the final mixed cover song.

        Returns
        -------
        Path
            Path to the output file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Mixing tracks for: %s", output_path.name)

        # Load all tracks
        tracks: List[Tuple[np.ndarray, float]] = []

        vocal, vr = self._load_track(vocal_path, "vocals")
        tracks.append((vocal, self.vocal_gain_db))

        instrumental, ir = self._load_track(instrumental_path, "instrumental")
        tracks.append((instrumental, self.instrumental_gain_db))

        reverb, rr = self._load_track(reverb_path, "reverb")
        tracks.append((reverb, self.reverb_gain_db))

        # Log track info
        for name, (arr, gain) in zip(
            ["vocals", "instrumental", "reverb"], tracks
        ):
            logger.info(
                "  %s: shape=%s, peak=%.2fdB, gain=%.1fdB",
                name, arr.shape, _peak_db(arr), gain,
            )

        # ---- Process each track ----
        processed = []
        for name, (arr, gain_db) in zip(
            ["vocals", "instrumental", "reverb"], tracks
        ):
            # Resample if needed
            # (tracks loaded at self.sample_rate by _load_track)

            # Apply gain
            gain_linear = 10.0 ** (gain_db / 20.0)
            arr = arr * gain_linear
            logger.debug("  %s: applied gain %.1fdB (%.3fx)", name, gain_db, gain_linear)

            # Convert to stereo (2, samples)
            arr = _ensure_stereo(arr)
            processed.append(arr)

        # ---- Length alignment ----
        max_len = max(arr.shape[-1] for arr in processed)
        logger.info("Max track length: %.1fs", max_len / self.sample_rate)

        aligned = []
        for name, arr in zip(
            ["vocals", "instrumental", "reverb"], processed
        ):
            if arr.shape[-1] < max_len:
                # Zero-pad shorter tracks
                pad_width = max_len - arr.shape[-1]
                arr = np.pad(arr, ((0, 0), (0, pad_width)), mode="constant")
                logger.info("  %s: padded %.1fs", name, pad_width / self.sample_rate)
            elif arr.shape[-1] > max_len:
                # Truncate longer tracks (shouldn't happen)
                arr = arr[:, :max_len]
                logger.info("  %s: truncated to %.1fs", name, max_len / self.sample_rate)
            aligned.append(arr)

        # ---- Sum tracks ----
        mix = np.zeros_like(aligned[0])
        for arr in aligned:
            mix += arr

        # ---- Peak normalization ----
        peak = np.max(np.abs(mix))
        logger.info("Pre-normalization peak: %.2fdB", _linear_to_db(peak))

        if self.normalize_output and peak > 0.0:
            if peak > 1.0:
                # Clipping would occur — normalize down
                mix = mix / peak * 0.99
                logger.info("Normalized: peak reduced to -0.09 dB")
            elif peak < 0.01:
                # Very quiet — normalize up
                mix = mix / peak * 0.99
                logger.info("Normalized: peak boosted to -0.09 dB")

        final_peak = np.max(np.abs(mix))
        logger.info("Final peak: %.2fdB", _linear_to_db(final_peak))

        # ---- Write output ----
        subtype = "FLOAT"
        sf.write(
            str(output_path), mix.T, self.sample_rate,
            format=self.output_format.upper(), subtype=subtype,
        )
        duration = mix.shape[-1] / self.sample_rate
        logger.info("Final mix written: %s (%.1fs, stereo)", output_path.name, duration)

        return output_path

    def mix_generic(
        self,
        tracks: Dict[str, Path],
        output_path: str | Path,
        gains: Optional[Dict[str, float]] = None,
    ) -> Path:
        """
        Generic multi-track mixer for arbitrary track combinations.

        Parameters
        ----------
        tracks : dict
            Mapping of track_name -> file_path.
        output_path : Path
            Output file path.
        gains : dict, optional
            Mapping of track_name -> gain_db. Default 0.0 for all.

        Returns
        -------
        Path
            Output file path.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if gains is None:
            gains = {}

        logger.info("Mixing %d tracks for: %s", len(tracks), output_path.name)

        processed = []
        for name, path in tracks.items():
            arr, _ = self._load_track(path, name)
            gain_db = gains.get(name, 0.0)
            gain_linear = 10.0 ** (gain_db / 20.0)
            arr = arr * gain_linear
            arr = _ensure_stereo(arr)
            processed.append(arr)
            logger.info("  %s: peak=%.2fdB, gain=%.1fdB", name, _peak_db(arr), gain_db)

        # Align lengths
        max_len = max(arr.shape[-1] for arr in processed)
        aligned = []
        for arr in processed:
            if arr.shape[-1] < max_len:
                arr = np.pad(arr, ((0, 0), (0, max_len - arr.shape[-1])), mode="constant")
            aligned.append(arr)

        # Sum and normalize
        mix = np.sum(aligned, axis=0)
        peak = np.max(np.abs(mix))
        if self.normalize_output and peak > 0.0:
            if peak > 1.0:
                mix = mix / peak * 0.99
            elif peak < 0.01:
                mix = mix / peak * 0.99

        sf.write(
            str(output_path), mix.T, self.sample_rate,
            format=self.output_format.upper(), subtype="FLOAT",
        )
        logger.info("Mix written: %s", output_path.name)
        return output_path

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_track(self, path: str | Path, label: str) -> Tuple[np.ndarray, int]:
        """
        Load an audio file and return (waveform, sample_rate).

        Waveform is returned as (channels, samples).
        Resamples to self.sample_rate if needed.
        Uses soundfile for fast I/O, falls back to librosa for resampling.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Track file not found: {path} ({label})")

        # Fast path: use soundfile for pure load (3-5x faster than librosa)
        audio, sr = sf.read(str(path))

        # Normalize shape: always (channels, samples)
        if audio.ndim == 1:
            audio = np.expand_dims(audio, axis=0)
        else:
            audio = audio.T  # (samples, channels) -> (channels, samples)
        audio = audio.astype(np.float32)

        # Resample if needed
        if sr != self.sample_rate:
            logger.info(
                "  %s: resampling %d -> %d Hz", label, sr, self.sample_rate
            )
            import librosa
            audio = librosa.resample(
                audio, orig_sr=sr, target_sr=self.sample_rate, axis=-1
            )

        return audio, self.sample_rate


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _ensure_stereo(arr: np.ndarray) -> np.ndarray:
    """
    Ensure audio array is stereo (2, samples).

    Mono (1, samples) is duplicated to both channels.
    Stereo (2, samples) is passed through.
    Multi-channel (>2) is averaged to mono then duplicated.
    """
    if arr.ndim == 1:
        arr = np.expand_dims(arr, axis=0)

    if arr.shape[0] == 1:
        # Mono → duplicate to stereo
        return np.concatenate([arr, arr], axis=0)
    elif arr.shape[0] == 2:
        # Already stereo
        return arr
    else:
        # Multi-channel → average to mono → stereo
        mono = np.mean(arr, axis=0, keepdims=True)
        return np.concatenate([mono, mono], axis=0)


def _peak_db(arr: np.ndarray) -> float:
    """Return peak amplitude in dB."""
    peak = np.max(np.abs(arr))
    if peak <= 0.0:
        return -float("inf")
    return 20.0 * np.log10(peak)


def _linear_to_db(x: float) -> float:
    """Convert linear amplitude ratio to dB."""
    if x <= 0.0:
        return -float("inf")
    return 20.0 * np.log10(x)
