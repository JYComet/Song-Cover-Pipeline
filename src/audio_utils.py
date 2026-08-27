"""Shared, model-independent audio array operations."""

from __future__ import annotations

import numpy as np


def align_audio_pair(
    first: np.ndarray, second: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return two channel-first float32 arrays with compatible channels."""
    first = np.asarray(first, dtype=np.float32)
    second = np.asarray(second, dtype=np.float32)
    if first.ndim == 1:
        first = first[np.newaxis, :]
    if second.ndim == 1:
        second = second[np.newaxis, :]
    if first.ndim != 2 or second.ndim != 2:
        raise ValueError("Audio arrays must have shape (channels, samples)")

    if first.shape[0] != second.shape[0]:
        if first.shape[0] == 1:
            first = np.repeat(first, second.shape[0], axis=0)
        elif second.shape[0] == 1:
            second = np.repeat(second, first.shape[0], axis=0)
        else:
            raise ValueError(
                f"Incompatible channel counts: {first.shape[0]} and {second.shape[0]}"
            )
    return first, second


def compute_residuals(
    original: np.ndarray,
    vocals: np.ndarray,
    noreverb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute instrumental and reverb stems by direct subtraction.

    STFT is linear, so STFT -> subtract -> iSTFT is redundant for residuals.
    Direct subtraction avoids both transforms and reconstruction round-off.
    """
    original, vocals_for_mix = align_audio_pair(original, vocals)
    vocals, noreverb = align_audio_pair(vocals, noreverb)

    inst_len = min(original.shape[-1], vocals_for_mix.shape[-1])
    reverb_len = min(vocals.shape[-1], noreverb.shape[-1])
    instrumental = original[:, :inst_len] - vocals_for_mix[:, :inst_len]
    reverb = vocals[:, :reverb_len] - noreverb[:, :reverb_len]
    return (
        np.ascontiguousarray(instrumental, dtype=np.float32),
        np.ascontiguousarray(reverb, dtype=np.float32),
    )
