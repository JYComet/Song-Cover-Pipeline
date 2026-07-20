#!/usr/bin/env python3
"""
Batch De-Harmonization & De-Reverb Processing Script
批量去和声+去混响处理脚本

功能:
  1. 扫描 E:\Auburn (映射到 /mnt/local_E/Auburn) 中的所有音频文件
  2. 对每个文件执行和声分离(karaoke) + 混响分离(dereverb)
  3. 将结果分类输出到 E:\Auburn干声\:
     - 人声/    : 纯净干声 (去混响后的主唱人声)
     - 和声/    : 和声+伴奏 (Instrumental，包含背景和声)
     - 混响/    : 混响尾 (reverb tail)

Usage:
    python batch_deharm_reverb.py [--force] [--dry-run]
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# Ensure project root and src/ are importable
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("batch_deharm_reverb")

# ── Paths ──
INPUT_DIR = Path("/mnt/local_E/Auburn")
OUTPUT_BASE = Path("/mnt/local_E/Auburn干声")

# Subdirectories for classification
DIR_VOCALS = OUTPUT_BASE / "人声"        # dry clean vocals
DIR_HARMONY = OUTPUT_BASE / "和声"       # instrumental + harmony vocals
DIR_REVERB = OUTPUT_BASE / "混响"        # reverb tail

# Supported audio extensions
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".aiff", ".ape", ".wma", ".opus"}

# ── MSST model configs (same as ria_cover.yaml) ──
KARAOKE_CONFIG = {
    "msst_code_dir": "vendor/msst",
    "model_type": "bs_roformer",
    "config_path": "vendor/msst/configs/config_karaoke_frazer_becruily.yaml",
    "checkpoint_path": "models/MSST/bs_roformer_karaoke_frazer_becruily.ckpt",
    "device": "cuda:0",
    "chunk_batch": 16,
    "use_compile": True,
    "output_sample_rate": 44100,
}

DEREVERB_CONFIG = {
    "msst_code_dir": "vendor/msst",
    "model_type": "bs_roformer",
    "config_path": "vendor/msst/configs/dereverb_bs_roformer_anvuew.yaml",
    "checkpoint_path": "models/MSST/dereverb_bs_roformer_anvuew_sdr_22.5050.ckpt",
    "device": "cuda:0",
    "chunk_batch": 8,
    "use_compile": True,
    "output_sample_rate": 44100,
}


def resolve_path(rel_path: str) -> Path:
    """Resolve a path relative to PROJECT_ROOT."""
    p = Path(rel_path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def find_audio_files(input_dir: Path) -> list[Path]:
    """Recursively find all audio files in the input directory."""
    audio_files = []
    for f in sorted(input_dir.iterdir()):
        if f.is_file() and f.suffix.lower() in AUDIO_EXTS:
            audio_files.append(f)
        elif f.is_dir():
            audio_files.extend(find_audio_files(f))
    return audio_files


def load_audio(file_path: Path) -> tuple[np.ndarray, int]:
    """Load audio file, return (waveform, sample_rate). waveform shape: (channels, samples)."""
    logger.info("Loading: %s", file_path.name)
    audio, sr = sf.read(str(file_path))
    if audio.ndim == 1:
        audio = np.expand_dims(audio, axis=0)
    else:
        audio = audio.T  # (samples, channels) -> (channels, samples)
    return audio.astype(np.float32), sr


def process_batch():
    """Main batch processing function."""
    parser = argparse.ArgumentParser(
        description="批量去和声+去混响处理",
    )
    parser.add_argument("--force", "-f", action="store_true",
                        help="Force re-process even if output already exists")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only list files to process, don't run")
    args = parser.parse_args()

    # ── Validate paths ──
    for rel_path in [KARAOKE_CONFIG["config_path"], KARAOKE_CONFIG["checkpoint_path"],
                     DEREVERB_CONFIG["config_path"], DEREVERB_CONFIG["checkpoint_path"]]:
        p = resolve_path(rel_path)
        if not p.exists():
            logger.error("Required file not found: %s", p)
            sys.exit(1)

    if not INPUT_DIR.is_dir():
        logger.error("Input directory not found: %s", INPUT_DIR)
        sys.exit(1)

    # ── Create output directories ──
    for d in [DIR_VOCALS, DIR_HARMONY, DIR_REVERB]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Find audio files ──
    audio_files = find_audio_files(INPUT_DIR)
    logger.info("Found %d audio file(s) in %s", len(audio_files), INPUT_DIR)

    if not audio_files:
        logger.warning("No audio files found!")
        return

    if args.dry_run:
        logger.info("=== DRY RUN ===")
        for i, f in enumerate(audio_files, 1):
            logger.info("  %2d. %s", i, f.name)
        return

    # ── Filter already-processed files (unless --force) ──
    if not args.force:
        to_process = []
        skipped = 0
        for f in audio_files:
            stem = f.stem
            dry_output = DIR_VOCALS / f"{stem}_noreverb.wav"
            if dry_output.is_file():
                skipped += 1
            else:
                to_process.append(f)
        if skipped > 0:
            logger.info("Skipping %d already-processed file(s) (use --force to redo)", skipped)
        audio_files = to_process

    if not audio_files:
        logger.info("All files already processed. Done!")
        return

    logger.info("Processing %d audio file(s)...", len(audio_files))

    # ── Check GPU ──
    if not torch.cuda.is_available():
        logger.error("CUDA GPU not available! This script requires GPU for inference.")
        sys.exit(1)
    logger.info("GPU: %s (%.1f GB VRAM)", torch.cuda.get_device_name(0),
                torch.cuda.get_device_properties(0).total_memory / 1024**3)

    # ── Import MSST ──
    from src.msst_separator import MSSTSeparator

    # ── Load models (once for all files) ──
    logger.info("=" * 60)
    logger.info("Loading karaoke model (和声分离)...")
    karaoke = MSSTSeparator(
        msst_code_dir=resolve_path(KARAOKE_CONFIG["msst_code_dir"]),
        model_type=KARAOKE_CONFIG["model_type"],
        config_path=resolve_path(KARAOKE_CONFIG["config_path"]),
        checkpoint_path=resolve_path(KARAOKE_CONFIG["checkpoint_path"]),
        device=KARAOKE_CONFIG["device"],
        chunk_batch=KARAOKE_CONFIG["chunk_batch"],
        use_compile=KARAOKE_CONFIG["use_compile"],
    )

    logger.info("Loading dereverb model (混响分离)...")
    dereverb = MSSTSeparator(
        msst_code_dir=resolve_path(DEREVERB_CONFIG["msst_code_dir"]),
        model_type=DEREVERB_CONFIG["model_type"],
        config_path=resolve_path(DEREVERB_CONFIG["config_path"]),
        checkpoint_path=resolve_path(DEREVERB_CONFIG["checkpoint_path"]),
        device=DEREVERB_CONFIG["device"],
        chunk_batch=DEREVERB_CONFIG["chunk_batch"],
        use_compile=DEREVERB_CONFIG["use_compile"],
    )

    try:
        karaoke.load_model()
        dereverb.load_model()
        output_sr = KARAOKE_CONFIG["output_sample_rate"]

        # ── Process each file sequentially ──
        # Approach: karaoke → keep vocals in RAM → dereverb → STFT residuals
        # This avoids the LinkedMSSTSeparator's STFT pass-through, which has
        # flash attention kernel compatibility issues with torch.compile.
        total_start = time.time()
        success_count = 0
        fail_count = 0

        for idx, audio_path in enumerate(audio_files, 1):
            song_name = audio_path.stem
            logger.info("=" * 60)
            logger.info("[%d/%d] Processing: %s", idx, len(audio_files), audio_path.name)
            logger.info("=" * 60)

            try:
                t0 = time.time()

                # Load audio → (channels, samples) float32
                audio, sr = load_audio(audio_path)
                audio_dur = audio.shape[-1] / sr
                logger.info("Duration: %.1fs, shape=%s, sr=%d", audio_dur, audio.shape, sr)

                # ── Step 1: Karaoke separation (和声分离) ──
                logger.info("  [1/3] Karaoke separation...")
                karaoke_waveforms = karaoke.separate(audio, sr, "Vocals")
                vocals = karaoke_waveforms["Vocals"]  # (ch, samples) — keep in RAM

                # ── Step 2: Dereverb separation on vocals (混响分离) ──
                logger.info("  [2/3] Dereverb separation...")
                dereverb_waveforms = dereverb.separate(vocals, sr, "noreverb")
                noreverb = dereverb_waveforms["noreverb"]  # (ch, samples) — dry vocals

                # ── Step 3: Residuals (STFT domain, fallback to time domain) ──
                logger.info("  [3/3] Computing residuals (STFT domain)...")
                try:
                    instrumental, reverb = _compute_residuals_stft(
                        audio, vocals, noreverb, sr,
                    )
                except Exception as stft_err:
                    logger.warning("  STFT residual failed, falling back to time domain: %s", stft_err)
                    instrumental, reverb = _compute_residuals_time(
                        audio, vocals, noreverb,
                    )

                # ── Save to category directories ──
                save_wav(noreverb, DIR_VOCALS / f"{song_name}_noreverb.wav", output_sr)
                logger.info("  ✅ 人声(干声): %s_noreverb.wav", song_name)

                save_wav(instrumental, DIR_HARMONY / f"{song_name}_Instrumental.wav", output_sr)
                logger.info("  ✅ 和声+伴奏: %s_Instrumental.wav", song_name)

                save_wav(reverb, DIR_REVERB / f"{song_name}_reverb.wav", output_sr)
                logger.info("  ✅ 混响: %s_reverb.wav", song_name)

                elapsed = time.time() - t0
                logger.info("  ⏱️  Completed in %.1fs (%.1fx realtime)", elapsed, audio_dur / max(elapsed, 0.001))
                success_count += 1

            except Exception as e:
                logger.error("  ❌ Failed: %s", e, exc_info=True)
                fail_count += 1
                # Clear GPU cache after error
                torch.cuda.empty_cache()
                continue

        # ── Summary ──
        total_elapsed = time.time() - total_start
        logger.info("=" * 60)
        logger.info("BATCH COMPLETE")
        logger.info("  Total: %d | Success: %d | Failed: %d",
                    len(audio_files), success_count, fail_count)
        logger.info("  Total time: %.1fs (%.1f min)", total_elapsed, total_elapsed / 60)
        logger.info("  Output: %s", OUTPUT_BASE)
        logger.info("    - 人声/: %d files", len(list(DIR_VOCALS.glob("*.wav"))))
        logger.info("    - 和声/: %d files", len(list(DIR_HARMONY.glob("*.wav"))))
        logger.info("    - 混响/: %d files", len(list(DIR_REVERB.glob("*.wav"))))
        logger.info("=" * 60)

    finally:
        # Clean up
        karaoke.unload_model()
        dereverb.unload_model()


def save_wav(waveform: np.ndarray, path: Path, sample_rate: int):
    """Save waveform to WAV file. waveform shape: (channels, samples)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), waveform.T, sample_rate, format="WAV", subtype="FLOAT")


def _compute_residuals_stft(
    original: np.ndarray,
    vocals: np.ndarray,
    noreverb: np.ndarray,
    sample_rate: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: int = 2048,
) -> tuple:
    """
    Compute instrumental and reverb residuals in the STFT domain.

    instrumental = original - vocals   (complex STFT domain)
    reverb       = vocals - noreverb   (complex STFT domain)

    Returns (instrumental, reverb) as numpy arrays.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    window = torch.hann_window(win_length, device=device)

    def _stft(audio_np):
        audio_t = torch.from_numpy(audio_np.astype(np.float32)).to(device)
        stfts = []
        for ch in range(audio_t.shape[0]):
            s = torch.stft(
                audio_t[ch],
                n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                window=window, return_complex=True,
            )
            stfts.append(s)
        return stfts

    def _istft(stft_list, length):
        audios = []
        for s in stft_list:
            a = torch.istft(
                s, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                window=window, length=length,
            )
            audios.append(a)
        return torch.stack(audios, dim=0)

    stft_orig = _stft(original)
    stft_voc = _stft(vocals)
    stft_dry = _stft(noreverb)

    # Align to minimum time frames
    min_frames = min(s.shape[-1] for s in stft_orig + stft_voc + stft_dry)
    stft_orig = [s[..., :min_frames] for s in stft_orig]
    stft_voc = [s[..., :min_frames] for s in stft_voc]
    stft_dry = [s[..., :min_frames] for s in stft_dry]

    # Residuals in STFT domain (complex subtraction)
    stft_inst = [o - v for o, v in zip(stft_orig, stft_voc)]
    stft_rev = [v - d for v, d in zip(stft_voc, stft_dry)]

    instrumental = _istft(stft_inst, original.shape[-1]).cpu().numpy()
    reverb = _istft(stft_rev, vocals.shape[-1]).cpu().numpy()
    return instrumental.astype(np.float32), reverb.astype(np.float32)


def _compute_residuals_time(
    original: np.ndarray,
    vocals: np.ndarray,
    noreverb: np.ndarray,
) -> tuple:
    """
    Compute instrumental and reverb residuals in the time domain.

    Fallback when STFT-domain computation fails (e.g. iSTFT window overlap issues).
    instrumental = original - vocals
    reverb       = vocals - noreverb
    """
    # Align channel counts
    def _match_channels(ref, other):
        if ref.shape[0] != other.shape[0]:
            if ref.shape[0] == 1:
                return np.repeat(other, ref.shape[0], axis=0)
            elif other.shape[0] == 1:
                return np.repeat(ref, other.shape[0], axis=0)
            else:
                return other[:ref.shape[0], :]
        return other

    vocals_m = _match_channels(original, vocals)
    noreverb_m = _match_channels(original, noreverb)

    # Align lengths
    min_len_inst = min(original.shape[-1], vocals_m.shape[-1])
    min_len_rev = min(vocals_m.shape[-1], noreverb_m.shape[-1])

    instrumental = original[:, :min_len_inst] - vocals_m[:, :min_len_inst]
    reverb = vocals_m[:, :min_len_rev] - noreverb_m[:, :min_len_rev]

    return instrumental.astype(np.float32), reverb.astype(np.float32)


def _copy_file(src: Path, dst: Path):
    """Copy file from src to dst."""
    import shutil
    if dst.exists():
        dst.unlink()
    shutil.copy2(str(src), str(dst))


if __name__ == "__main__":
    process_batch()
