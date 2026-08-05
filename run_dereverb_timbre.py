#!/usr/bin/env python3
"""
Step 1: Reverb separation (dereverb) → dry vocals
Step 2: Timbre conversion on dry vocals (infer_step=200, method=rk4)
"""

import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Config ──────────────────────────────────────────────────────────
INPUT_FILE = PROJECT_ROOT / "input/0227 AI REF LEAD VOX.mp3"
OUTPUT_DIR = PROJECT_ROOT / "output/0227_cover/dereverb_timbre"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_NAME = INPUT_FILE.stem

# MSST dereverb model paths
MSST_CODE_DIR = PROJECT_ROOT / "vendor/msst"
DEREVERB_CONFIG = PROJECT_ROOT / "vendor/msst/configs/dereverb_bs_roformer_anvuew.yaml"
DEREVERB_CKPT = PROJECT_ROOT / "models/MSST/dereverb_bs_roformer_anvuew_sdr_22.5050.ckpt"

# DDSP model
DDSP_DIR = PROJECT_ROOT / "vendor/ddsp"
MODEL_CKPT = PROJECT_ROOT / "models/DDSP/paipai.pt"

print("=" * 60)
print("  Dereverb + Timbre Conversion")
print("=" * 60)
print(f"  Input:       {INPUT_FILE}")
print(f"  Dereverb:    {DEREVERB_CKPT.name}")
print(f"  Timbre model:{MODEL_CKPT.name}")
print(f"  Output dir:  {OUTPUT_DIR}")
print(f"  infer_step=200, method=rk4")
print("=" * 60)
print()

# ══════════════════════════════════════════════════════════════════════
# Step 1: Reverb separation
# ══════════════════════════════════════════════════════════════════════
print("─" * 40)
print("  Step 1: Reverb Separation (混响分离)")
print("─" * 40)

from src.msst_separator import MSSTSeparator

dereverb = MSSTSeparator(
    msst_code_dir=MSST_CODE_DIR,
    model_type="bs_roformer",
    config_path=DEREVERB_CONFIG,
    checkpoint_path=DEREVERB_CKPT,
    device="cuda:0",
    chunk_batch=8,
    use_compile=True,
)

t1 = time.time()
try:
    dereverb.load_model()

    # Load audio
    audio, sr = sf.read(str(INPUT_FILE))
    if audio.ndim == 1:
        audio = np.expand_dims(audio, axis=0)
    else:
        audio = audio.T  # (samples, ch) → (ch, samples)
    audio = audio.astype(np.float32)

    # Run separation: extract dry vocals + reverb
    saved = dereverb.separate_to_file(
        input_path=INPUT_FILE,
        output_dir=OUTPUT_DIR,
        target_stem="noreverb",
        other_stems=["reverb"],
        output_sample_rate=44100,
    )
    dry_vocals_path = saved["noreverb"]
    reverb_path = saved.get("reverb")
    print(f"  Dry vocals: {dry_vocals_path}")
    if reverb_path:
        print(f"  Reverb:     {reverb_path}")

finally:
    dereverb.unload_model()

t1_elapsed = time.time() - t1
print(f"  Step 1 done in {t1_elapsed:.1f}s")
print()

# ══════════════════════════════════════════════════════════════════════
# Step 2: Timbre conversion
# ══════════════════════════════════════════════════════════════════════
print("─" * 40)
print("  Step 2: Timbre Conversion (音色替换)")
print("  infer_step=200, method=rk4")
print("─" * 40)

from src.ddsp_converter import DDSPConverter

converter = DDSPConverter(
    ddsp_project_dir=DDSP_DIR,
    model_ckpt=MODEL_CKPT,
    device="cuda:0",
    infer_step=200,
    t_start=0.4,
    method="rk4",
    pitch_extractor="fcpe",
    key=0,
    vocal_register_shift=0,
    formant_shift=0,
    f0_min=65,
    f0_max=800,
    threshold=-60,
    spk_id=1,
    use_compile=False,
    use_amp=False,
    segment_batch_size=1,
)

output_file = OUTPUT_DIR / f"{BASE_NAME}_noreverb_converted.wav"

t2 = time.time()
try:
    converter.load_model()
    converter.convert(input_path=dry_vocals_path, output_path=output_file)
finally:
    converter.unload_model()

t2_elapsed = time.time() - t2
print(f"  Step 2 done in {t2_elapsed:.1f}s")
print()

# ══════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════
total = time.time() - t1
print("=" * 60)
print("  Done ✅")
print(f"  Step 1 (dereverb):      {t1_elapsed:.1f}s")
print(f"  Step 2 (timbre, rk4):   {t2_elapsed:.1f}s")
print(f"  Total:                  {total:.1f}s")
print(f"  Output: {output_file}")
print("=" * 60)
