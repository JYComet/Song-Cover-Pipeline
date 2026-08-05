#!/usr/bin/env python3
"""
Direct timbre conversion — no separation stages.
Usage:
    python run_timbre_only.py
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ddsp_converter import DDSPConverter

# Input / output
input_file = PROJECT_ROOT / "input/0227 AI REF LEAD VOX.mp3"
output_dir = PROJECT_ROOT / "output/0227_cover/timbre_only"
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / f"{input_file.stem}_converted.wav"

print("=" * 60)
print("  Timbre-Only Conversion (仅音色替换)")
print("=" * 60)
print(f"  Input:  {input_file}")
print(f"  Model:  models/DDSP/paipai.pt")
print(f"  Output: {output_file}")
print("=" * 60)
print()

converter = DDSPConverter(
    ddsp_project_dir="vendor/ddsp",
    model_ckpt="models/DDSP/paipai.pt",
    device="cuda:0",
    infer_step=100,
    t_start=0.4,
    method="euler",
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

t_start = time.time()
try:
    converter.load_model()
    converter.convert(input_path=input_file, output_path=output_file)
finally:
    converter.unload_model()

elapsed = time.time() - t_start
print()
print(f"✅ Done in {elapsed:.1f}s")
print(f"   Output: {output_file}")
