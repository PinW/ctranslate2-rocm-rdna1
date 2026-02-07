import os

from pathlib import Path
rocm_base = Path(r'C:\Program Files\AMD\ROCm')
if rocm_base.exists():
    for version_dir in sorted(rocm_base.iterdir(), reverse=True):
        rocm_bin = version_dir / 'bin'
        if rocm_bin.exists() and (rocm_bin / 'amdhip64_6.dll').exists():
            os.add_dll_directory(str(rocm_bin))
            break

import numpy as np
from faster_whisper import WhisperModel

model = WhisperModel("tiny", device="cuda", compute_type="float32")
print("Model loaded on GPU\n")

silence = np.zeros(16000 * 3, dtype=np.float32)
noise = np.random.randn(16000 * 3).astype(np.float32) * 0.001

print("=== Encoder sanity: language detection ===")
print("(Silence/noise should give low-confidence results, NOT confident wrong languages)\n")

for label, audio in [("silence", silence), ("weak noise", noise)]:
    lang, prob, all_probs = model.detect_language(audio)
    top5 = sorted(all_probs, key=lambda x: x[1], reverse=True)[:5]
    print(f"[{label}]")
    print(f"  detected: {lang} (prob={prob:.3f})")
    for l, p in top5:
        print(f"    {l}: {p:.4f}")
    print()

print("=== CPU comparison (stock ctranslate2) ===")
try:
    model_cpu = WhisperModel("tiny", device="cpu", compute_type="float32")
    for label, audio in [("silence", silence), ("weak noise", noise)]:
        lang, prob, all_probs = model_cpu.detect_language(audio)
        top5 = sorted(all_probs, key=lambda x: x[1], reverse=True)[:5]
        print(f"[{label} CPU] detected: {lang} (prob={prob:.3f})")
        for l, p in top5:
            print(f"    {l}: {p:.4f}")
        print()
except Exception as e:
    print(f"CPU failed (no SGEMM backend): {e}")
