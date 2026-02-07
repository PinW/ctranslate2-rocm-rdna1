import os
import sys
import numpy as np

from pathlib import Path
rocm_base = Path(r'C:\Program Files\AMD\ROCm')
if rocm_base.exists():
    for version_dir in sorted(rocm_base.iterdir(), reverse=True):
        rocm_bin = version_dir / 'bin'
        if rocm_bin.exists() and (rocm_bin / 'amdhip64_6.dll').exists():
            os.add_dll_directory(str(rocm_bin))
            break

import ctranslate2

model_path = None
for p in Path.home().rglob('.cache/huggingface/hub/models--Systran--faster-whisper-tiny/snapshots/*/model.bin'):
    model_path = str(p.parent)
    break

print(f"Model: {model_path}")
model = ctranslate2.models.Whisper(model_path, device="cuda", compute_type="float32")
print("Loaded on GPU")

np.random.seed(12345)
features = np.random.randn(1, 80, 3000).astype(np.float32) * 0.1
features_sv = ctranslate2.StorageView.from_array(features)

print(f"\nInput: {features_sv.shape}")
print("Encoding on GPU...")
output_gpu = model.encode(features_sv)
print(f"Output: {output_gpu.shape}, device={output_gpu.device}")

# Move to CPU to get numpy array
output_cpu_sv = output_gpu.to_device(ctranslate2.Device.cpu)
output_np = np.array(output_cpu_sv)
print(f"Numpy shape: {output_np.shape}, dtype: {output_np.dtype}")

print(f"\n=== GPU Encoder Output Stats ===")
print(f"  mean:    {np.mean(output_np):.6f}")
print(f"  std:     {np.std(output_np):.6f}")
print(f"  min:     {np.min(output_np):.6f}")
print(f"  max:     {np.max(output_np):.6f}")
print(f"  has NaN: {np.any(np.isnan(output_np))}")
print(f"  has Inf: {np.any(np.isinf(output_np))}")
print(f"  abs_mean: {np.mean(np.abs(output_np)):.6f}")

# Check a few rows for consistency
print(f"\n=== Sample rows ===")
for i in [0, 750, 1499]:
    row = output_np[0, i, :]
    print(f"  row {i}: mean={np.mean(row):.4f}, std={np.std(row):.4f}, "
          f"min={np.min(row):.4f}, max={np.max(row):.4f}")

print(f"\n  [0, 0, :10]:    {output_np[0, 0, :10]}")
print(f"  [0, 0, -10:]:   {output_np[0, 0, -10:]}")
print(f"  [0, 750, :10]:  {output_np[0, 750, :10]}")
print(f"  [0, 1499, :10]: {output_np[0, 1499, :10]}")

np.save("encoder_output_gpu.npy", output_np)
print(f"\nSaved encoder_output_gpu.npy ({output_np.nbytes / 1024:.0f} KB)")

# Now test: use this encoder output to detect language
# This tells us if the encoder output is valid by seeing if the decoder can use it
print(f"\n=== Language detection with this encoder output ===")
silence = np.zeros(16000 * 3, dtype=np.float32)
lang, prob, all_probs = ctranslate2.models.Whisper.detect_language(model, features_sv)
top5 = sorted(all_probs, key=lambda x: x[1], reverse=True)[:5]
print(f"  Result: {lang} (prob={prob:.3f})")
for l, p in top5:
    print(f"    {l}: {p:.4f}")
