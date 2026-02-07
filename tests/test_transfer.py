import os
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

print("=== Test StorageView CPU <-> GPU transfer ===\n")

# Test 1: Small array
np.random.seed(42)
a = np.random.randn(3, 4).astype(np.float32)
sv_cpu = ctranslate2.StorageView.from_array(a)
print(f"Original: shape={sv_cpu.shape}, device={sv_cpu.device}")
print(f"  values: {a}")

sv_gpu = sv_cpu.to_device(ctranslate2.Device.cuda)
print(f"On GPU: shape={sv_gpu.shape}, device={sv_gpu.device}")

sv_back = sv_gpu.to_device(ctranslate2.Device.cpu)
b = np.array(sv_back)
print(f"Back on CPU: shape={b.shape}")
print(f"  values: {b}")

match = np.allclose(a, b)
print(f"  Match: {match}")
if not match:
    print(f"  max_diff: {np.max(np.abs(a - b))}")

# Test 2: Mel-spectrogram sized array
print()
np.random.seed(12345)
features = np.random.randn(1, 80, 3000).astype(np.float32) * 0.1

sv_f_cpu = ctranslate2.StorageView.from_array(features)
sv_f_gpu = sv_f_cpu.to_device(ctranslate2.Device.cuda)
sv_f_back = sv_f_gpu.to_device(ctranslate2.Device.cpu)
features_back = np.array(sv_f_back)

match2 = np.allclose(features, features_back)
print(f"Mel features roundtrip: match={match2}")
if not match2:
    diff = np.abs(features - features_back)
    print(f"  max_diff: {np.max(diff)}")
    print(f"  mean_diff: {np.mean(diff)}")
else:
    print(f"  Perfect roundtrip for {features.size} float32 values")

# Test 3: Test that the features going INTO the encoder match
print("\n=== Test: What features does the encoder actually receive? ===")

model_path = None
for p in Path.home().rglob('.cache/huggingface/hub/models--Systran--faster-whisper-tiny/snapshots/*/model.bin'):
    model_path = str(p.parent)
    break

model = ctranslate2.models.Whisper(model_path, device="cuda", compute_type="float32")

# Encode with CPU features (auto-transferred to GPU)
output1 = model.encode(sv_f_cpu)
out1 = np.array(output1.to_device(ctranslate2.Device.cpu))

# Encode with GPU features (already on GPU)
output2 = model.encode(sv_f_gpu)
out2 = np.array(output2.to_device(ctranslate2.Device.cpu))

match3 = np.allclose(out1, out2)
print(f"CPU features vs GPU features into encoder: match={match3}")
if not match3:
    diff = np.abs(out1 - out2)
    print(f"  max_diff: {np.max(diff)}")
    print(f"  Note: Both should give same result if transfer works")
else:
    print(f"  Same result regardless of input device")

# Test 4: Encode twice with same input - deterministic?
output3 = model.encode(sv_f_cpu)
out3 = np.array(output3.to_device(ctranslate2.Device.cpu))

match4 = np.allclose(out1, out3)
print(f"\nDeterministic: same input twice gives same output: {match4}")
if not match4:
    diff = np.abs(out1 - out3)
    print(f"  max_diff: {np.max(diff)}")
