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

# Load on GPU
model = ctranslate2.models.Whisper(model_path, device="cuda", compute_type="float32")
print("Model loaded on GPU")

# Load CPU reference encoder output
cpu_file = Path(r"C:\Users\pinwa\projects\5700xt-rocm\encoder_output_cpu.npy")
if not cpu_file.exists():
    print("ERROR: encoder_output_cpu.npy not found! Run test_encoder_cpu.py first.")
    sys.exit(1)

cpu_output = np.load(str(cpu_file))
print(f"CPU encoder output: shape={cpu_output.shape}")

# Test 1: Feed CPU encoder output to GPU decoder for language detection
print("\n=== Test 1: GPU decoder with CORRECT (CPU) encoder output ===")
cpu_sv = ctranslate2.StorageView.from_array(cpu_output)
print(f"Created StorageView: shape={cpu_sv.shape}, device={cpu_sv.device}")

# detect_language takes features, not encoder output
# Let's use the lower-level generate method instead
# First, let's see what methods are available
methods = [m for m in dir(model) if not m.startswith('_')]
print(f"Model methods: {methods}")

# Try to detect language using the model's detect_language
np.random.seed(12345)
features = np.random.randn(1, 80, 3000).astype(np.float32) * 0.1
features_sv = ctranslate2.StorageView.from_array(features)

print("\n=== Test 2: GPU encoder + GPU decoder (our broken pipeline) ===")
result_gpu = model.detect_language(features_sv)
print(f"Result type: {type(result_gpu)}")
if isinstance(result_gpu, list):
    for item in result_gpu[:1]:
        sorted_probs = sorted(item, key=lambda x: x[1], reverse=True)[:5]
        print(f"  Top 5: {sorted_probs}")
else:
    print(f"  Result: {result_gpu}")

# Test 3: Check if we can pass encoder output directly to generate
print("\n=== Test 3: Try to use CPU encoder output with GPU decoder ===")
try:
    # The Whisper model's generate method takes encoder output
    # Let's try feeding the correct CPU encoder output
    prompt = ctranslate2.StorageView.from_array(
        np.array([[50258, 50259, 50359]], dtype=np.int32)  # <|startoftranscript|> <|en|> <|transcribe|>
    )

    # GPU encoder output (wrong)
    gpu_output = model.encode(features_sv)
    gpu_output_cpu = gpu_output.to_device(ctranslate2.Device.cpu)
    gpu_np = np.array(gpu_output_cpu)

    # Try generating with GPU encoder output
    print("Generating with GPU encoder output...")
    gen_result = model.generate(gpu_output, prompt, max_length=5)
    print(f"  GPU encoder -> decoder: {gen_result}")

    # Try generating with CPU encoder output
    print("Generating with CPU encoder output...")
    gen_result2 = model.generate(cpu_sv, prompt, max_length=5)
    print(f"  CPU encoder -> decoder: {gen_result2}")

except Exception as e:
    import traceback
    print(f"  Error: {e}")
    traceback.print_exc()
