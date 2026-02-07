import numpy as np
import ctranslate2
from pathlib import Path

model_path = None
for p in Path.home().rglob('.cache/huggingface/hub/models--Systran--faster-whisper-tiny/snapshots/*/model.bin'):
    model_path = str(p.parent)
    break

print(f"Model: {model_path}")

np.random.seed(12345)
features = np.random.randn(1, 80, 3000).astype(np.float32) * 0.1
features_sv = ctranslate2.StorageView.from_array(features)

truncated_dir_1 = Path(r'C:\Users\pinwa\projects\5700xt-rocm\dist\truncated_model_1layer')
if not (truncated_dir_1 / 'model.bin').exists():
    print("ERROR: Run test_bisect_encoder.py first to create truncated models!")
    exit(1)

print("\n=== 1-LAYER MODEL (CPU stock) ===")
model_1 = ctranslate2.models.Whisper(str(truncated_dir_1), device="cpu", compute_type="float32")
out_1 = model_1.encode(features_sv)
out_1_np = np.array(out_1)
print(f"  Shape: {out_1_np.shape}")
print(f"  mean={np.mean(out_1_np):.6f}, std={np.std(out_1_np):.6f}")
print(f"  [0,0,:10]: {out_1_np[0,0,:10]}")
np.save(r"C:\Users\pinwa\projects\5700xt-rocm\dist\encoder_1layer_cpu.npy", out_1_np)
del model_1

print("\n=== FULL MODEL (CPU stock) ===")
model_full = ctranslate2.models.Whisper(model_path, device="cpu", compute_type="float32")
out_full = model_full.encode(features_sv)
out_full_np = np.array(out_full)
print(f"  Shape: {out_full_np.shape}")
print(f"  mean={np.mean(out_full_np):.6f}, std={np.std(out_full_np):.6f}")
np.save(r"C:\Users\pinwa\projects\5700xt-rocm\dist\encoder_full_cpu_stock.npy", out_full_np)

gpu_1layer = Path(r'C:\Users\pinwa\projects\5700xt-rocm\dist\encoder_1layer_gpu.npy')
if gpu_1layer.exists():
    gpu_1_np = np.load(str(gpu_1layer))
    corr = np.corrcoef(gpu_1_np.flat, out_1_np.flat)[0, 1]
    diff = np.abs(gpu_1_np - out_1_np)
    print(f"\n=== GPU 1-layer vs CPU 1-layer ===")
    print(f"  corr={corr:.6f}, max_diff={np.max(diff):.4f}, mean_diff={np.mean(diff):.4f}")
    if corr > 0.99:
        print("  >>> GPU AND CPU MATCH FOR 1-LAYER! Conv1d + 1 transformer OK.")
        print("  >>> Bug is in layers 2-4 interaction or accumulation.")
    elif corr > 0.5:
        print("  >>> PARTIAL MATCH — 1st transformer layer has some GPU errors.")
    else:
        print("  >>> NO MATCH — GPU diverges even with just 1 transformer layer!")

    print(f"\n  GPU [0,0,:10]: {gpu_1_np[0,0,:10]}")
    print(f"  CPU [0,0,:10]: {out_1_np[0,0,:10]}")
