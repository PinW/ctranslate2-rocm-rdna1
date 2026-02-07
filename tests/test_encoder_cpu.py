import numpy as np
import ctranslate2

from pathlib import Path

model_path = None
for p in Path.home().rglob('.cache/huggingface/hub/models--Systran--faster-whisper-tiny/snapshots/*/model.bin'):
    model_path = str(p.parent)
    break

print(f"Model: {model_path}")
model = ctranslate2.models.Whisper(model_path, device="cpu", compute_type="float32")
print("Loaded on CPU")

np.random.seed(12345)
features = np.random.randn(1, 80, 3000).astype(np.float32) * 0.1
features_sv = ctranslate2.StorageView.from_array(features)

print(f"\nInput: {features_sv.shape}")
print("Encoding on CPU...")
output = model.encode(features_sv)
output_np = np.array(output)
print(f"Output: {output_np.shape}")

print(f"\n=== CPU Encoder Output Stats ===")
print(f"  mean:    {np.mean(output_np):.6f}")
print(f"  std:     {np.std(output_np):.6f}")
print(f"  min:     {np.min(output_np):.6f}")
print(f"  max:     {np.max(output_np):.6f}")

print(f"\n  [0, 0, :10]:    {output_np[0, 0, :10]}")
print(f"  [0, 750, :10]:  {output_np[0, 750, :10]}")
print(f"  [0, 1499, :10]: {output_np[0, 1499, :10]}")

np.save(r"C:\Users\pinwa\projects\5700xt-rocm\encoder_output_cpu.npy", output_np)
print(f"\nSaved encoder_output_cpu.npy")

# Compare with GPU output if available
gpu_path = Path(r"C:\Users\pinwa\projects\5700xt-rocm\encoder_output_gpu.npy")
if gpu_path.exists():
    gpu_np = np.load(gpu_path)
    print(f"\n=== GPU vs CPU Comparison ===")
    diff = np.abs(gpu_np - output_np)
    print(f"  max_diff:  {np.max(diff):.6f}")
    print(f"  mean_diff: {np.mean(diff):.6f}")
    print(f"  median_diff: {np.median(diff):.6f}")

    rel_diff = diff / (np.abs(output_np) + 1e-8)
    print(f"  max_rel_diff:  {np.max(rel_diff):.6f}")
    print(f"  mean_rel_diff: {np.mean(rel_diff):.6f}")

    # Check per-position
    print(f"\n  GPU [0,0,:10]:  {gpu_np[0, 0, :10]}")
    print(f"  CPU [0,0,:10]:  {output_np[0, 0, :10]}")
    print(f"  diff[0,0,:10]:  {diff[0, 0, :10]}")

    # Check correlation
    from numpy import corrcoef
    r = corrcoef(gpu_np.flat, output_np.flat)[0, 1]
    print(f"\n  Pearson correlation: {r:.6f}")
    print(f"  (1.0 = identical, 0.0 = uncorrelated, <0 = anti-correlated)")
else:
    print("\nNo GPU output found for comparison")
