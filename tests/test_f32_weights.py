import os, struct, shutil
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
import ctypes

hip_dll = ctypes.CDLL(str(next(
    p / 'bin' / 'amdhip64_6.dll'
    for p in sorted(Path(r'C:\Program Files\AMD\ROCm').iterdir(), reverse=True)
    if (p / 'bin' / 'amdhip64_6.dll').exists()
)))
hip_dll.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
hip_dll.hipMemcpy.restype = ctypes.c_int


def sv_to_numpy(sv):
    shape = list(sv.shape)
    total = 1
    for s in shape:
        total *= s
    out = np.empty(total, dtype=np.float32)
    repr_str = repr(sv)
    if 'cuda' in repr_str:
        gpu_ptr_line = repr_str.split('\n')[-1] if '\n' in repr_str else repr_str
    # Try np.array first (works if StorageView supports __array__ on CPU)
    try:
        arr = np.array(sv)
        if arr.ndim > 0 and arr.shape == tuple(shape):
            return arr
    except:
        pass
    # Fallback: encode to features, generate, or save trick
    # The StorageView print shows values — parse them
    # Actually, use CTranslate2's internal copy mechanism
    cpu_sv = ctranslate2.StorageView.from_array(np.zeros(shape, dtype=np.float32))
    # We can't directly copy GPU→CPU without the right API
    # Let's try a different approach: run on CPU and compare
    return None


model_path = None
for p in Path(r'C:\Users\pinwa\.cache\huggingface\hub').rglob('models--Systran--faster-whisper-tiny/snapshots/*/model.bin'):
    model_path = str(p.parent)
    break

np.random.seed(12345)
features = np.random.randn(1, 80, 3000).astype(np.float32) * 0.1
features_sv = ctranslate2.StorageView.from_array(features)

# Try the simplest approach: let CTranslate2 handle GPU→CPU internally
print("=" * 60)
print("TEST: Does GPU StorageView auto-transfer to numpy?")
print("=" * 60)

model = ctranslate2.models.Whisper(str(model_path), device="cuda", compute_type="float32")
out = model.encode(features_sv)
print(f"  Shape: {list(out.shape)}, device: {out.device}, dtype: {out.dtype}")

# Method 1: Direct np.array
try:
    arr = np.array(out)
    print(f"  np.array: shape={arr.shape}, dtype={arr.dtype}")
    if arr.ndim > 0:
        print(f"  SUCCESS: {arr[0,0,:5]}")
        np.save(r'C:\Users\pinwa\projects\5700xt-rocm\dist\gpu_output_direct.npy', arr)
except Exception as e:
    print(f"  np.array failed: {e}")

# Method 2: numpy() method
for method in ['numpy', 'to_numpy', 'cpu', 'detach']:
    if hasattr(out, method):
        try:
            result = getattr(out, method)()
            print(f"  .{method}(): {type(result)}")
        except Exception as e:
            print(f"  .{method}() failed: {e}")

# Method 3: to_device
try:
    cpu_out = out.to_device(0)
    print(f"  to_device(0): device={cpu_out.device}")
    arr = np.array(cpu_out)
    print(f"  then np.array: shape={arr.shape}, dtype={arr.dtype}")
except Exception as e:
    print(f"  to_device(0) failed: {e}")

try:
    cpu_out = out.to_device(-1)
    print(f"  to_device(-1): device={cpu_out.device}")
except Exception as e:
    print(f"  to_device(-1) failed: {e}")

# Method 4: Run on CPU instead
print("\n" + "=" * 60)
print("RUNNING ON CPU (custom build)")
print("=" * 60)
del model
try:
    model_cpu = ctranslate2.models.Whisper(str(model_path), device="cpu", compute_type="float32")
    out_cpu = model_cpu.encode(features_sv)
    out_cpu_np = np.array(out_cpu)
    print(f"  CPU Shape: {out_cpu_np.shape}")
    print(f"  CPU [0,0,:5]: {out_cpu_np[0,0,:5]}")
    np.save(r'C:\Users\pinwa\projects\5700xt-rocm\dist\gpu_f32_cpu_encode.npy', out_cpu_np)

    cpu_ref = np.load(r'C:\Users\pinwa\projects\5700xt-rocm\encoder_output_cpu.npy')
    corr = np.corrcoef(out_cpu_np.flat, cpu_ref.flat)[0, 1]
    diff = np.abs(out_cpu_np - cpu_ref)
    print(f"  Custom CPU vs stock CPU: corr={corr:.6f}, max_diff={np.max(diff):.4f}")
    del model_cpu
except Exception as e:
    print(f"  CPU FAILED: {e}")
    import traceback; traceback.print_exc()

# Method 5: Brute force - use hipMemcpy to copy GPU StorageView data
print("\n" + "=" * 60)
print("BRUTE FORCE: hipMemcpy from GPU StorageView")
print("=" * 60)
model = ctranslate2.models.Whisper(str(model_path), device="cuda", compute_type="float32")
out = model.encode(features_sv)

# The StorageView repr contains the values - let's try to get the pointer
# StorageView should support buffer protocol or have a data_ptr
for attr in ['data_ptr', 'data', 'ctypes', '__cuda_array_interface__', '__array_interface__']:
    if hasattr(out, attr):
        print(f"  Has {attr}: {getattr(out, attr)}")

# Try __cuda_array_interface__
try:
    iface = out.__cuda_array_interface__
    print(f"  __cuda_array_interface__: {iface}")
    gpu_ptr = iface['data'][0]
    shape = iface['shape']
    total = 1
    for s in shape:
        total *= s
    host_arr = np.empty(shape, dtype=np.float32)
    status = hip_dll.hipMemcpy(
        ctypes.c_void_p(host_arr.ctypes.data),
        ctypes.c_void_p(gpu_ptr),
        total * 4,
        2  # hipMemcpyDeviceToHost
    )
    print(f"  hipMemcpy status: {status}")
    print(f"  Result shape: {host_arr.shape}")
    print(f"  [0,0,:5]: {host_arr[0,0,:5]}")
    np.save(r'C:\Users\pinwa\projects\5700xt-rocm\dist\gpu_output_hipMemcpy.npy', host_arr)

    cpu_ref = np.load(r'C:\Users\pinwa\projects\5700xt-rocm\encoder_output_cpu.npy')
    corr = np.corrcoef(host_arr.flat, cpu_ref.flat)[0, 1]
    diff = np.abs(host_arr - cpu_ref)
    print(f"\n  GPU vs stock CPU: corr={corr:.6f}, max_diff={np.max(diff):.4f}, mean_diff={np.mean(diff):.4f}")
    if corr > 0.99:
        print("  >>> GPU ENCODER IS CORRECT!")
    else:
        print("  >>> GPU ENCODER IS STILL WRONG")
except AttributeError:
    print("  No __cuda_array_interface__ - trying DLPack")
    try:
        dl = out.__dlpack__()
        print(f"  __dlpack__: {type(dl)}")
    except Exception as e2:
        print(f"  No DLPack either: {e2}")
        print("  Cannot extract GPU data - need to use CTranslate2 GPU→CPU transfer")

del model
print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
