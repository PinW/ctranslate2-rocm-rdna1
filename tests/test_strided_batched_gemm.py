import ctypes
import numpy as np
from pathlib import Path

hip_dll = None
hipblas_dll = None
for rocm_dir in sorted(Path(r'C:\Program Files\AMD\ROCm').iterdir(), reverse=True):
    hip_path = rocm_dir / 'bin' / 'amdhip64_6.dll'
    hipblas_path = rocm_dir / 'bin' / 'hipblas.dll'
    if hip_path.exists() and hipblas_path.exists():
        hip_dll = ctypes.CDLL(str(hip_path))
        hipblas_dll = ctypes.CDLL(str(hipblas_path))
        break

assert hip_dll and hipblas_dll, "Could not load HIP/hipBLAS"

hipMalloc = hip_dll.hipMalloc
hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
hipMalloc.restype = ctypes.c_int
hip_dll.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
hip_dll.hipMemcpy.restype = ctypes.c_int
hipFree = hip_dll.hipFree
hipFree.argtypes = [ctypes.c_void_p]
hipDeviceSynchronize = hip_dll.hipDeviceSynchronize

handle = ctypes.c_void_p()
hipblas_dll.hipblasCreate(ctypes.byref(handle))

SgemmStridedBatched = hipblas_dll.hipblasSgemmStridedBatched
SgemmStridedBatched.restype = ctypes.c_int
SgemmStridedBatched.argtypes = [
    ctypes.c_void_p,   # handle
    ctypes.c_int,      # transa
    ctypes.c_int,      # transb
    ctypes.c_int,      # m
    ctypes.c_int,      # n
    ctypes.c_int,      # k
    ctypes.POINTER(ctypes.c_float),  # alpha
    ctypes.c_void_p,   # A
    ctypes.c_int,      # lda
    ctypes.c_longlong,  # strideA
    ctypes.c_void_p,   # B
    ctypes.c_int,      # ldb
    ctypes.c_longlong,  # strideB
    ctypes.POINTER(ctypes.c_float),  # beta
    ctypes.c_void_p,   # C
    ctypes.c_int,      # ldc
    ctypes.c_longlong,  # strideC
    ctypes.c_int,      # batchCount
]

HIPBLAS_OP_N = 111
HIPBLAS_OP_T = 112

def alloc_gpu(arr):
    ptr = ctypes.c_void_p()
    hipMalloc(ctypes.byref(ptr), arr.nbytes)
    src = ctypes.c_void_p(arr.ctypes.data)
    hip_dll.hipMemcpy(ptr, src, arr.nbytes, 1)
    return ptr

def read_gpu(ptr, shape, dtype=np.float32):
    out = np.empty(shape, dtype=dtype)
    dst = ctypes.c_void_p(out.ctypes.data)
    hip_dll.hipMemcpy(dst, ptr, out.nbytes, 2)
    return out

def test_strided_batched(name, batch, m, n, k, trans_a, trans_b, scale=1.0):
    np.random.seed(42)
    A = np.random.randn(batch, m, k).astype(np.float32) * 0.1
    B = np.random.randn(batch, k if not trans_b else n, n if not trans_b else k).astype(np.float32) * 0.1
    if trans_b:
        B = np.random.randn(batch, n, k).astype(np.float32) * 0.1

    ref = np.zeros((batch, m, n), dtype=np.float32)
    for b in range(batch):
        Bmat = B[b].T if trans_b else B[b]
        ref[b] = scale * (A[b] @ Bmat)

    lda_host = A.shape[-1]
    ldb_host = B.shape[-1]
    stridea_host = m * k
    strideb_host = B.shape[1] * B.shape[2]
    stridec_host = m * n

    A_gpu = alloc_gpu(A)
    B_gpu = alloc_gpu(B)
    C_host = np.zeros((batch, m, n), dtype=np.float32)
    C_gpu = alloc_gpu(C_host)

    # CTranslate2 swaps A<->B and m<->n for column-major
    op_a_cublas = HIPBLAS_OP_T if trans_b else HIPBLAS_OP_N
    op_b_cublas = HIPBLAS_OP_T if trans_a else HIPBLAS_OP_N

    alpha = ctypes.c_float(scale)
    beta = ctypes.c_float(0.0)

    status = SgemmStridedBatched(
        handle,
        op_a_cublas, op_b_cublas,
        n, m, k,
        ctypes.byref(alpha),
        B_gpu, ldb_host, strideb_host,
        A_gpu, lda_host, stridea_host,
        ctypes.byref(beta),
        C_gpu, n, stridec_host,
        batch
    )
    hipDeviceSynchronize()

    result = read_gpu(C_gpu, (batch, m, n))
    hipFree(A_gpu)
    hipFree(B_gpu)
    hipFree(C_gpu)

    diff = np.abs(result - ref)
    corr = np.corrcoef(result.flat, ref.flat)[0, 1]
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    status_str = "PASS" if corr > 0.999 else "FAIL"
    print(f"  {name}: status={status}, corr={corr:.6f}, max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f} [{status_str}]")
    return corr > 0.999

print("=" * 60)
print("TEST: hipblasSgemmStridedBatched")
print("=" * 60)

results = []

# Test 1: Q @ K^T — exact Whisper tiny encoder attention shape
print("\n--- Whisper Attention Shapes ---")
results.append(test_strided_batched(
    "Q@K^T (6 heads, 1500x1500, k=64, trans_b)",
    batch=6, m=1500, n=1500, k=64, trans_a=False, trans_b=True, scale=0.125
))

# Test 2: attn @ V — exact Whisper tiny shape
results.append(test_strided_batched(
    "attn@V (6 heads, 1500x64, k=1500)",
    batch=6, m=1500, n=64, k=1500, trans_a=False, trans_b=False
))

# Test 3: Smaller sanity checks
print("\n--- Sanity Checks ---")
results.append(test_strided_batched(
    "Small (2 batches, 4x4, k=4)",
    batch=2, m=4, n=4, k=4, trans_a=False, trans_b=False
))

results.append(test_strided_batched(
    "Small trans_b (2 batches, 4x4, k=4)",
    batch=2, m=4, n=4, k=4, trans_a=False, trans_b=True
))

results.append(test_strided_batched(
    "Medium (6 heads, 64x64, k=64)",
    batch=6, m=64, n=64, k=64, trans_a=False, trans_b=True, scale=0.125
))

# Test 4: Single batch (should match regular GEMM)
print("\n--- Single Batch (baseline) ---")
results.append(test_strided_batched(
    "Single batch Q@K^T shape",
    batch=1, m=1500, n=1500, k=64, trans_a=False, trans_b=True, scale=0.125
))

print(f"\n{'='*60}")
print(f"RESULTS: {sum(results)}/{len(results)} passed")
if all(results):
    print("ALL STRIDED BATCHED GEMM TESTS PASS — bug is NOT in hipblasSgemmStridedBatched")
    print(">>> Bug must be in transpose_0213 or Split GPU kernel")
else:
    print("STRIDED BATCHED GEMM FAILED — THIS IS THE BUG!")
print("=" * 60)

hipblas_dll.hipblasDestroy(handle)
