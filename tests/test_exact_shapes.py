import os
import ctypes
import numpy as np

from pathlib import Path
rocm_base = Path(r'C:\Program Files\AMD\ROCm')
rocm_bin = None
if rocm_base.exists():
    for version_dir in sorted(rocm_base.iterdir(), reverse=True):
        rocm_bin = version_dir / 'bin'
        if rocm_bin.exists() and (rocm_bin / 'amdhip64_6.dll').exists():
            os.add_dll_directory(str(rocm_bin))
            break

hip = ctypes.CDLL(str(rocm_bin / 'amdhip64_6.dll'))
hipblas = ctypes.CDLL(str(rocm_bin / 'hipblas.dll'))

hipMalloc = hip.hipMalloc
hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
hipMalloc.restype = ctypes.c_int
hipMemcpy = hip.hipMemcpy
hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
hipMemcpy.restype = ctypes.c_int
hipFree = hip.hipFree
hipFree.argtypes = [ctypes.c_void_p]
hipFree.restype = ctypes.c_int
hipDeviceSynchronize = hip.hipDeviceSynchronize
hipDeviceSynchronize.restype = ctypes.c_int

hipblasCreate = hipblas.hipblasCreate
hipblasCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
hipblasCreate.restype = ctypes.c_int

hipblasSgemm_fn = hipblas.hipblasSgemm
hipblasSgemm_fn.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_void_p, ctypes.c_int,
    ctypes.c_void_p, ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_void_p, ctypes.c_int,
]
hipblasSgemm_fn.restype = ctypes.c_int

HIPBLAS_OP_N = 111
HIPBLAS_OP_T = 112
H2D = 1
D2H = 2

handle = ctypes.c_void_p()
hipblasCreate(ctypes.byref(handle))


def test_sgemm(M, N, K, alpha_val=1.0, desc=""):
    np.random.seed(42)
    # Simple row-major test: C = alpha * A @ B
    # A is M x K, B is K x N, C is M x N
    # hipBLAS is column-major, so we use the transpose trick:
    # hipblasSgemm(N, N, N, M, K, alpha, B_cm, N, A_cm, K, beta, C_cm, N)
    # where _cm means column-major storage = row-major transposed
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    expected = alpha_val * (A @ B)

    # For column-major: pass B^T and A^T, compute B^T * A^T = (A*B)^T
    # numpy arrays are row-major (C order), which is column-major transposed
    # So A (M x K in row-major) is A^T (K x M in column-major)
    # hipblasSgemm(OP_N, OP_N, N, M, K, alpha, B, N, A, K, beta, C, N)
    d_A = ctypes.c_void_p()
    d_B = ctypes.c_void_p()
    d_C = ctypes.c_void_p()
    hipMalloc(ctypes.byref(d_A), A.nbytes)
    hipMalloc(ctypes.byref(d_B), B.nbytes)
    hipMalloc(ctypes.byref(d_C), M * N * 4)

    hipMemcpy(d_A, A.ctypes.data, A.nbytes, H2D)
    hipMemcpy(d_B, B.ctypes.data, B.nbytes, H2D)

    alpha = ctypes.c_float(alpha_val)
    beta = ctypes.c_float(0.0)

    status = hipblasSgemm_fn(handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                              N, M, K,
                              ctypes.byref(alpha),
                              d_B, N,
                              d_A, K,
                              ctypes.byref(beta),
                              d_C, N)

    hipDeviceSynchronize()
    C = np.zeros((M, N), dtype=np.float32)
    hipMemcpy(C.ctypes.data, d_C, M * N * 4, D2H)

    hipFree(d_A)
    hipFree(d_B)
    hipFree(d_C)

    max_err = np.max(np.abs(C - expected))
    mean_err = np.mean(np.abs(C - expected))
    rel_err = max_err / (np.max(np.abs(expected)) + 1e-10)
    tol = K * 2e-5
    passed = max_err < tol

    status_str = "PASS" if passed else "FAIL"
    print(f"[{status_str}] {desc}")
    print(f"  {M}x{K} @ {K}x{N} -> {M}x{N}  status={status}  max_err={max_err:.2e}  rel_err={rel_err:.2e}")
    if not passed:
        print(f"  *** Expected[0,:5]: {expected[0,:5]}")
        print(f"  *** Got[0,:5]:      {C[0,:5]}")
    return passed


print("Testing exact GEMM shapes from Whisper tiny encoder")
print("=" * 70)

results = []

# Conv1d layers
results.append(test_sgemm(384, 3000, 240, desc="conv1d layer 1 (384x240 weight @ 240x3000 patches)"))
results.append(test_sgemm(384, 1500, 1152, desc="conv1d layer 2 (384x1152 weight @ 1152x1500 patches)"))

# Encoder self-attention
results.append(test_sgemm(1152, 1500, 384, desc="encoder QKV projection (1152x384 @ 384x1500)"))
results.append(test_sgemm(1500, 1500, 64, 0.125, desc="attention Q*K^T (1500x64 @ 64x1500, scale=0.125)"))
results.append(test_sgemm(64, 1500, 1500, desc="attention attn*V (64x1500 @ 1500x1500)"))
results.append(test_sgemm(384, 1500, 384, desc="attention output proj (384x384 @ 384x1500)"))

# FFN
results.append(test_sgemm(1536, 1500, 384, desc="FFN up-projection (1536x384 @ 384x1500)"))
results.append(test_sgemm(384, 1500, 1536, desc="FFN down-projection (384x1536 @ 1536x1500)"))

# Decoder
results.append(test_sgemm(768, 1500, 384, desc="cross-attn KV projection"))
results.append(test_sgemm(51865, 1, 384, desc="final logits (51865x384 @ 384x1)"))

print("=" * 70)
passed = sum(results)
total = len(results)
print(f"Results: {passed}/{total} passed")
if passed == total:
    print("ALL SGEMM shapes produce correct results!")
    print("The bug is NOT in rocBLAS — it's in CTranslate2 custom kernels.")
