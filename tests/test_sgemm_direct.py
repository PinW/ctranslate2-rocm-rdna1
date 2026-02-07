import os
import ctypes
import ctypes.wintypes
import numpy as np

from pathlib import Path
rocm_base = Path(r'C:\Program Files\AMD\ROCm')
rocm_bin = None
if rocm_base.exists():
    for version_dir in sorted(rocm_base.iterdir(), reverse=True):
        rocm_bin = version_dir / 'bin'
        if rocm_bin.exists() and (rocm_bin / 'amdhip64_6.dll').exists():
            os.add_dll_directory(str(rocm_bin))
            print(f"ROCm: {rocm_bin}")
            break

print("\n=== TEST A: hipblasSgemm via ctypes ===")
print("(Tests rocBLAS SGEMM kernels directly, bypassing CTranslate2)\n")

try:
    hipblas = ctypes.CDLL(str(rocm_bin / 'hipblas.dll'))
    hip = ctypes.CDLL(str(rocm_bin / 'amdhip64_6.dll'))

    hipMalloc = hip.hipMalloc
    hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    hipMalloc.restype = ctypes.c_int

    hipMemcpy = hip.hipMemcpy
    hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    hipMemcpy.restype = ctypes.c_int

    hipFree = hip.hipFree
    hipFree.argtypes = [ctypes.c_void_p]
    hipFree.restype = ctypes.c_int

    hipblasCreate = hipblas.hipblasCreate
    hipblasCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    hipblasCreate.restype = ctypes.c_int

    hipblasSgemm_fn = hipblas.hipblasSgemm
    hipblasSgemm_fn.argtypes = [
        ctypes.c_void_p,  # handle
        ctypes.c_int,     # transA
        ctypes.c_int,     # transB
        ctypes.c_int,     # m
        ctypes.c_int,     # n
        ctypes.c_int,     # k
        ctypes.POINTER(ctypes.c_float),  # alpha
        ctypes.c_void_p,  # A
        ctypes.c_int,     # lda
        ctypes.c_void_p,  # B
        ctypes.c_int,     # ldb
        ctypes.POINTER(ctypes.c_float),  # beta
        ctypes.c_void_p,  # C
        ctypes.c_int,     # ldc
    ]
    hipblasSgemm_fn.restype = ctypes.c_int

    HIPBLAS_OP_N = 111
    HIPBLAS_OP_T = 112
    hipMemcpyHostToDevice = 1
    hipMemcpyDeviceToHost = 2

    handle = ctypes.c_void_p()
    status = hipblasCreate(ctypes.byref(handle))
    print(f"hipblasCreate: status={status}")

    for M, N, K, label in [(4, 4, 4, "4x4"), (64, 64, 64, "64x64"), (384, 384, 384, "384x384")]:
        np.random.seed(42)
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        expected = A @ B

        d_A = ctypes.c_void_p()
        d_B = ctypes.c_void_p()
        d_C = ctypes.c_void_p()
        hipMalloc(ctypes.byref(d_A), M * K * 4)
        hipMalloc(ctypes.byref(d_B), K * N * 4)
        hipMalloc(ctypes.byref(d_C), M * N * 4)

        hipMemcpy(d_A, A.ctypes.data, M * K * 4, hipMemcpyHostToDevice)
        hipMemcpy(d_B, B.ctypes.data, K * N * 4, hipMemcpyHostToDevice)

        alpha = ctypes.c_float(1.0)
        beta = ctypes.c_float(0.0)

        # Column-major: C = B^T @ A^T gives us row-major A @ B
        status = hipblasSgemm_fn(
            handle,
            HIPBLAS_OP_N, HIPBLAS_OP_N,
            N, M, K,
            ctypes.byref(alpha),
            d_B, N,
            d_A, K,
            ctypes.byref(beta),
            d_C, N,
        )

        C = np.zeros((M, N), dtype=np.float32)
        hipMemcpy(C.ctypes.data, d_C, M * N * 4, hipMemcpyDeviceToHost)

        hipFree(d_A)
        hipFree(d_B)
        hipFree(d_C)

        max_err = np.max(np.abs(C - expected))
        mean_err = np.mean(np.abs(C - expected))
        match = "PASS" if max_err < 1e-3 else "FAIL"
        print(f"[{label}] sgemm status={status} max_err={max_err:.2e} mean_err={mean_err:.2e} {match}")

        if M <= 4:
            print(f"  Expected[0]: {expected[0]}")
            print(f"  Got[0]:      {C[0]}")

except Exception as e:
    import traceback
    print(f"Direct hipblasSgemm test failed: {e}")
    traceback.print_exc()

print("\n=== TEST B: CTranslate2 with ROCBLAS_LAYER tracing ===")
print("(Runs encoder, traces all rocBLAS calls)\n")

os.environ["ROCBLAS_LAYER"] = "2"
os.environ["ROCBLAS_LOG_TRACE_PATH"] = "rocblas_trace.log"

from faster_whisper import WhisperModel

model = WhisperModel("tiny", device="cuda", compute_type="float32")

silence = np.zeros(16000 * 2, dtype=np.float32)
lang, prob, _ = model.detect_language(silence)
print(f"detect_language result: {lang} (prob={prob:.3f})")

trace_path = Path("rocblas_trace.log")
if trace_path.exists():
    lines = trace_path.read_text().strip().split('\n')
    print(f"\nrocBLAS trace: {len(lines)} calls")
    for line in lines[:20]:
        print(f"  {line}")
    if len(lines) > 20:
        print(f"  ... ({len(lines) - 20} more)")
else:
    print("No trace file generated")
