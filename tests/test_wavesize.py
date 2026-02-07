import os
import ctypes

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

hipGetDeviceProperties = hip.hipGetDeviceProperties

class hipDeviceProp(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char * 256),
        ("totalGlobalMem", ctypes.c_size_t),
        ("sharedMemPerBlock", ctypes.c_size_t),
        ("regsPerBlock", ctypes.c_int),
        ("warpSize", ctypes.c_int),
        ("memPitch", ctypes.c_size_t),
        ("maxThreadsPerBlock", ctypes.c_int),
        ("maxThreadsDim", ctypes.c_int * 3),
        ("maxGridSize", ctypes.c_int * 3),
        ("clockRate", ctypes.c_int),
        ("totalConstMem", ctypes.c_size_t),
        ("major", ctypes.c_int),
        ("minor", ctypes.c_int),
        # ... many more fields but we only need the above
        ("_padding", ctypes.c_char * 4096),
    ]

props = hipDeviceProp()
status = hipGetDeviceProperties(ctypes.byref(props), 0)
print(f"hipGetDeviceProperties status: {status}")
print(f"Device name: {props.name.decode()}")
print(f"Warp size: {props.warpSize}")
print(f"Max threads per block: {props.maxThreadsPerBlock}")
print(f"Compute capability: {props.major}.{props.minor}")
print(f"Shared mem per block: {props.sharedMemPerBlock}")

# Also check via hipDeviceGetAttribute
hipDeviceGetAttribute = hip.hipDeviceGetAttribute
hipDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
hipDeviceGetAttribute.restype = ctypes.c_int

val = ctypes.c_int()
# hipDeviceAttributeWarpSize = 10
hipDeviceGetAttribute(ctypes.byref(val), 10, 0)
print(f"hipDeviceAttribute WarpSize: {val.value}")

print()
print("=== Checking CTranslate2 build flags ===")
import ctranslate2
print(f"CTranslate2 version: {ctranslate2.__version__}")

# Check what CUDA compute capability CT2 reports
print(f"GPU count: {ctranslate2.get_cuda_device_count()}")
