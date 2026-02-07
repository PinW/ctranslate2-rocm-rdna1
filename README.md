# CTranslate2 ROCm Build for RDNA 1 (gfx1010)

CTranslate2 v4.7.1 built from source with ROCm 6.2 on Windows, targeting AMD RDNA 1 GPUs (RX 5700 XT, RX 5700, RX 5600 XT, RX 5500 XT). GPU float16/float32 and multi-threaded CPU inference verified on RX 5700 XT.

The official CTranslate2 ROCm wheel targets ROCm 7 / gfx1030+ (RDNA 2+). RDNA 1 needs ROCm 6.2, community rocBLAS kernels, and source patches to work.

## Quick start

1. Install [HIP SDK 6.2](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html) (full install)
2. Replace stock rocBLAS with [community gfx1010 build](https://github.com/likelovewant/ROCmLibs-for-gfx1103-AMD780M-APU/releases/tag/v0.6.2.4) (see [instructions](#2-community-rocblas-with-gfx1010-tensile-kernels))
3. Install the wheel from [releases](https://github.com/PinW/ctranslate2-rocm-rdna1/releases/latest):
```
pip install ctranslate2-4.7.1+rocm62.gfx1010-cp313-cp313-win_amd64.whl --force-reinstall --no-deps
```

## Building from source

### 1. HIP SDK 6.2

Install **HIP SDK 6.2.x** (NOT 7.x). The Adrenalin gaming driver ships ROCm 6 runtime DLLs, so the SDK version must match. HIP SDK 7 will see zero devices on RDNA 1.

During installation, select the **full install** -- you need both:
- **HIP Runtime** -- `amdhip64.dll`, device management, kernel launch
- **HIP Runtime Compiler** -- includes `amd_comgr.dll` (Code Object Manager), required for GPU initialization. Without it you get a misleading "no ROCm-capable device" error even though the GPU is present.

The installer sets `HIP_PATH` as a system environment variable. The wheel's `__init__.py` reads this to locate the ROCm DLLs at runtime.

### 2. Community rocBLAS with gfx1010 Tensile kernels

The stock rocBLAS in HIP SDK 6.2 has no gfx1010 kernels. Download community-built ones from [likelovewant/ROCmLibs](https://github.com/likelovewant/ROCmLibs-for-gfx1103-AMD780M-APU/releases/tag/v0.6.2.4) (v0.6.2.4). A copy of the archive is included in `community-rocblas-gfx1010/`.

Extract and replace `rocblas.dll` + `library/` in `C:\Program Files\AMD\ROCm\6.2\bin\rocblas\`.

### 3. MSVC Build Tools

ROCm's clang compiler on Windows delegates linking to MSVC. Install **Build Tools for Visual Studio 2022 or later** with the **"Desktop development with C++"** workload. The components needed:
- **MSVC Build Tools for x64/x86 (Latest)** -- compiler, linker, runtime libs (`msvcrt.lib`, `oldnames.lib`)
- **MSVC v143 (VS 2022) toolchain** -- also check this in the installer. MSVC 2026 (v14.50) STL headers use `__builtin_verbose_trap` which ROCm's Clang 19.0.0 doesn't support. The build scripts work around this, but having v143 installed is a safety net.
- **Windows SDK** -- system headers and libs

### 4. Other tools

| Tool | Version used | Install |
|------|-------------|---------|
| CMake | 4.2+ | `pip install cmake` |
| Ninja | 1.13+ | `pip install ninja` |
| Python | 3.13 | Target runtime for the wheel |
| pybind11 | 3.0+ | `pip install pybind11` (installed by build script) |

### 5. Build

Run in order:

1. `build_onednn.bat` -- build oneDNN 3.1.1 static lib (run once, output goes to `onednn-install/`)
2. `configure.bat` -- CMake configure (deletes previous build dir)
3. `build.bat` -- compile (10-30 min)
4. `install_and_wheel.bat` -- package into Python wheel

Install the wheel:
```
pip install CTranslate2\python\dist\ctranslate2-4.7.1+rocm62.gfx1010-cp313-cp313-win_amd64.whl --force-reinstall --no-deps
```

The wheel reads the `HIP_PATH` environment variable (set by the HIP SDK installer) to locate ROCm DLLs at runtime. No manual DLL path setup needed.

## What's in this repo

### `CTranslate2/` -- Patched source tree

CTranslate2 v4.7.1 with 4 files patched for ROCm 6.2 / gfx1010:

| Patched File | What Changed |
|--------------|-------------|
| `src/cuda/primitives.cu` | Use hipBLAS v2 API (`hipblasGemmEx_v2`) with `hipblasComputeType_t` and `hipDataType` for correct float16/float32 GEMM dispatch |
| `src/cuda/helpers.h` | `__syncwarp(mask)` defined as no-op (RDNA 1 wavefronts are lockstep; mapping to `__syncthreads()` causes a barrier race in `block_reduce()` since the GPU's `s_barrier` counts all wavefront arrivals regardless of code location) |
| `python/ctranslate2/__init__.py` | Add `HIP_PATH` env var lookup to DLL search path for ROCm runtime DLLs |
| `python/ctranslate2/version.py` | Version marked as `4.7.1+rocm62.gfx1010` to distinguish from upstream |

Build outputs: `build/install/bin/ctranslate2.dll` (compiled library), `python/dist/*.whl` (Python wheel)

### `community-rocblas-gfx1010/` -- Community rocBLAS

7z archive from [likelovewant/ROCmLibs](https://github.com/likelovewant/ROCmLibs-for-gfx1103-AMD780M-APU/releases/tag/v0.6.2.4) (v0.6.2.4) containing `rocblas.dll` + Tensile kernel library for gfx1010. Replaces the stock files in `C:\Program Files\AMD\ROCm\6.2\bin\rocblas\`.

### `onednn-3.1.1.tar.gz` -- oneDNN source

Source archive for [oneDNN 3.1.1](https://github.com/oneapi-src/oneDNN/releases/tag/v3.1.1). Built into a static library linked into ctranslate2.dll, providing multi-threaded CPU GEMM support (int8, float32) via Intel OpenMP.

### `tests/` -- Debug scripts

Test scripts (.py, .hip) from GPU debugging investigation. Not needed for normal use.

## Tested with

- GPU: AMD RX 5700 XT (gfx1010, RDNA 1)
- OS: Windows 11
- HIP SDK: 6.2
- Compiler: ROCm Clang 19.0.0
- MSVC: Build Tools 2026 (v14.50)
- Python: 3.13.5
- Community rocBLAS: likelovewant/ROCmLibs v0.6.2.4
