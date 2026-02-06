# CTranslate2 ROCm Build for RDNA 1 (gfx1010)

CTranslate2 v4.7.1 built from source with ROCm 6.2 on Windows, targeting AMD RDNA 1 GPUs (gfx1010). Tested and verified working on an RX 5700 XT.

## Why this exists

The official CTranslate2 v4.7.1 ROCm wheel (from GitHub releases) is built for **ROCm 7.1.1** targeting **gfx1030+** (RDNA 2 and newer). It doesn't work on RDNA 1 for multiple reasons:

- **RDNA 1 (RX 5000 series) is not detected by ROCm 7.** The Adrenalin gaming driver ships ROCm 6 runtime DLLs. HIP SDK 7 returns zero devices; only **HIP SDK 6.2** recognizes RDNA 1 hardware.
- **No gfx1010 GPU code in the wheel.** The pre-built binary contains GPU kernels for gfx1030/gfx1100+ only. Code compiled for one GPU architecture cannot run on another.
- **No gfx1010 Tensile kernels in stock rocBLAS.** The HIP SDK's rocBLAS only ships matrix math kernels for officially supported architectures. RDNA 1 was never officially supported.
- **ROCm 6 vs 7 API differences.** `hipblasDatatype_t` and `hipblasComputeType_t` changed between versions, requiring source patches to compile against ROCm 6.2.

This repo contains the patched CTranslate2 source (3 files changed), build scripts, community rocBLAS with gfx1010 kernels, and the ready-to-install wheel + DLL.

### RDNA 1 GPUs (gfx1010)

RX 5700 XT, RX 5700, RX 5600 XT, RX 5500 XT — all use the gfx1010 architecture and should work with this build. Only tested on RX 5700 XT.

## Prerequisites

### HIP SDK 6.2

Install **HIP SDK 6.2.x** (NOT 7.x). The Adrenalin gaming driver ships ROCm 6 runtime DLLs, so the SDK version must match. HIP SDK 7 will see zero devices on RDNA 1.

During installation, select the **full install** — you need both:
- **HIP Runtime** — `amdhip64.dll`, device management, kernel launch
- **HIP Runtime Compiler** — includes `amd_comgr.dll` (Code Object Manager), required for GPU initialization. Without it you get a misleading "no ROCm-capable device" error even though the GPU is present.

### Community rocBLAS with gfx1010 Tensile kernels

The stock rocBLAS in HIP SDK 6.2 has no gfx1010 kernels. Download community-built ones from [likelovewant/ROCmLibs](https://github.com/likelovewant/ROCmLibs-for-gfx1103-AMD780M-APU/releases/tag/v0.6.2.4) (v0.6.2.4). A copy of the archive is included in `community-rocblas-gfx1010/`.

Extract and replace `rocblas.dll` + `library/` in `C:\Program Files\AMD\ROCm\6.2\bin\rocblas\`.

### MSVC Build Tools

ROCm's clang compiler on Windows delegates linking to MSVC. Install **Build Tools for Visual Studio 2022 or later** with the **"Desktop development with C++"** workload. The components needed:
- **MSVC Build Tools for x64/x86 (Latest)** — compiler, linker, runtime libs (`msvcrt.lib`, `oldnames.lib`)
- **MSVC v143 (VS 2022) toolchain** — also check this in the installer. MSVC 2026 (v14.50) STL headers use `__builtin_verbose_trap` which ROCm's Clang 19.0.0 doesn't support. The build scripts work around this, but having v143 installed is a safety net.
- **Windows SDK** — system headers and libs

### Other tools

| Tool | Version used | Install |
|------|-------------|---------|
| CMake | 4.2+ | `pip install cmake` |
| Ninja | 1.13+ | `pip install ninja` |
| Python | 3.13 | Target runtime for the wheel |
| pybind11 | 3.0+ | `pip install pybind11` (installed by build script) |

## What this folder contains

### Build Scripts (root)

| File | Purpose |
|------|---------|
| `configure.bat` | CMake configure step. Sets up MSVC + ROCm environment, then runs cmake with all the flags needed for a ROCm 6.2 / gfx1010 build. Deletes previous build dir first. |
| `build.bat` | Compile step. Sets up environment, then runs `cmake --build` with parallel jobs. Takes 10-30 min. |
| `install_and_wheel.bat` | Package step. Installs the built C++ library, then builds a Python `.whl` file from the `python/` subdirectory. |

Run in order: `configure.bat` then `build.bat` then `install_and_wheel.bat`.

### `dist/` — Build output (ready to install)

| File | What |
|------|------|
| `ctranslate2-4.7.1-cp313-cp313-win_amd64.whl` | Python wheel — `pip install --force-reinstall <this>` |
| `ctranslate2.dll` | C++ library — copy to `site-packages/ctranslate2/` after pip install |

### `CTranslate2/` — Source tree (forked from GitHub)

CTranslate2 v4.7.1 source code with 3 patches applied for ROCm 6.2 compatibility:

| Patched File | What Changed |
|--------------|-------------|
| `src/cuda/primitives.cu` | Cast `HIP_R_*` constants to `hipblasDatatype_t` and map compute types to `hipblasDatatype_t` (ROCm 6 vs 7 API difference) |
| `src/cuda/helpers.h` | Define `__syncwarp` as `__syncthreads()` for HIP (AMD wavefronts are lockstep) |
| `python/ctranslate2/__init__.py` | Add ROCm 6.2 bin directory to DLL search path |

Key build outputs inside this tree:

| Path | What |
|------|------|
| `build/` | Full CMake build directory (object files, intermediates) |
| `build/install/bin/ctranslate2.dll` | The compiled C++ library with gfx1010 GPU code |
| `build/install/include/` | C++ headers (for linking) |
| `build/install/lib/` | Import libraries (for linking) |
| `python/dist/ctranslate2-4.7.1-cp313-cp313-win_amd64.whl` | The custom Python wheel — install this with pip |

### `community-rocblas-gfx1010/` — Community rocBLAS

Downloaded from [likelovewant/ROCmLibs](https://github.com/likelovewant/ROCmLibs-for-gfx1103-AMD780M-APU/releases/tag/v0.6.2.4) (v0.6.2.4).

| Path | What |
|------|------|
| `rocm.gfx1010-xnack-...for.hip.sdk.6.2.4.7z` | 7z archive (~12MB) containing rocblas.dll + Tensile kernel library (~600MB extracted) |

Extract with `7z x <archive>` to get `rocblas.dll` and `library/` folder. These replace the stock files in `C:\Program Files\AMD\ROCm\6.2\bin\rocblas\`. The stock rocBLAS only has kernels for gfx906/gfx1030/gfx1100+ — no gfx1010.

### `docs/` — Documentation

| File | What |
|------|------|
| `ctranslate2-rocm6-build.md` | Step-by-step record of the successful build. Every command, every patch, every workaround. |
| `rocm-gfx1010-build-plan.md` | The full journey: gate checks, DLL shim attempt, diagnosis, and resolution. |
| `gpu-stack-explainer.html` | Visual explainer: how the CUDA/ROCm/CTranslate2/faster-whisper stack fits together. |
| `rocm-build-explainer.html` | Visual explainer: what ROCm components are, what gfx codes mean, why rocBLAS is the bottleneck. |
| `ctranslate2-build-explainer.html` | Visual explainer: why the pre-built wheel fails and what building from source actually does. |

## How to use the build output

```
pip install --force-reinstall dist\ctranslate2-4.7.1-cp313-cp313-win_amd64.whl
copy dist\ctranslate2.dll → site-packages\ctranslate2\
```

The app also needs `os.add_dll_directory(r"C:\Program Files\AMD\ROCm\6.2\bin")` called before importing CTranslate2, so Windows can find the ROCm DLLs at runtime.

## Environment this was built on

- GPU: AMD RX 5700 XT (gfx1010, RDNA 1)
- OS: Windows 10+
- HIP SDK: 6.2
- Compiler: ROCm Clang 19.0.0
- MSVC: Build Tools 2026 (v14.50)
- Python: 3.13.5
- Community rocBLAS: likelovewant/ROCmLibs v0.6.2.4
