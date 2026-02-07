@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1

set ROCM=C:\PROGRA~1\AMD\ROCm\6.2
set ROCM_PATH=%ROCM%
set HIP_PATH=%ROCM%
set HIP_PLATFORM=amd
set HIP_DEVICE_LIB_PATH=%ROCM%\amdgcn\bitcode
set CC=%ROCM%\bin\clang.exe
set CXX=%ROCM%\bin\clang++.exe
set PATH=%ROCM%\bin;%PATH%
set INTEL_ROOT=C:\Users\pinwa\AppData\Local\Programs\Python\Python313\Library

cd /d C:\Users\pinwa\projects\5700xt-rocm\CTranslate2
rmdir /s /q build 2>nul

cmake -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -S . -B build -DBUILD_CLI=OFF -DWITH_MKL=OFF -DWITH_DNNL=ON -DOPENMP_RUNTIME=INTEL -DWITH_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1010 -DCMAKE_HIP_COMPILER=C:/PROGRA~1/AMD/ROCm/6.2/bin/clang++.exe -DCMAKE_HIP_COMPILER_ROCM_ROOT=C:/PROGRA~1/AMD/ROCm/6.2 -DCMAKE_PREFIX_PATH="C:/PROGRA~1/AMD/ROCm/6.2;C:/Users/pinwa/projects/5700xt-rocm/onednn-install" "-DCMAKE_HIP_FLAGS=-D_MSVC_STL_DOOM_FUNCTION(mesg)=__builtin_trap()" "-DCMAKE_CXX_FLAGS=-fopenmp -D_MSVC_STL_DOOM_FUNCTION(mesg)=__builtin_trap()" -DINTEL_ROOT=C:/Users/pinwa/AppData/Local/Programs/Python/Python313/Library
