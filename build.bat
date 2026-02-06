@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1

set ROCM=C:\PROGRA~1\AMD\ROCm\6.2
set ROCM_PATH=%ROCM%
set HIP_PATH=%ROCM%
set HIP_PLATFORM=amd
set HIP_DEVICE_LIB_PATH=%ROCM%\amdgcn\bitcode
set PATH=%ROCM%\bin;%PATH%

cd /d %~dp0CTranslate2
cmake --build build --config Release -j
