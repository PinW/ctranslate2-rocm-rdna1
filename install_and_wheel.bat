@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1

set ROCM=C:\PROGRA~1\AMD\ROCm\6.2
set ROCM_PATH=%ROCM%
set HIP_PATH=%ROCM%
set PATH=%ROCM%\bin;%PATH%

cd /d %~dp0CTranslate2

REM Install the built C++ library
cmake --install build --prefix build/install

REM Copy DLL into Python package so it gets included in the wheel
copy build\install\bin\ctranslate2.dll python\ctranslate2\ctranslate2.dll

REM Build the Python wheel
set CTRANSLATE2_ROOT=%~dp0CTranslate2\build\install
cd python
pip install pybind11
pip wheel . --no-deps -w dist
