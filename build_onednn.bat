@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1

cd /d C:\Users\pinwa\projects\5700xt-rocm\oneDNN-3.1.1

cmake -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DONEDNN_LIBRARY_TYPE=STATIC -DONEDNN_BUILD_EXAMPLES=OFF -DONEDNN_BUILD_TESTS=OFF -DONEDNN_ENABLE_WORKLOAD=INFERENCE "-DONEDNN_ENABLE_PRIMITIVE=CONVOLUTION;REORDER" -DONEDNN_BUILD_GRAPH=OFF -DDNNL_CPU_RUNTIME=OMP -DCMAKE_INSTALL_PREFIX=C:\Users\pinwa\projects\5700xt-rocm\onednn-install -S . -B build

cmake --build build --config Release -j
cmake --install build
