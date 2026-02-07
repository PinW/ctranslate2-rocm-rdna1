import os
import sys
import struct
import numpy as np
from pathlib import Path

rocm_base = Path(r'C:\Program Files\AMD\ROCm')
if rocm_base.exists():
    for version_dir in sorted(rocm_base.iterdir(), reverse=True):
        rocm_bin = version_dir / 'bin'
        if rocm_bin.exists() and (rocm_bin / 'amdhip64_6.dll').exists():
            os.add_dll_directory(str(rocm_bin))
            break

import ctranslate2

model_path = None
for p in Path.home().rglob('.cache/huggingface/hub/models--Systran--faster-whisper-tiny/snapshots/*/model.bin'):
    model_path = str(p.parent)
    break

print(f"Model: {model_path}")

# Load model on GPU
model_gpu = ctranslate2.models.Whisper(model_path, device="cuda", compute_type="float32")

# Check if we can list model variables
print("\n=== Checking model variable access ===")
try:
    vars_list = model_gpu.model_spec if hasattr(model_gpu, 'model_spec') else None
    print(f"model_spec: {vars_list}")
except Exception as e:
    print(f"model_spec failed: {e}")

# Try loading the model file directly to extract weights
model_bin = Path(model_path) / "model.bin"
print(f"\nModel file: {model_bin} ({model_bin.stat().st_size / 1024 / 1024:.1f} MB)")

# CTranslate2 model format: we can read variables using the reader
# Let's try using ctranslate2's internal reader
try:
    # Load model on CPU to access weights
    import ctranslate2
    reader = ctranslate2.models.Whisper(model_path, device="cpu", compute_type="float32")
    print("Loaded CPU model for weight extraction")

    # The model format stores variables — let's try to access them
    print(f"\nMethods: {[m for m in dir(reader) if not m.startswith('_')]}")
except Exception as e:
    print(f"CPU model failed (expected - no SGEMM): {e}")

# Alternative: read the CTranslate2 model binary directly
# The format is documented and we can parse it
print("\n=== Trying to read model weights directly ===")

# CTranslate2 model.bin format:
# - Header: magic, spec revision, etc
# - Variables: name, shape, type, data
# Let's just check if we can read the first few variable names

with open(str(model_bin), 'rb') as f:
    # Read binary version
    data = f.read(4)
    binary_version = struct.unpack('<I', data)[0]
    print(f"Binary version: {binary_version}")

    # Read spec revision
    data = f.read(4)
    spec_revision = struct.unpack('<I', data)[0]
    print(f"Spec revision: {spec_revision}")

    # Read number of variables
    data = f.read(4)
    num_variables = struct.unpack('<I', data)[0]
    print(f"Number of variables: {num_variables}")

    # Read first few variable names
    print(f"\nFirst 10 variables:")
    for i in range(min(10, num_variables)):
        # Read variable name length
        name_len = struct.unpack('<H', f.read(2))[0]
        name = f.read(name_len).decode('utf-8')

        # Read number of dimensions
        ndims = struct.unpack('<B', f.read(1))[0]

        # Read dimensions
        dims = []
        for _ in range(ndims):
            dims.append(struct.unpack('<I', f.read(4))[0])

        # Read data type (1=float32, 4=float16, etc)
        dtype = struct.unpack('<B', f.read(1))[0]

        num_elements = 1
        for d in dims:
            num_elements *= d

        # Calculate size based on dtype
        if dtype == 0:
            elem_size = 4  # float32
            dtype_name = "float32"
        elif dtype == 4:
            elem_size = 2  # float16
            dtype_name = "float16"
        else:
            elem_size = 4
            dtype_name = f"unknown({dtype})"

        total_bytes = num_elements * elem_size

        print(f"  {name}: shape={dims}, dtype={dtype_name}, size={total_bytes} bytes")

        if "encoder" in name and "conv1" in name and "weight" in name and ndims > 0:
            # Read this weight!
            weight_data = f.read(total_bytes)
            weight = np.frombuffer(weight_data, dtype=np.float32 if dtype == 0 else np.float16)
            weight = weight.reshape(dims)
            print(f"    -> Loaded! shape={weight.shape}, "
                  f"mean={np.mean(weight):.6f}, std={np.std(weight):.6f}")
            np.save(f"weight_{name.replace('/', '_')}.npy", weight)
            print(f"    -> Saved to weight_{name.replace('/', '_')}.npy")
        else:
            # Skip the data
            f.seek(total_bytes, 1)
