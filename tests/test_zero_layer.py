import os
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


def read_string(f):
    str_len = struct.unpack('<H', f.read(2))[0]
    return f.read(str_len).decode('utf-8').rstrip('\x00')


def write_string(f, s):
    s_bytes = s.encode('utf-8')
    f.write(struct.pack('<H', len(s_bytes) + 1))
    f.write(s_bytes)
    f.write(struct.pack('<B', 0))


DTYPE_MAP = {0: (np.float32, 4), 1: (np.int8, 1), 2: (np.int16, 2),
             3: (np.int32, 4), 4: (np.float16, 2), 5: (np.float16, 2)}


def read_model(filepath):
    variables = []
    with open(str(filepath), 'rb') as f:
        binary_version = struct.unpack('<I', f.read(4))[0]
        spec = ""
        spec_revision = 1
        if binary_version >= 2:
            spec = read_string(f)
            spec_revision = struct.unpack('<I', f.read(4))[0]
        num_variables = struct.unpack('<I', f.read(4))[0]
        header = (binary_version, spec, spec_revision)
        for _ in range(num_variables):
            name = read_string(f)
            rank = struct.unpack('<B', f.read(1))[0]
            dims = [struct.unpack('<I', f.read(4))[0] for _ in range(rank)]
            if binary_version >= 4:
                type_id = struct.unpack('<B', f.read(1))[0]
                num_bytes = struct.unpack('<I', f.read(4))[0]
            else:
                item_size = struct.unpack('<B', f.read(1))[0]
                num_items = struct.unpack('<I', f.read(4))[0]
                num_bytes = num_items * item_size
                type_id = {4: 0}.get(item_size, 0)
            raw = f.read(num_bytes)
            variables.append({'name': name, 'dims': dims, 'type_id': type_id,
                              'num_bytes': num_bytes, 'raw_data': raw})
        aliases = []
        if binary_version >= 3:
            remaining = f.read(4)
            if len(remaining) == 4:
                num_aliases = struct.unpack('<I', remaining)[0]
                for _ in range(num_aliases):
                    aliases.append((read_string(f), read_string(f)))
    return header, variables, aliases


def write_model(filepath, header, variables, aliases):
    binary_version, spec, spec_revision = header
    with open(str(filepath), 'wb') as f:
        f.write(struct.pack('<I', binary_version))
        if binary_version >= 2:
            write_string(f, spec)
            f.write(struct.pack('<I', spec_revision))
        f.write(struct.pack('<I', len(variables)))
        for v in variables:
            write_string(f, v['name'])
            f.write(struct.pack('<B', len(v['dims'])))
            for d in v['dims']:
                f.write(struct.pack('<I', d))
            if binary_version >= 4:
                f.write(struct.pack('<B', v['type_id']))
                f.write(struct.pack('<I', v['num_bytes']))
            f.write(v['raw_data'])
        if binary_version >= 3:
            f.write(struct.pack('<I', len(aliases)))
            for a, t in aliases:
                write_string(f, a)
                write_string(f, t)


header, variables, aliases = read_model(Path(model_path) / "model.bin")
print(f"Read {len(variables)} variables from model")

zeroed_vars = []
for v in variables:
    name = v['name']
    if name.startswith('encoder/layer_0/'):
        np_dtype = DTYPE_MAP.get(v['type_id'], (np.float32, 4))[0]
        zero_data = np.zeros(v['num_bytes'] // np.dtype(np_dtype).itemsize, dtype=np_dtype).tobytes()
        zeroed_vars.append({**v, 'raw_data': zero_data})
        print(f"  Zeroed: {name} ({v['dims']})")
    else:
        zeroed_vars.append(v)

only_layer0 = [v for v in zeroed_vars
               if not (v['name'].startswith('encoder/layer_') and '/layer_0/' not in v['name']
                       and v['name'] != 'encoder/layer_norm/gamma' and v['name'] != 'encoder/layer_norm/beta')]
filtered_aliases = [(a, t) for a, t in aliases
                    if 'encoder/layer_' not in a or 'layer_0' in a]

import shutil
zero_dir = Path(r'C:\Users\pinwa\projects\5700xt-rocm\dist\zero_layer_model')
zero_dir.mkdir(parents=True, exist_ok=True)
for fname in ['vocabulary.json', 'vocabulary.txt', 'tokenizer.json', 'config.json']:
    src = Path(model_path) / fname
    if src.exists():
        shutil.copy2(str(src), str(zero_dir / fname))

write_model(zero_dir / 'model.bin', header, only_layer0, filtered_aliases)
print(f"\nWrote zero-layer model: {len(only_layer0)} vars")

np.random.seed(12345)
features = np.random.randn(1, 80, 3000).astype(np.float32) * 0.1
features_sv = ctranslate2.StorageView.from_array(features)

print("\n=== ZERO-WEIGHT LAYER MODEL (GPU) ===")
print("Transformer layer weights all zero -> passes input through unchanged")
print("Output = conv1d + conv2d + transpose + pos_embed + output_layer_norm\n")

model = ctranslate2.models.Whisper(str(zero_dir), device="cuda", compute_type="float32")
out = model.encode(features_sv, to_cpu=True)
gpu_zero = np.array(out)
print(f"Shape: {gpu_zero.shape}")
print(f"mean={np.mean(gpu_zero):.6f}, std={np.std(gpu_zero):.6f}")
print(f"NaN: {np.any(np.isnan(gpu_zero))}, Inf: {np.any(np.isinf(gpu_zero))}")
print(f"[0,0,:10]: {gpu_zero[0,0,:10]}")
np.save(r"C:\Users\pinwa\projects\5700xt-rocm\dist\encoder_zero_layer_gpu.npy", gpu_zero)
del model

numpy_ref = Path(r'C:\Users\pinwa\projects\5700xt-rocm\dist\encoder_0layers_numpy.npy')
if numpy_ref.exists():
    ref = np.load(str(numpy_ref))
    corr = np.corrcoef(gpu_zero.flat, ref.flat)[0, 1]
    diff = np.abs(gpu_zero - ref)
    print(f"\nGPU zero-layer vs numpy 0-layer: corr={corr:.6f}, max_diff={np.max(diff):.4f}")
    print(f"  numpy [0,0,:5]: {ref[0,0,:5]}")
    print(f"  GPU   [0,0,:5]: {gpu_zero[0,0,:5]}")
    if corr > 0.99:
        print(">>> CONV1D + TRANSPOSE + POS_EMBED + LAYER_NORM IS CORRECT ON GPU!")
        print(">>> Bug is in the TRANSFORMER LAYER (self-attention or FFN)!")
    else:
        print(">>> CONV1D STAGE IS BROKEN ON GPU!")
        print(">>> The bug is in conv1d, transpose, pos_embed, or layer_norm on GPU!")

print("\n=== Also test on CPU (stock ctranslate2) ===")
print("Run this in temp_cpu_venv:")
print(f"  cd {zero_dir}")
print("  python -c \"import ctranslate2; import numpy as np; np.random.seed(12345); ...")
