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
model_bin = Path(model_path) / "model.bin"

DTYPE_MAP = {0: (np.float32, 4), 1: (np.int8, 1), 2: (np.int16, 2),
             3: (np.int32, 4), 4: (np.float16, 2), 5: (np.float16, 2)}


def read_string(f):
    str_len = struct.unpack('<H', f.read(2))[0]
    return f.read(str_len).decode('utf-8').rstrip('\x00')


def write_string(f, s):
    s_bytes = s.encode('utf-8')
    f.write(struct.pack('<H', len(s_bytes) + 1))
    f.write(s_bytes)
    f.write(struct.pack('<B', 0))


def read_model_variables(filepath):
    variables = []
    with open(str(filepath), 'rb') as f:
        binary_version = struct.unpack('<I', f.read(4))[0]
        print(f"  Binary version: {binary_version}")

        spec = ""
        spec_revision = 1
        if binary_version >= 2:
            spec = read_string(f)
            spec_revision = struct.unpack('<I', f.read(4))[0]
            print(f"  Spec: '{spec}', revision: {spec_revision}")

        num_variables = struct.unpack('<I', f.read(4))[0]
        print(f"  Variables: {num_variables}")
        header_info = (binary_version, spec, spec_revision)

        for i in range(num_variables):
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
                type_id = {4: 0, 1: 1, 2: 2}.get(item_size, 0)

            raw_data = f.read(num_bytes)

            np_dtype, elem_size = DTYPE_MAP.get(type_id, (np.float32, 4))
            num_elements = 1
            for d in dims:
                num_elements *= d

            weight = np.frombuffer(raw_data, dtype=np_dtype).copy()
            if dims:
                weight = weight.reshape(dims)

            variables.append({
                'name': name, 'dims': dims, 'type_id': type_id,
                'num_bytes': num_bytes, 'data': weight, 'raw_data': raw_data
            })

        aliases = []
        if binary_version >= 3:
            remaining = f.read(4)
            if len(remaining) == 4:
                num_aliases = struct.unpack('<I', remaining)[0]
                for _ in range(num_aliases):
                    alias = read_string(f)
                    target = read_string(f)
                    aliases.append((alias, target))

    return header_info, variables, aliases


def write_model_bin(filepath, header_info, variables, aliases=None):
    binary_version, spec, spec_revision = header_info
    with open(str(filepath), 'wb') as f:
        f.write(struct.pack('<I', binary_version))
        if binary_version >= 2:
            write_string(f, spec)
            f.write(struct.pack('<I', spec_revision))

        f.write(struct.pack('<I', len(variables)))
        for var in variables:
            write_string(f, var['name'])
            f.write(struct.pack('<B', len(var['dims'])))
            for d in var['dims']:
                f.write(struct.pack('<I', d))
            if binary_version >= 4:
                f.write(struct.pack('<B', var['type_id']))
                f.write(struct.pack('<I', var['num_bytes']))
            else:
                np_dtype, elem_size = DTYPE_MAP.get(var['type_id'], (np.float32, 4))
                num_elements = max(1, 1)
                for d in var['dims']:
                    num_elements *= d
                f.write(struct.pack('<B', elem_size))
                f.write(struct.pack('<I', num_elements))
            f.write(var['raw_data'])

        if binary_version >= 3 and aliases:
            f.write(struct.pack('<I', len(aliases)))
            for alias, target in aliases:
                write_string(f, alias)
                write_string(f, target)
        elif binary_version >= 3:
            f.write(struct.pack('<I', 0))


print("=== Reading model.bin ===")
header_info, all_vars, aliases = read_model_variables(model_bin)
print(f"\nAll model variables ({len(all_vars)}):")
for v in all_vars:
    prefix = "  [ENC] " if v['name'].startswith('encoder/') else "  [DEC] "
    if not v['name'].startswith('encoder/') and not v['name'].startswith('decoder/'):
        prefix = "  [TOP] "
    print(f"{prefix}{v['name']}: shape={v['dims']}, type_id={v['type_id']}, bytes={v['num_bytes']}")

if aliases:
    print(f"\nAliases ({len(aliases)}):")
    for a, t in aliases:
        print(f"  {a} -> {t}")

encoder_conv_vars = []
encoder_pos_vars = []
encoder_norm_vars = []
encoder_layer_vars = []
decoder_vars = []
other_vars = []

for v in all_vars:
    name = v['name']
    if name.startswith('encoder/conv'):
        encoder_conv_vars.append(v)
    elif name.startswith('encoder/position'):
        encoder_pos_vars.append(v)
    elif name.startswith('encoder/layer_norm'):
        encoder_norm_vars.append(v)
    elif name.startswith('encoder/layer_'):
        encoder_layer_vars.append(v)
    elif name.startswith('decoder/'):
        decoder_vars.append(v)
    else:
        other_vars.append(v)

print(f"\nEncoder conv: {len(encoder_conv_vars)}, pos: {len(encoder_pos_vars)}, "
      f"norm: {len(encoder_norm_vars)}, layers: {len(encoder_layer_vars)}, "
      f"decoder: {len(decoder_vars)}, other: {len(other_vars)}")

layer_nums = set()
for v in encoder_layer_vars:
    for p in v['name'].split('/'):
        if p.startswith('layer_') and p != 'layer_norm':
            layer_nums.add(int(p.split('_')[1]))
layer_nums = sorted(layer_nums)
print(f"Transformer layers: {layer_nums}")

import shutil

truncated_dir = Path(r'C:\Users\pinwa\projects\5700xt-rocm\dist\truncated_model_0layers')
truncated_dir.mkdir(parents=True, exist_ok=True)
for fname in ['vocabulary.json', 'vocabulary.txt', 'tokenizer.json', 'config.json']:
    src = Path(model_path) / fname
    if src.exists():
        shutil.copy2(str(src), str(truncated_dir / fname))

vars_0layers = encoder_conv_vars + encoder_pos_vars + encoder_norm_vars + decoder_vars + other_vars
filtered_aliases = [(a, t) for a, t in aliases
                    if not any(f'encoder/layer_{n}/' in a or f'encoder/layer_{n}/' in t for n in layer_nums)]
write_model_bin(truncated_dir / 'model.bin', header_info, vars_0layers, filtered_aliases)
print(f"\nWrote 0-layer model: {len(vars_0layers)} vars (removed {len(encoder_layer_vars)})")

truncated_dir_1 = Path(r'C:\Users\pinwa\projects\5700xt-rocm\dist\truncated_model_1layer')
truncated_dir_1.mkdir(parents=True, exist_ok=True)
for fname in ['vocabulary.json', 'vocabulary.txt', 'tokenizer.json', 'config.json']:
    src = Path(model_path) / fname
    if src.exists():
        shutil.copy2(str(src), str(truncated_dir_1 / fname))

layer0_vars = [v for v in encoder_layer_vars if '/layer_0/' in v['name']]
vars_1layer = encoder_conv_vars + encoder_pos_vars + encoder_norm_vars + layer0_vars + decoder_vars + other_vars
aliases_1 = [(a, t) for a, t in aliases
             if not any(f'encoder/layer_{n}/' in a or f'encoder/layer_{n}/' in t for n in layer_nums if n > 0)]
write_model_bin(truncated_dir_1 / 'model.bin', header_info, vars_1layer, aliases_1)
print(f"Wrote 1-layer model: {len(vars_1layer)} vars")

np.random.seed(12345)
features = np.random.randn(1, 80, 3000).astype(np.float32) * 0.1
features_sv = ctranslate2.StorageView.from_array(features)

print("\n" + "="*60)
print("FULL MODEL (GPU)")
print("="*60)
model_full = ctranslate2.models.Whisper(model_path, device="cuda", compute_type="float32")
out_full = model_full.encode(features_sv, to_cpu=True)
out_full_np = np.array(out_full)
print(f"  Shape: {out_full_np.shape}")
print(f"  mean={np.mean(out_full_np):.6f}, std={np.std(out_full_np):.6f}")
print(f"  [0,0,:5]: {out_full_np[0,0,:5]}")
del model_full

cpu_ref = Path(r"C:\Users\pinwa\projects\5700xt-rocm\encoder_output_cpu.npy")
cpu_np = None
if cpu_ref.exists():
    cpu_np = np.load(str(cpu_ref))
    diff = np.abs(out_full_np - cpu_np)
    corr = np.corrcoef(out_full_np.flat, cpu_np.flat)[0, 1]
    print(f"  vs stock CPU ref: max_diff={np.max(diff):.4f}, mean_diff={np.mean(diff):.4f}, corr={corr:.6f}")

print("\n" + "="*60)
print("0-LAYER MODEL (GPU) = conv1d*2 + transpose + pos_embed + layer_norm")
print("="*60)
out_0_np = None
try:
    model_0 = ctranslate2.models.Whisper(str(truncated_dir), device="cuda", compute_type="float32")
    out_0 = model_0.encode(features_sv, to_cpu=True)
    out_0_np = np.array(out_0)
    print(f"  Shape: {out_0_np.shape}")
    print(f"  mean={np.mean(out_0_np):.6f}, std={np.std(out_0_np):.6f}")
    print(f"  min={np.min(out_0_np):.6f}, max={np.max(out_0_np):.6f}")
    print(f"  NaN: {np.any(np.isnan(out_0_np))}, Inf: {np.any(np.isinf(out_0_np))}")
    print(f"  [0,0,:10]: {out_0_np[0,0,:10]}")
    np.save(r"C:\Users\pinwa\projects\5700xt-rocm\dist\encoder_0layers_gpu.npy", out_0_np)
    del model_0
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback; traceback.print_exc()

print("\n" + "="*60)
print("1-LAYER MODEL (GPU) = above + 1 transformer layer")
print("="*60)
out_1_np = None
try:
    model_1 = ctranslate2.models.Whisper(str(truncated_dir_1), device="cuda", compute_type="float32")
    out_1 = model_1.encode(features_sv, to_cpu=True)
    out_1_np = np.array(out_1)
    print(f"  Shape: {out_1_np.shape}")
    print(f"  mean={np.mean(out_1_np):.6f}, std={np.std(out_1_np):.6f}")
    print(f"  NaN: {np.any(np.isnan(out_1_np))}, Inf: {np.any(np.isinf(out_1_np))}")
    print(f"  [0,0,:10]: {out_1_np[0,0,:10]}")
    np.save(r"C:\Users\pinwa\projects\5700xt-rocm\dist\encoder_1layer_gpu.npy", out_1_np)
    del model_1
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback; traceback.print_exc()

print("\n" + "="*60)
print("FULL MODEL (CPU) — custom build")
print("="*60)
out_cpu_np = None
out_cpu_0_np = None
try:
    model_cpu = ctranslate2.models.Whisper(model_path, device="cpu", compute_type="float32")
    out_cpu = model_cpu.encode(features_sv)
    out_cpu_np = np.array(out_cpu)
    print(f"  Shape: {out_cpu_np.shape}")
    print(f"  mean={np.mean(out_cpu_np):.6f}, std={np.std(out_cpu_np):.6f}")
    print(f"  [0,0,:5]: {out_cpu_np[0,0,:5]}")

    if cpu_np is not None:
        diff_cpu = np.abs(out_cpu_np - cpu_np)
        corr_cpu = np.corrcoef(out_cpu_np.flat, cpu_np.flat)[0, 1]
        print(f"  vs stock CPU: max_diff={np.max(diff_cpu):.6f}, corr={corr_cpu:.6f}")
        if corr_cpu > 0.99:
            print(f"  >>> CUSTOM BUILD CPU MATCHES STOCK CPU!")
        else:
            print(f"  >>> CUSTOM BUILD CPU DOES NOT MATCH!")

    print("\n  0-LAYER MODEL (CPU):")
    model_cpu_0 = ctranslate2.models.Whisper(str(truncated_dir), device="cpu", compute_type="float32")
    out_cpu_0 = model_cpu_0.encode(features_sv)
    out_cpu_0_np = np.array(out_cpu_0)
    print(f"    Shape: {out_cpu_0_np.shape}, mean={np.mean(out_cpu_0_np):.6f}")
    print(f"    [0,0,:10]: {out_cpu_0_np[0,0,:10]}")
    np.save(r"C:\Users\pinwa\projects\5700xt-rocm\dist\encoder_0layers_cpu.npy", out_cpu_0_np)
    del model_cpu, model_cpu_0
except Exception as e:
    print(f"  CPU failed: {e}")
    import traceback; traceback.print_exc()

print("\n" + "="*60)
print("MANUAL NUMPY CONV1D — ground truth")
print("="*60)

weights = {v['name']: v['data'].astype(np.float32) for v in all_vars if v['name'].startswith('encoder/')}
print(f"  Encoder weights: {len(weights)}")
for name in sorted(weights.keys()):
    if 'conv' in name or 'position' in name or 'layer_norm' in name:
        print(f"    {name}: {weights[name].shape}")


def numpy_conv1d(x, weight, bias, stride=1, padding=1):
    batch, in_ch, in_len = x.shape
    out_ch, in_ch_w, kernel = weight.shape
    out_len = (in_len + 2 * padding - kernel) // stride + 1
    x_padded = np.pad(x, ((0,0),(0,0),(padding,padding))) if padding > 0 else x
    w_flat = weight.reshape(out_ch, -1)
    out = np.zeros((batch, out_ch, out_len), dtype=np.float32)
    for b in range(batch):
        for t in range(out_len):
            patch = x_padded[b, :, t*stride:t*stride+kernel].reshape(-1)
            out[b, :, t] = w_flat @ patch
    if bias is not None:
        out += bias.reshape(1, -1, 1)
    return out


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


x = features.copy()
print(f"\n  Conv1: {weights['encoder/conv1/weight'].shape}")
conv1_out = gelu(numpy_conv1d(x, weights['encoder/conv1/weight'], weights['encoder/conv1/bias'], stride=1, padding=1))
print(f"  -> {conv1_out.shape}, mean={np.mean(conv1_out):.6f}")

print(f"  Conv2: {weights['encoder/conv2/weight'].shape}")
conv2_out = gelu(numpy_conv1d(conv1_out, weights['encoder/conv2/weight'], weights['encoder/conv2/bias'], stride=2, padding=1))
print(f"  -> {conv2_out.shape}, mean={np.mean(conv2_out):.6f}")

transposed = conv2_out.transpose(0, 2, 1)
pos_enc = weights['encoder/position_encodings/encodings']
print(f"  Position encodings: shape={pos_enc.shape}")
if pos_enc.ndim == 2:
    pos_added = transposed + pos_enc[np.newaxis, :transposed.shape[1], :]
else:
    pos_added = transposed + pos_enc[:, :transposed.shape[1], :]
print(f"  After transpose+pos: mean={np.mean(pos_added):.6f}")

ln_w = weights['encoder/layer_norm/gamma']
ln_b = weights['encoder/layer_norm/beta']
mean_v = np.mean(pos_added, axis=-1, keepdims=True)
var_v = np.var(pos_added, axis=-1, keepdims=True)
ln_out = (pos_added - mean_v) / np.sqrt(var_v + 1e-5) * ln_w + ln_b
print(f"  Layer norm: {ln_out.shape}, mean={np.mean(ln_out):.6f}")
print(f"  [0,0,:10]: {ln_out[0,0,:10]}")
np.save(r"C:\Users\pinwa\projects\5700xt-rocm\dist\encoder_0layers_numpy.npy", ln_out)

print("\n" + "="*60)
print("COMPARISONS")
print("="*60)

def compare(name, a, b):
    if a is None or b is None:
        print(f"  {name}: SKIPPED (data unavailable)")
        return
    corr = np.corrcoef(a.flat, b.flat)[0, 1]
    diff = np.abs(a - b)
    verdict = "MATCH" if corr > 0.99 else ("PARTIAL" if corr > 0.5 else "NO MATCH")
    print(f"  {name}: corr={corr:.6f}, max_diff={np.max(diff):.4f}, mean_diff={np.mean(diff):.4f} [{verdict}]")

compare("GPU 0-layer vs numpy", out_0_np, ln_out)
compare("CPU 0-layer vs numpy", out_cpu_0_np, ln_out)
compare("GPU 0-layer vs CPU 0-layer", out_0_np, out_cpu_0_np)
compare("GPU full vs CPU full (custom)", out_full_np, out_cpu_np)
if cpu_np is not None:
    compare("GPU full vs stock CPU ref", out_full_np, cpu_np)
    compare("CPU full (custom) vs stock CPU ref", out_cpu_np, cpu_np)

print("\n" + "="*60)
print("DONE")
print("="*60)
