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
import shutil

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
        bv = struct.unpack('<I', f.read(4))[0]
        spec, sr = "", 1
        if bv >= 2:
            spec = read_string(f)
            sr = struct.unpack('<I', f.read(4))[0]
        nv = struct.unpack('<I', f.read(4))[0]
        for _ in range(nv):
            name = read_string(f)
            rank = struct.unpack('<B', f.read(1))[0]
            dims = [struct.unpack('<I', f.read(4))[0] for _ in range(rank)]
            if bv >= 4:
                tid = struct.unpack('<B', f.read(1))[0]
                nb = struct.unpack('<I', f.read(4))[0]
            else:
                isz = struct.unpack('<B', f.read(1))[0]
                ni = struct.unpack('<I', f.read(4))[0]
                nb = ni * isz
                tid = {4: 0}.get(isz, 0)
            raw = f.read(nb)
            variables.append({'name': name, 'dims': dims, 'type_id': tid, 'num_bytes': nb, 'raw_data': raw})
        aliases = []
        if bv >= 3:
            r = f.read(4)
            if len(r) == 4:
                na = struct.unpack('<I', r)[0]
                for _ in range(na):
                    aliases.append((read_string(f), read_string(f)))
    return (bv, spec, sr), variables, aliases


def write_model(filepath, header, variables, aliases):
    bv, spec, sr = header
    with open(str(filepath), 'wb') as f:
        f.write(struct.pack('<I', bv))
        if bv >= 2:
            write_string(f, spec)
            f.write(struct.pack('<I', sr))
        f.write(struct.pack('<I', len(variables)))
        for v in variables:
            write_string(f, v['name'])
            f.write(struct.pack('<B', len(v['dims'])))
            for d in v['dims']:
                f.write(struct.pack('<I', d))
            if bv >= 4:
                f.write(struct.pack('<B', v['type_id']))
                f.write(struct.pack('<I', v['num_bytes']))
            f.write(v['raw_data'])
        if bv >= 3:
            f.write(struct.pack('<I', len(aliases)))
            for a, t in aliases:
                write_string(f, a)
                write_string(f, t)


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def layer_norm(x, gamma, beta, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gamma + beta


def numpy_conv1d(x, weight, bias, stride=1, padding=1):
    batch, in_ch, in_len = x.shape
    out_ch, _, kernel = weight.shape
    out_len = (in_len + 2 * padding - kernel) // stride + 1
    x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding))) if padding > 0 else x
    w_flat = weight.reshape(out_ch, -1)
    out = np.zeros((batch, out_ch, out_len), dtype=np.float32)
    for b in range(batch):
        im2col = np.zeros((out_len, w_flat.shape[1]), dtype=np.float32)
        for t in range(out_len):
            im2col[t] = x_padded[b, :, t * stride:t * stride + kernel].reshape(-1)
        out[b] = w_flat @ im2col.T
    if bias is not None:
        out += bias.reshape(1, -1, 1)
    return out


header, variables, aliases = read_model(Path(model_path) / "model.bin")
base = Path(r'C:\Users\pinwa\projects\5700xt-rocm\dist')

W = {}
for v in variables:
    np_dtype = DTYPE_MAP.get(v['type_id'], (np.float32, 4))[0]
    w = np.frombuffer(v['raw_data'], dtype=np_dtype).copy().astype(np.float32)
    if v['dims']:
        w = w.reshape(v['dims'])
    W[v['name']] = w

np.random.seed(12345)
features = np.random.randn(1, 80, 3000).astype(np.float32) * 0.1

x = features.copy()
x = gelu(numpy_conv1d(x, W['encoder/conv1/weight'], W['encoder/conv1/bias'], stride=1, padding=1))
x = gelu(numpy_conv1d(x, W['encoder/conv2/weight'], W['encoder/conv2/bias'], stride=2, padding=1))
x = x.transpose(0, 2, 1)
pos = W['encoder/position_encodings/encodings']
x = x + pos[np.newaxis, :x.shape[1], :]
pre_attn = x.copy()
print(f"Pre-attention input: {pre_attn.shape}")

NUM_HEADS = 6
D_MODEL = 384
D_HEAD = D_MODEL // NUM_HEADS
T = pre_attn.shape[1]

qkv_bias = W['encoder/layer_0/self_attention/linear_0/bias']
out_weight = W['encoder/layer_0/self_attention/linear_1/weight']
out_bias = W['encoder/layer_0/self_attention/linear_1/bias']
sa_ln_g = W['encoder/layer_0/self_attention/layer_norm/gamma']
sa_ln_b = W['encoder/layer_0/self_attention/layer_norm/beta']
out_ln_g = W['encoder/layer_norm/gamma']
out_ln_b = W['encoder/layer_norm/beta']

print(f"\nQKV bias shape: {qkv_bias.shape}")
print(f"Output weight shape: {out_weight.shape}")

# Numpy reference: QKV weight = 0, so attention input = layer_norm(pre_attn) * 0 + bias = bias
normed = layer_norm(pre_attn, sa_ln_g, sa_ln_b)

# With QKV weight=0: fused_proj = bias (same at every time step)
fused = np.tile(qkv_bias, (1, T, 1))  # [1, 1500, 1152]

# Split into Q, K, V
q_bias, k_bias, v_bias = np.split(fused, 3, axis=-1)  # each [1, 1500, 384]

# Reshape to multi-head: [1, T, H, D_HEAD] → [1, H, T, D_HEAD]
Q = q_bias.reshape(1, T, NUM_HEADS, D_HEAD).transpose(0, 2, 1, 3)
K = k_bias.reshape(1, T, NUM_HEADS, D_HEAD).transpose(0, 2, 1, 3)
V = v_bias.reshape(1, T, NUM_HEADS, D_HEAD).transpose(0, 2, 1, 3)

print(f"\nWith QKV weight=0:")
print(f"  Q[0,0,0,:5] = {Q[0,0,0,:5]} (same at every position)")
print(f"  Q[0,0,1,:5] = {Q[0,0,1,:5]} (should be identical)")

# Q @ K^T / sqrt(d_head) — every element in 1500×1500 matrix is the same
scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(D_HEAD)
print(f"  Attention scores [0,0,0,:3] = {scores[0,0,0,:3]} (all elements should be equal)")
print(f"  Attention scores [0,0,1,:3] = {scores[0,0,1,:3]}")

# Softmax of constant row = uniform (1/T)
attn = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
attn = attn / np.sum(attn, axis=-1, keepdims=True)
print(f"  Softmax [0,0,0,:3] = {attn[0,0,0,:3]} (should be ~{1/T:.6f})")

# attn @ V = V (weighted average of identical values)
context = attn @ V
print(f"  Context == V? max_diff = {np.max(np.abs(context - V)):.2e}")

# combine_heads: [1, H, T, D_HEAD] → [1, T, H, D_HEAD] → [1, T, D_MODEL]
combined = context.transpose(0, 2, 1, 3).reshape(1, T, D_MODEL)

# Output projection
attn_out = combined @ out_weight.T + out_bias

# Residual + layer norm
result = pre_attn + attn_out
numpy_output = layer_norm(result, out_ln_g, out_ln_b)
print(f"\nNumpy output: mean={np.mean(numpy_output):.6f}, [0,0,:5]={numpy_output[0,0,:5]}")

# Create model with QKV weight=0, FFN=0
print("\n" + "=" * 60)
print("Creating model: QKV weight=0, FFN=0")
print("=" * 60)

model_dir = base / 'model_zero_qkv_weight'
model_dir.mkdir(parents=True, exist_ok=True)
for fn in ['vocabulary.json', 'vocabulary.txt', 'tokenizer.json', 'config.json']:
    src = Path(model_path) / fn
    if src.exists():
        shutil.copy2(str(src), str(model_dir / fn))

new_vars = []
for v in variables:
    if v['name'].startswith('encoder/layer_') and '/layer_0/' not in v['name'] \
            and v['name'] != 'encoder/layer_norm/gamma' and v['name'] != 'encoder/layer_norm/beta':
        continue
    np_dtype = DTYPE_MAP.get(v['type_id'], (np.float32, 4))[0]
    if v['name'] == 'encoder/layer_0/self_attention/linear_0/weight':
        print(f"  Zeroing: {v['name']} {v['dims']}")
        zero_data = np.zeros(v['num_bytes'] // np.dtype(np_dtype).itemsize, dtype=np_dtype).tobytes()
        new_vars.append({**v, 'raw_data': zero_data})
    elif v['name'].startswith('encoder/layer_0/ffn/'):
        print(f"  Zeroing: {v['name']} {v['dims']}")
        zero_data = np.zeros(v['num_bytes'] // np.dtype(np_dtype).itemsize, dtype=np_dtype).tobytes()
        new_vars.append({**v, 'raw_data': zero_data})
    else:
        new_vars.append(v)

fa = [(a, t) for a, t in aliases if 'encoder/layer_' not in a or 'layer_0' in a]
write_model(model_dir / 'model.bin', header, new_vars, fa)

model = ctranslate2.models.Whisper(str(model_dir), device="cuda", compute_type="float32")
features_sv = ctranslate2.StorageView.from_array(features)
gpu_output = np.array(model.encode(features_sv, to_cpu=True))

print(f"\nGPU output: mean={np.mean(gpu_output):.6f}, [0,0,:5]={gpu_output[0,0,:5]}")
print(f"Numpy output: mean={np.mean(numpy_output):.6f}, [0,0,:5]={numpy_output[0,0,:5]}")

corr = np.corrcoef(gpu_output.flat, numpy_output.flat)[0, 1]
max_diff = np.max(np.abs(gpu_output - numpy_output))
print(f"\nCorrelation: {corr:.6f}")
print(f"Max diff: {max_diff:.6f}")

if corr > 0.99:
    print(">>> PASS: Trivial attention (constant Q=K=V) works correctly!")
    print(">>> Bug only manifests with non-trivial data patterns")
else:
    print(">>> FAIL: Even trivial attention is broken!")
    print(">>> Bug is in a basic operation (split_heads, softmax, combine_heads, or output proj)")

    # Extra diagnostics: what does the GPU output look like?
    print(f"\n  GPU output variance across time: {np.var(gpu_output, axis=1).mean():.6f}")
    print(f"  Numpy output variance across time: {np.var(numpy_output, axis=1).mean():.6f}")
    print(f"  GPU first 10 values: {gpu_output[0,0,:10]}")
    print(f"  Numpy first 10 values: {numpy_output[0,0,:10]}")

del model
