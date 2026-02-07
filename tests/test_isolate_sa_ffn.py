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


def make_zeroed_model(orig_vars, zero_prefix, aliases):
    result = []
    for v in orig_vars:
        if v['name'].startswith('encoder/layer_') and '/layer_0/' not in v['name'] \
                and v['name'] != 'encoder/layer_norm/gamma' and v['name'] != 'encoder/layer_norm/beta':
            continue
        if v['name'].startswith(zero_prefix):
            np_dtype = DTYPE_MAP.get(v['type_id'], (np.float32, 4))[0]
            zero_data = np.zeros(v['num_bytes'] // np.dtype(np_dtype).itemsize, dtype=np_dtype).tobytes()
            result.append({**v, 'raw_data': zero_data})
        else:
            result.append(v)
    fa = [(a, t) for a, t in aliases if 'encoder/layer_' not in a or 'layer_0' in a]
    return result, fa


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
    x_padded = np.pad(x, ((0,0),(0,0),(padding,padding))) if padding > 0 else x
    w_flat = weight.reshape(out_ch, -1)
    out = np.zeros((batch, out_ch, out_len), dtype=np.float32)
    for b in range(batch):
        im2col = np.zeros((out_len, w_flat.shape[1]), dtype=np.float32)
        for t in range(out_len):
            im2col[t] = x_padded[b, :, t*stride:t*stride+kernel].reshape(-1)
        out[b] = w_flat @ im2col.T
    if bias is not None:
        out += bias.reshape(1, -1, 1)
    return out


def self_attention(x, w_qkv, b_qkv, w_out, b_out, num_heads):
    B, T, D = x.shape
    hd = D // num_heads
    qkv = x @ w_qkv.T + b_qkv
    q, k, v = np.split(qkv, 3, axis=-1)
    q = q.reshape(B, T, num_heads, hd).transpose(0, 2, 1, 3)
    k = k.reshape(B, T, num_heads, hd).transpose(0, 2, 1, 3)
    v = v.reshape(B, T, num_heads, hd).transpose(0, 2, 1, 3)
    attn = q @ k.transpose(0, 1, 3, 2) / np.sqrt(hd)
    attn = np.exp(attn - np.max(attn, axis=-1, keepdims=True))
    attn = attn / np.sum(attn, axis=-1, keepdims=True)
    ctx = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, D)
    return ctx @ w_out.T + b_out


header, variables, aliases = read_model(Path(model_path) / "model.bin")

import shutil
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
pre_transformer = x.copy()
print(f"Pre-transformer: {x.shape}, mean={np.mean(x):.6f}")

# ===== Test A: Zero FFN, keep self-attention =====
print("\n" + "="*60)
print("TEST A: ZERO FFN, KEEP SELF-ATTENTION")
print("="*60)

sa_zero_ffn = pre_transformer.copy()
sa_ln_g = W['encoder/layer_0/self_attention/layer_norm/gamma']
sa_ln_b = W['encoder/layer_0/self_attention/layer_norm/beta']
normed = layer_norm(sa_zero_ffn, sa_ln_g, sa_ln_b)
attn_out = self_attention(normed,
    W['encoder/layer_0/self_attention/linear_0/weight'],
    W['encoder/layer_0/self_attention/linear_0/bias'],
    W['encoder/layer_0/self_attention/linear_1/weight'],
    W['encoder/layer_0/self_attention/linear_1/bias'], 6)
sa_zero_ffn = sa_zero_ffn + attn_out
out_ln_g = W['encoder/layer_norm/gamma']
out_ln_b = W['encoder/layer_norm/beta']
numpy_sa_only = layer_norm(sa_zero_ffn, out_ln_g, out_ln_b)
print(f"Numpy SA-only output: mean={np.mean(numpy_sa_only):.6f}")
print(f"  [0,0,:5]: {numpy_sa_only[0,0,:5]}")

zero_ffn_dir = base / 'model_zero_ffn'
zero_ffn_dir.mkdir(parents=True, exist_ok=True)
for fn in ['vocabulary.json', 'vocabulary.txt', 'tokenizer.json', 'config.json']:
    src = Path(model_path) / fn
    if src.exists():
        shutil.copy2(str(src), str(zero_ffn_dir / fn))

vars_zf, aliases_zf = make_zeroed_model(variables, 'encoder/layer_0/ffn/', aliases)
write_model(zero_ffn_dir / 'model.bin', header, vars_zf, aliases_zf)

model_zf = ctranslate2.models.Whisper(str(zero_ffn_dir), device="cuda", compute_type="float32")
features_sv = ctranslate2.StorageView.from_array(features)
gpu_zf = np.array(model_zf.encode(features_sv, to_cpu=True))
print(f"GPU SA-only output: mean={np.mean(gpu_zf):.6f}")
print(f"  [0,0,:5]: {gpu_zf[0,0,:5]}")
corr = np.corrcoef(gpu_zf.flat, numpy_sa_only.flat)[0, 1]
diff = np.abs(gpu_zf - numpy_sa_only)
print(f"Correlation: {corr:.6f}, max_diff: {np.max(diff):.4f}")
if corr > 0.99:
    print(">>> SELF-ATTENTION IS CORRECT ON GPU! Bug must be in FFN.")
else:
    print(">>> SELF-ATTENTION IS BROKEN ON GPU!")
del model_zf

# ===== Test B: Zero self-attention, keep FFN =====
print("\n" + "="*60)
print("TEST B: ZERO SELF-ATTENTION, KEEP FFN")
print("="*60)

ffn_only = pre_transformer.copy()
ffn_ln_g = W['encoder/layer_0/ffn/layer_norm/gamma']
ffn_ln_b = W['encoder/layer_0/ffn/layer_norm/beta']
normed2 = layer_norm(ffn_only, ffn_ln_g, ffn_ln_b)
hidden = gelu(normed2 @ W['encoder/layer_0/ffn/linear_0/weight'].T + W['encoder/layer_0/ffn/linear_0/bias'])
ffn_out = hidden @ W['encoder/layer_0/ffn/linear_1/weight'].T + W['encoder/layer_0/ffn/linear_1/bias']
ffn_only = ffn_only + ffn_out
numpy_ffn_only = layer_norm(ffn_only, out_ln_g, out_ln_b)
print(f"Numpy FFN-only output: mean={np.mean(numpy_ffn_only):.6f}")
print(f"  [0,0,:5]: {numpy_ffn_only[0,0,:5]}")

zero_sa_dir = base / 'model_zero_sa'
zero_sa_dir.mkdir(parents=True, exist_ok=True)
for fn in ['vocabulary.json', 'vocabulary.txt', 'tokenizer.json', 'config.json']:
    src = Path(model_path) / fn
    if src.exists():
        shutil.copy2(str(src), str(zero_sa_dir / fn))

vars_zs, aliases_zs = make_zeroed_model(variables, 'encoder/layer_0/self_attention/', aliases)
write_model(zero_sa_dir / 'model.bin', header, vars_zs, aliases_zs)

model_zs = ctranslate2.models.Whisper(str(zero_sa_dir), device="cuda", compute_type="float32")
gpu_zs = np.array(model_zs.encode(features_sv, to_cpu=True))
print(f"GPU FFN-only output: mean={np.mean(gpu_zs):.6f}")
print(f"  [0,0,:5]: {gpu_zs[0,0,:5]}")
corr2 = np.corrcoef(gpu_zs.flat, numpy_ffn_only.flat)[0, 1]
diff2 = np.abs(gpu_zs - numpy_ffn_only)
print(f"Correlation: {corr2:.6f}, max_diff: {np.max(diff2):.4f}")
if corr2 > 0.99:
    print(">>> FFN IS CORRECT ON GPU! Bug must be in self-attention.")
else:
    print(">>> FFN IS BROKEN ON GPU!")
del model_zs

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"  Self-attention test: corr={corr:.6f} {'PASS' if corr > 0.99 else 'FAIL'}")
print(f"  FFN test: corr={corr2:.6f} {'PASS' if corr2 > 0.99 else 'FAIL'}")
