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

print(f"Model: {model_path}")


def read_string(f):
    str_len = struct.unpack('<H', f.read(2))[0]
    return f.read(str_len).decode('utf-8').rstrip('\x00')


DTYPE_MAP = {0: (np.float32, 4), 1: (np.int8, 1), 2: (np.int16, 2),
             3: (np.int32, 4), 4: (np.float16, 2), 5: (np.float16, 2)}


def read_all_weights(filepath):
    weights = {}
    with open(str(filepath), 'rb') as f:
        binary_version = struct.unpack('<I', f.read(4))[0]
        if binary_version >= 2:
            read_string(f)
            struct.unpack('<I', f.read(4))
        num_variables = struct.unpack('<I', f.read(4))[0]
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
                type_id = {4: 0, 1: 1, 2: 2}.get(item_size, 0)
            raw = f.read(num_bytes)
            np_dtype = DTYPE_MAP.get(type_id, (np.float32, 4))[0]
            w = np.frombuffer(raw, dtype=np_dtype).copy().astype(np.float32)
            if dims:
                w = w.reshape(dims)
            weights[name] = w
    return weights


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
        out[b] = (w_flat @ im2col.T)
    if bias is not None:
        out += bias.reshape(1, -1, 1)
    return out


def self_attention(x, w_qkv, b_qkv, w_out, b_out, num_heads):
    B, T, D = x.shape
    head_dim = D // num_heads
    qkv = x @ w_qkv.T + b_qkv
    q, k, v = np.split(qkv, 3, axis=-1)
    q = q.reshape(B, T, num_heads, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(B, T, num_heads, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(B, T, num_heads, head_dim).transpose(0, 2, 1, 3)
    scale = 1.0 / np.sqrt(head_dim)
    attn = q @ k.transpose(0, 1, 3, 2) * scale
    attn_max = np.max(attn, axis=-1, keepdims=True)
    attn_exp = np.exp(attn - attn_max)
    attn_probs = attn_exp / np.sum(attn_exp, axis=-1, keepdims=True)
    context = attn_probs @ v
    context = context.transpose(0, 2, 1, 3).reshape(B, T, D)
    return context @ w_out.T + b_out


def ffn(x, w0, b0, w1, b1):
    hidden = gelu(x @ w0.T + b0)
    return hidden @ w1.T + b1


W = read_all_weights(Path(model_path) / "model.bin")
print(f"Loaded {len(W)} weights")

np.random.seed(12345)
features = np.random.randn(1, 80, 3000).astype(np.float32) * 0.1

print("\n=== NUMPY ENCODER (step by step) ===")

x = features.copy()
print(f"Input: {x.shape}, mean={np.mean(x):.6f}")

x = gelu(numpy_conv1d(x, W['encoder/conv1/weight'], W['encoder/conv1/bias'], stride=1, padding=1))
print(f"After conv1+GELU: {x.shape}, mean={np.mean(x):.6f}, std={np.std(x):.6f}")
print(f"  [0,:5,0]: {x[0,:5,0]}")

x = gelu(numpy_conv1d(x, W['encoder/conv2/weight'], W['encoder/conv2/bias'], stride=2, padding=1))
print(f"After conv2+GELU: {x.shape}, mean={np.mean(x):.6f}, std={np.std(x):.6f}")

x = x.transpose(0, 2, 1)
print(f"After transpose: {x.shape}")

pos = W['encoder/position_encodings/encodings']
x = x + pos[np.newaxis, :x.shape[1], :]
print(f"After pos embed: {x.shape}, mean={np.mean(x):.6f}, std={np.std(x):.6f}")
pre_transformer = x.copy()

num_heads = 6
sa_ln_g = W['encoder/layer_0/self_attention/layer_norm/gamma']
sa_ln_b = W['encoder/layer_0/self_attention/layer_norm/beta']
sa_w_qkv = W['encoder/layer_0/self_attention/linear_0/weight']
sa_b_qkv = W['encoder/layer_0/self_attention/linear_0/bias']
sa_w_out = W['encoder/layer_0/self_attention/linear_1/weight']
sa_b_out = W['encoder/layer_0/self_attention/linear_1/bias']

ffn_ln_g = W['encoder/layer_0/ffn/layer_norm/gamma']
ffn_ln_b = W['encoder/layer_0/ffn/layer_norm/beta']
ffn_w0 = W['encoder/layer_0/ffn/linear_0/weight']
ffn_b0 = W['encoder/layer_0/ffn/linear_0/bias']
ffn_w1 = W['encoder/layer_0/ffn/linear_1/weight']
ffn_b1 = W['encoder/layer_0/ffn/linear_1/bias']

normed = layer_norm(x, sa_ln_g, sa_ln_b)
print(f"After SA layer_norm: mean={np.mean(normed):.6f}, std={np.std(normed):.6f}")
attn_out = self_attention(normed, sa_w_qkv, sa_b_qkv, sa_w_out, sa_b_out, num_heads)
print(f"After self_attention: mean={np.mean(attn_out):.6f}, std={np.std(attn_out):.6f}")
x = x + attn_out
print(f"After SA residual: mean={np.mean(x):.6f}, std={np.std(x):.6f}")

normed2 = layer_norm(x, ffn_ln_g, ffn_ln_b)
ffn_out = ffn(normed2, ffn_w0, ffn_b0, ffn_w1, ffn_b1)
print(f"After FFN: mean={np.mean(ffn_out):.6f}, std={np.std(ffn_out):.6f}")
x = x + ffn_out
print(f"After FFN residual: mean={np.mean(x):.6f}, std={np.std(x):.6f}")

out_ln_g = W['encoder/layer_norm/gamma']
out_ln_b = W['encoder/layer_norm/beta']
output = layer_norm(x, out_ln_g, out_ln_b)
print(f"\nFinal output: {output.shape}, mean={np.mean(output):.6f}, std={np.std(output):.6f}")
print(f"  [0,0,:10]: {output[0,0,:10]}")

np.save(r"C:\Users\pinwa\projects\5700xt-rocm\dist\encoder_1layer_numpy.npy", output)

cpu_1layer = Path(r'C:\Users\pinwa\projects\5700xt-rocm\dist\encoder_1layer_cpu.npy')
gpu_1layer = Path(r'C:\Users\pinwa\projects\5700xt-rocm\dist\encoder_1layer_gpu.npy')

print("\n" + "="*60)
print("COMPARISONS")
print("="*60)

if cpu_1layer.exists():
    cpu = np.load(str(cpu_1layer))
    corr = np.corrcoef(output.flat, cpu.flat)[0, 1]
    diff = np.abs(output - cpu)
    print(f"  Numpy 1-layer vs CPU 1-layer: corr={corr:.6f}, max_diff={np.max(diff):.4f}")
    print(f"    numpy [0,0,:5]: {output[0,0,:5]}")
    print(f"    CPU   [0,0,:5]: {cpu[0,0,:5]}")

if gpu_1layer.exists():
    gpu = np.load(str(gpu_1layer))
    corr = np.corrcoef(output.flat, gpu.flat)[0, 1]
    diff = np.abs(output - gpu)
    print(f"  Numpy 1-layer vs GPU 1-layer: corr={corr:.6f}, max_diff={np.max(diff):.4f}")
    print(f"    numpy [0,0,:5]: {output[0,0,:5]}")
    print(f"    GPU   [0,0,:5]: {gpu[0,0,:5]}")

if cpu_1layer.exists() and gpu_1layer.exists():
    corr = np.corrcoef(cpu.flat, gpu.flat)[0, 1]
    diff = np.abs(cpu - gpu)
    print(f"  CPU 1-layer vs GPU 1-layer: corr={corr:.6f}, max_diff={np.max(diff):.4f}")

print("\nDONE")
