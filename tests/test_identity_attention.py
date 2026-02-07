import os, struct, shutil
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
import ctypes

hip_dll = ctypes.CDLL(str(next(
    p / 'bin' / 'amdhip64_6.dll'
    for p in sorted(Path(r'C:\Program Files\AMD\ROCm').iterdir(), reverse=True)
    if (p / 'bin' / 'amdhip64_6.dll').exists()
)))
hip_dll.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
hip_dll.hipMemcpy.restype = ctypes.c_int


def sv_to_numpy(sv):
    iface = sv.__cuda_array_interface__
    shape = iface['shape']
    total = 1
    for s in shape:
        total *= s
    host_arr = np.empty(shape, dtype=np.float32)
    hip_dll.hipMemcpy(
        ctypes.c_void_p(host_arr.ctypes.data),
        ctypes.c_void_p(iface['data'][0]),
        total * 4, 2)
    return host_arr


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


def read_model(filepath):
    variables = []
    with open(str(filepath), 'rb') as f:
        binary_version = struct.unpack('<I', f.read(4))[0]
        spec = read_string(f) if binary_version >= 2 else ""
        spec_revision = struct.unpack('<I', f.read(4))[0] if binary_version >= 2 else 1
        num_variables = struct.unpack('<I', f.read(4))[0]
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
                type_id = {4: 0, 2: 4, 1: 1}.get(item_size, 0)
            raw_data = f.read(num_bytes)
            np_dtype = DTYPE_MAP.get(type_id, (np.float32, 4))[0]
            weight = np.frombuffer(raw_data, dtype=np_dtype).copy().reshape(dims) if dims else np.frombuffer(raw_data, dtype=np_dtype).copy()
            variables.append({'name': name, 'dims': dims, 'type_id': type_id,
                              'num_bytes': num_bytes, 'data': weight, 'raw_data': raw_data})
        aliases = []
        if binary_version >= 3:
            remaining = f.read(4)
            if len(remaining) == 4:
                num_aliases = struct.unpack('<I', remaining)[0]
                for _ in range(num_aliases):
                    aliases.append((read_string(f), read_string(f)))
    return (binary_version, spec, spec_revision), variables, aliases


def write_model(filepath, header_info, variables, aliases=None):
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
                num_elements = 1
                for d in var['dims']:
                    num_elements *= d
                f.write(struct.pack('<B', elem_size))
                f.write(struct.pack('<I', num_elements))
            f.write(var['raw_data'])
        if binary_version >= 3:
            f.write(struct.pack('<I', len(aliases) if aliases else 0))
            if aliases:
                for alias, target in aliases:
                    write_string(f, alias)
                    write_string(f, target)


model_path = None
for p in Path(r'C:\Users\pinwa\.cache\huggingface\hub').rglob('models--Systran--faster-whisper-tiny/snapshots/*/model.bin'):
    model_path = str(p.parent)
    break

print(f"Source: {model_path}")
header, variables, aliases = read_model(Path(model_path) / 'model.bin')

# Create identity attention model:
# - Self-attention QKV weight = identity blocks, bias = 0
# - Output projection weight = identity, bias = 0
# - Layer norms = identity (gamma=1, beta=0)
# - Keep conv weights real, keep decoder weights real
# - Zero FFN weights to isolate attention

identity_dir = Path(r'C:\Users\pinwa\projects\5700xt-rocm\dist\model_identity_attn')
identity_dir.mkdir(parents=True, exist_ok=True)
for fname in ['vocabulary.json', 'vocabulary.txt', 'tokenizer.json', 'config.json']:
    src = Path(model_path) / fname
    if src.exists():
        shutil.copy2(str(src), str(identity_dir / fname))

d_model = 384
num_heads = 6
d_head = 64

for var in variables:
    name = var['name']

    # Encoder layer 0 self-attention: set to identity
    if 'encoder/layer_0/self_attention/linear_0' in name:
        if 'weight' in name:
            # QKV weight [1152, 384]: three stacked identity blocks
            w = np.zeros((d_model * 3, d_model), dtype=np.float16)
            for block in range(3):
                for i in range(d_model):
                    w[block * d_model + i, i] = 1.0
            var['data'] = w
            var['raw_data'] = w.tobytes()
            var['num_bytes'] = len(var['raw_data'])
            print(f"  Set {name} to identity blocks: {w.shape}")
        elif 'bias' in name:
            var['data'] = np.zeros_like(var['data'])
            var['raw_data'] = var['data'].tobytes()
            print(f"  Zeroed {name}")

    elif 'encoder/layer_0/self_attention/linear_1' in name:
        if 'weight' in name:
            # Output projection [384, 384]: identity
            w = np.eye(d_model, dtype=np.float16)
            var['data'] = w
            var['raw_data'] = w.tobytes()
            var['num_bytes'] = len(var['raw_data'])
            print(f"  Set {name} to identity: {w.shape}")
        elif 'bias' in name:
            var['data'] = np.zeros_like(var['data'])
            var['raw_data'] = var['data'].tobytes()
            print(f"  Zeroed {name}")

    # Zero FFN weights to isolate attention
    elif 'encoder/layer_0/ffn' in name and ('weight' in name or 'bias' in name):
        if 'layer_norm' not in name:
            var['data'] = np.zeros_like(var['data'])
            var['raw_data'] = var['data'].tobytes()
            print(f"  Zeroed {name}")

    # Set layer norms to identity
    elif 'encoder/layer_0' in name and 'layer_norm' in name:
        if 'gamma' in name:
            var['data'] = np.ones_like(var['data'])
            var['raw_data'] = var['data'].tobytes()
            print(f"  Set {name} to ones")
        elif 'beta' in name:
            var['data'] = np.zeros_like(var['data'])
            var['raw_data'] = var['data'].tobytes()
            print(f"  Zeroed {name}")

write_model(identity_dir / 'model.bin', header, variables, aliases)
print(f"\nWrote identity attention model to {identity_dir}")

# Run GPU encoder
print("\n" + "=" * 60)
print("GPU ENCODER — IDENTITY ATTENTION")
print("=" * 60)
np.random.seed(12345)
features = np.random.randn(1, 80, 3000).astype(np.float32) * 0.1
features_sv = ctranslate2.StorageView.from_array(features)

model = ctranslate2.models.Whisper(str(identity_dir), device="cuda", compute_type="float32")
out = model.encode(features_sv)
gpu_np = sv_to_numpy(out)
print(f"  Shape: {gpu_np.shape}")
print(f"  mean={np.mean(gpu_np):.6f}, std={np.std(gpu_np):.6f}")
print(f"  [0,0,:10]: {gpu_np[0,0,:10]}")
np.save(str(identity_dir / 'gpu_output.npy'), gpu_np)
del model

# Now compute expected output in numpy
# Re-read original model weights for conv pipeline
header_orig, vars_orig, _ = read_model(Path(model_path) / 'model.bin')
weights = {v['name']: v['data'].astype(np.float32) for v in vars_orig if v['name'].startswith('encoder/')}


def numpy_conv1d(x, weight, bias, stride=1, padding=1):
    batch, in_ch, in_len = x.shape
    out_ch, _, kernel = weight.shape
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
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


# Conv pipeline (using ORIGINAL weights — these are correct on GPU)
x = features.copy()
conv1_out = gelu(numpy_conv1d(x, weights['encoder/conv1/weight'], weights['encoder/conv1/bias'], stride=1, padding=1))
conv2_out = gelu(numpy_conv1d(conv1_out, weights['encoder/conv2/weight'], weights['encoder/conv2/bias'], stride=2, padding=1))
transposed = conv2_out.transpose(0, 2, 1)  # [1, 1500, 384]
pos_enc = weights['encoder/position_encodings/encodings']
pos_added = transposed + pos_enc[np.newaxis, :transposed.shape[1], :] if pos_enc.ndim == 2 else transposed + pos_enc[:, :transposed.shape[1], :]
print(f"\n  Pre-transformer: {pos_added.shape}, mean={np.mean(pos_added):.6f}")

# Final layer norm (encoder/layer_norm) — with ORIGINAL weights
ln_g = weights['encoder/layer_norm/gamma']
ln_b = weights['encoder/layer_norm/beta']


def layer_norm(x, gamma, beta, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gamma + beta


# Transformer layer with identity attention
# Pre-norm for self-attention: identity (gamma=1, beta=0)
sa_ln = layer_norm(pos_added, np.ones(d_model), np.zeros(d_model))

# QKV projection: input × weight^T = input × I = input for each of Q, K, V
# So Q = K = V = sa_ln
Q = sa_ln.copy()  # [1, 1500, 384]
K = sa_ln.copy()
V = sa_ln.copy()

# Reshape to heads
Q_heads = Q.reshape(1, 1500, num_heads, d_head).transpose(0, 2, 1, 3)  # [1, 6, 1500, 64]
K_heads = K.reshape(1, 1500, num_heads, d_head).transpose(0, 2, 1, 3)
V_heads = V.reshape(1, 1500, num_heads, d_head).transpose(0, 2, 1, 3)

# Attention scores
scale = 1.0 / np.sqrt(d_head)
scores = np.zeros((1, num_heads, 1500, 1500), dtype=np.float32)
for h in range(num_heads):
    scores[0, h] = scale * (Q_heads[0, h] @ K_heads[0, h].T)

# Softmax
scores_max = np.max(scores, axis=-1, keepdims=True)
scores_exp = np.exp(scores - scores_max)
attn_weights = scores_exp / np.sum(scores_exp, axis=-1, keepdims=True)

# Attention output
attn_out = np.zeros_like(V_heads)
for h in range(num_heads):
    attn_out[0, h] = attn_weights[0, h] @ V_heads[0, h]

# Combine heads
combined = attn_out.transpose(0, 2, 1, 3).reshape(1, 1500, d_model)

# Output projection = identity, bias = 0
attn_output = combined

# Residual connection
after_attn = pos_added + attn_output

# FFN is zeroed → identity residual
after_ffn = after_attn  # + 0

# Final encoder layer_norm (with ORIGINAL weights)
numpy_out = layer_norm(after_ffn, ln_g, ln_b)

print(f"  Numpy expected: {numpy_out.shape}, mean={np.mean(numpy_out):.6f}")
print(f"  [0,0,:10]: {numpy_out[0,0,:10]}")

# Compare
corr = np.corrcoef(gpu_np.flat, numpy_out.flat)[0, 1]
diff = np.abs(gpu_np - numpy_out)
verdict = "MATCH" if corr > 0.99 else ("PARTIAL" if corr > 0.5 else "NO MATCH")
print(f"\n  GPU vs numpy: corr={corr:.6f}, max_diff={np.max(diff):.4f}, mean_diff={np.mean(diff):.4f} [{verdict}]")

if corr > 0.99:
    print("\n>>> IDENTITY ATTENTION WORKS ON GPU!")
    print(">>> Bug is specific to real weight values — not structural")
else:
    print("\n>>> IDENTITY ATTENTION IS ALSO BROKEN ON GPU!")
    print(">>> Bug is structural — in the composition of operations")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
