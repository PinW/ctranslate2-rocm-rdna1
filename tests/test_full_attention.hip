// test_full_attention.hip — Complete self-attention pipeline test for gfx1010
//
// Tests the FULL Whisper tiny encoder self-attention data flow:
//   1. QKV projection GEMM: input[T,384] x weight[1152,384]^T + bias
//   2. split_heads: reshape [1,T,18,64] -> transpose_0213 -> [1,18,T,64]
//   3. Split axis=1: [1,18,T,64] -> Q[1,6,T,64], K[1,6,T,64], V[1,6,T,64]
//   4. Q@K^T with scale: strided batched GEMM (6 batches, alpha=0.125)
//   5. Softmax: per-row softmax of scores[6,T,T]
//   6. attn@V: strided batched GEMM (6 batches)
//   7. combine_heads: transpose_0213 [1,6,T,64] -> [1,T,6,64] -> reshape [1,T,384]
//   8. Output projection GEMM: context[T,384] x out_weight[384,384]^T + bias
//   9. Residual add: output += input
//
// Compile:
//   hipcc -O2 test_full_attention.hip -o test_full_attention.exe -lhipblas --offload-arch=gfx1010 '-D_MSVC_STL_DOOM_FUNCTION(mesg)=__builtin_trap()'
//
// Run:
//   test_full_attention.exe             (full Whisper tiny shapes, T=1500)
//   test_full_attention.exe 32          (quick test with T=32)
//   test_full_attention.exe 32 bias     (bias-only mode: weight=0, bias=deterministic)

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/system/hip/execution_policy.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cfloat>

#define CHECK_HIP(cmd) do { \
    hipError_t e = (cmd); \
    if (e != hipSuccess) { \
        fprintf(stderr, "HIP error '%s' at %s:%d\n", \
                hipGetErrorString(e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

#define CHECK_HIPBLAS(cmd) do { \
    hipblasStatus_t s = (cmd); \
    if (s != HIPBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "hipBLAS error %d at %s:%d\n", \
                (int)s, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

typedef unsigned int index_t;

// ================================================================
// Replicated EXACTLY from CTranslate2 source
// ================================================================

// From src/ops/concat_split_slide_gpu.cu lines 39-63
template <typename T>
class inner_dim_offset_map {
private:
    const T _offset;
    const T _input_dim;
    const T _output_dim;
    const T _inner_size;
public:
    inner_dim_offset_map(const T offset,
                         const T input_dim,
                         const T output_dim,
                         const T inner_size)
        : _offset(offset)
        , _input_dim(input_dim)
        , _output_dim(output_dim)
        , _inner_size(inner_size) {
    }
    __device__
    T operator()(const T i) const {
        const T i0 = i / (_input_dim * _inner_size);
        const T i1 = (i / _inner_size) % _input_dim;
        const T i2 = i % _inner_size;
        return i0 * (_output_dim * _inner_size) + (i1 + _offset) * _inner_size + i2;
    }
};

// From src/cuda/primitives.cu lines 424-443
template <typename T>
__global__ void transpose_0213(const T* in,
                               const index_t cols,
                               const index_t stride1,
                               const index_t stride2,
                               T* out) {
    const index_t stride = stride1 * stride2;
    const index_t j = blockIdx.x;
    const index_t z = j / stride;
    const index_t y = (j % stride) / stride1;
    const index_t x = (j % stride) % stride1;
    const index_t j2 = z * stride + x * stride2 + y;

    const T* row_in = in + j2 * cols;
    T* row_out = out + j * cols;

    for (index_t i = threadIdx.x; i < cols; i += blockDim.x) {
        row_out[i] = row_in[i];
    }
}

// ================================================================
// GPU softmax kernel (one block per row, shared memory reduction)
// ================================================================

__global__ void softmax_kernel(float* data, int rows, int cols) {
    extern __shared__ float sdata[];

    int row = blockIdx.x;
    if (row >= rows) return;

    float* row_data = data + row * cols;
    int tid = threadIdx.x;

    float local_max = -FLT_MAX;
    for (int i = tid; i < cols; i += blockDim.x) {
        float v = row_data[i];
        if (v > local_max) local_max = v;
    }

    sdata[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] > sdata[tid])
                sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }
    float row_max = sdata[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float v = expf(row_data[i] - row_max);
        row_data[i] = v;
        local_sum += v;
    }

    sdata[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float row_sum = sdata[0];
    __syncthreads();

    float inv_sum = 1.0f / row_sum;
    for (int i = tid; i < cols; i += blockDim.x) {
        row_data[i] *= inv_sum;
    }
}

// ================================================================
// GPU bias-add kernel (add bias vector to each row)
// ================================================================

__global__ void bias_add_kernel(float* data, const float* bias, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    float* row_data = data + row * cols;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        row_data[i] += bias[i];
    }
}

// ================================================================
// GPU residual add kernel
// ================================================================

__global__ void residual_add_kernel(float* output, const float* input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        output[idx] += input[idx];
}

// ================================================================
// Utilities
// ================================================================

float correlation(const float* a, const float* b, int n) {
    double sa = 0, sb = 0, sab = 0, sa2 = 0, sb2 = 0;
    for (int i = 0; i < n; i++) {
        sa += a[i]; sb += b[i];
        sab += (double)a[i] * b[i];
        sa2 += (double)a[i] * a[i];
        sb2 += (double)b[i] * b[i];
    }
    double num = n * sab - sa * sb;
    double den = sqrt((n * sa2 - sa * sa) * (n * sb2 - sb * sb));
    return den > 0 ? (float)(num / den) : 0.0f;
}

float max_abs_diff(const float* a, const float* b, int n) {
    float mx = 0;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

int count_mismatches(const float* a, const float* b, int n, float tol) {
    int c = 0;
    for (int i = 0; i < n; i++)
        if (fabsf(a[i] - b[i]) > tol) c++;
    return c;
}

void print_first_diffs(const float* ref, const float* gpu, int n, int max_show = 10) {
    int shown = 0;
    for (int i = 0; i < n && shown < max_show; i++) {
        if (fabsf(ref[i] - gpu[i]) > 1e-4f) {
            printf("  [%d] ref=%.6f gpu=%.6f diff=%.6e\n",
                   i, ref[i], gpu[i], ref[i] - gpu[i]);
            shown++;
        }
    }
}

void report(const char* label, const float* ref, const float* gpu, int n, float tol) {
    float corr = correlation(ref, gpu, n);
    int mis = count_mismatches(ref, gpu, n, tol);
    float maxd = max_abs_diff(ref, gpu, n);
    printf("  %s: corr=%.6f  mismatches(tol=%.0e)=%d/%d  max_diff=%.6e\n",
           label, corr, tol, mis, n, maxd);
}

// ================================================================
// CPU reference implementations (double precision accumulation)
// ================================================================

void cpu_gemm_bias(const float* input, const float* weight, const float* bias,
                   float* output, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0;
            for (int k = 0; k < K; k++)
                sum += (double)input[i * K + k] * weight[j * K + k];
            output[i * N + j] = (float)sum + bias[j];
        }
    }
}

void cpu_transpose_0213(const float* in, float* out,
                        int d0, int d1, int d2, int d3) {
    for (int b = 0; b < d0; b++)
        for (int i1 = 0; i1 < d1; i1++)
            for (int i2 = 0; i2 < d2; i2++)
                for (int i3 = 0; i3 < d3; i3++) {
                    int src = ((b * d1 + i1) * d2 + i2) * d3 + i3;
                    int dst = ((b * d2 + i2) * d1 + i1) * d3 + i3;
                    out[dst] = in[src];
                }
}

void cpu_split_axis1(const float* in, float* Q, float* K, float* V,
                     int d0, int total_heads, int split_heads,
                     int time, int d_head) {
    int inner = time * d_head;
    for (int b = 0; b < d0; b++)
        for (int h = 0; h < split_heads; h++)
            for (int t = 0; t < time; t++)
                for (int d = 0; d < d_head; d++) {
                    int out_idx = ((b * split_heads + h) * time + t) * d_head + d;
                    int base = b * total_heads * inner;
                    Q[out_idx] = in[base + ((h + 0 * split_heads) * time + t) * d_head + d];
                    K[out_idx] = in[base + ((h + 1 * split_heads) * time + t) * d_head + d];
                    V[out_idx] = in[base + ((h + 2 * split_heads) * time + t) * d_head + d];
                }
}

void cpu_batched_qk(const float* Q, const float* K, float* scores,
                    int num_heads, int T, int d_head, float scale) {
    for (int h = 0; h < num_heads; h++) {
        const float* q_head = Q + h * T * d_head;
        const float* k_head = K + h * T * d_head;
        float* s_head = scores + h * T * T;
        for (int i = 0; i < T; i++) {
            for (int j = 0; j < T; j++) {
                double sum = 0;
                for (int d = 0; d < d_head; d++)
                    sum += (double)q_head[i * d_head + d] * k_head[j * d_head + d];
                s_head[i * T + j] = (float)(sum * scale);
            }
        }
    }
}

void cpu_softmax(float* data, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        float* row = data + r * cols;
        double mx = row[0];
        for (int j = 1; j < cols; j++)
            if (row[j] > mx) mx = row[j];
        double sum = 0;
        for (int j = 0; j < cols; j++) {
            row[j] = (float)exp((double)row[j] - mx);
            sum += row[j];
        }
        float inv = (float)(1.0 / sum);
        for (int j = 0; j < cols; j++)
            row[j] *= inv;
    }
}

void cpu_batched_attn_v(const float* attn, const float* V, float* context,
                        int num_heads, int T, int d_head) {
    for (int h = 0; h < num_heads; h++) {
        const float* a_head = attn + h * T * T;
        const float* v_head = V + h * T * d_head;
        float* c_head = context + h * T * d_head;
        for (int i = 0; i < T; i++) {
            for (int d = 0; d < d_head; d++) {
                double sum = 0;
                for (int k = 0; k < T; k++)
                    sum += (double)a_head[i * T + k] * v_head[k * d_head + d];
                c_head[i * d_head + d] = (float)sum;
            }
        }
    }
}

// ================================================================
// GPU split using thrust::gather (replicates CT2 Split::compute)
// ================================================================

void gpu_split_thrust(const float* d_in, float* d_Q, float* d_K, float* d_V,
                      int d0, int total_heads, int split_heads,
                      int time, int d_head,
                      hipStream_t stream) {
    int inner_size = time * d_head;
    int inner_bytes = inner_size * (int)sizeof(float);
    int inner_u4 = inner_bytes / (int)sizeof(uint4);

    int out_floats = d0 * split_heads * time * d_head;
    int out_bytes = out_floats * (int)sizeof(float);
    int out_u4 = out_bytes / (int)sizeof(uint4);

    float* outputs[3] = { d_Q, d_K, d_V };
    int offset = 0;

    for (int s = 0; s < 3; s++) {
        auto map_ids = thrust::make_transform_iterator(
            thrust::counting_iterator<index_t>(0),
            inner_dim_offset_map<index_t>(offset,
                                          split_heads,
                                          total_heads,
                                          inner_u4));
        thrust::gather(thrust::hip::par_nosync.on(stream),
                       map_ids, map_ids + out_u4,
                       reinterpret_cast<const uint4*>(d_in),
                       reinterpret_cast<uint4*>(outputs[s]));
        offset += split_heads;
    }
}

// ================================================================
// Main
// ================================================================

int main(int argc, char** argv) {
    const int BATCH = 1;
    const int D_MODEL = 384;
    const int NUM_HEADS = 6;
    const int D_HEAD = 64;
    const int FUSED = 3 * NUM_HEADS * D_HEAD;
    const int TOTAL_HEADS = 3 * NUM_HEADS;

    int T = 1500;
    bool bias_only = false;

    if (argc > 1) T = atoi(argv[1]);
    if (argc > 2 && strcmp(argv[2], "bias") == 0) bias_only = true;

    printf("=== Full Self-Attention Pipeline Test for gfx1010 ===\n");
    printf("Mode: %s\n", bias_only ? "BIAS-ONLY (weight=0, bias=deterministic)" : "NORMAL (deterministic weights+biases)");
    printf("Shapes: batch=%d  time=%d  d_model=%d  heads=%d  d_head=%d\n",
           BATCH, T, D_MODEL, NUM_HEADS, D_HEAD);
    printf("\nPipeline:\n");
    printf("  1. QKV GEMM:       [%d,%d] x [%d,%d]^T + bias -> [%d,%d]\n",
           T, D_MODEL, FUSED, D_MODEL, T, FUSED);
    printf("  2. split_heads:    [%d,%d,%d,%d] -> transpose_0213 -> [%d,%d,%d,%d]\n",
           BATCH, T, TOTAL_HEADS, D_HEAD,
           BATCH, TOTAL_HEADS, T, D_HEAD);
    printf("  3. Split axis=1:   [%d,%d,%d,%d] -> Q,K,V each [%d,%d,%d,%d]\n",
           BATCH, TOTAL_HEADS, T, D_HEAD,
           BATCH, NUM_HEADS, T, D_HEAD);
    printf("  4. Q@K^T:          [%d,%d,%d] x [%d,%d,%d]^T * 0.125 -> [%d,%d,%d]\n",
           NUM_HEADS, T, D_HEAD, NUM_HEADS, T, D_HEAD, NUM_HEADS, T, T);
    printf("  5. Softmax:        [%d,%d,%d] per-row\n", NUM_HEADS, T, T);
    printf("  6. attn@V:         [%d,%d,%d] x [%d,%d,%d] -> [%d,%d,%d]\n",
           NUM_HEADS, T, T, NUM_HEADS, T, D_HEAD, NUM_HEADS, T, D_HEAD);
    printf("  7. combine_heads:  [%d,%d,%d,%d] -> transpose_0213 -> [%d,%d,%d,%d] -> [%d,%d]\n",
           BATCH, NUM_HEADS, T, D_HEAD,
           BATCH, T, NUM_HEADS, D_HEAD,
           T, D_MODEL);
    printf("  8. Out projection: [%d,%d] x [%d,%d]^T + bias -> [%d,%d]\n",
           T, D_MODEL, D_MODEL, D_MODEL, T, D_MODEL);
    printf("  9. Residual add:   output += input\n");

    long long qkv_count = (long long)BATCH * NUM_HEADS * T * D_HEAD;
    long long input_count = (long long)T * D_MODEL;
    long long fused_count = (long long)T * FUSED;
    long long qkv_weight_count = (long long)FUSED * D_MODEL;
    long long scores_count = (long long)NUM_HEADS * T * T;
    long long context_count = qkv_count;
    long long out_weight_count = (long long)D_MODEL * D_MODEL;

    size_t total_host_mb = (input_count + qkv_weight_count + FUSED +
                            fused_count * 2 + qkv_count * 6 +
                            scores_count * 2 + context_count * 2 +
                            out_weight_count + D_MODEL +
                            input_count * 2) * sizeof(float) / (1024 * 1024);
    size_t total_gpu_mb = (input_count + qkv_weight_count + FUSED +
                           fused_count + fused_count +
                           qkv_count * 3 + scores_count +
                           context_count + context_count +
                           out_weight_count + D_MODEL +
                           input_count) * sizeof(float) / (1024 * 1024);
    printf("\nMemory estimate: host ~%zuMB  gpu ~%zuMB\n", total_host_mb, total_gpu_mb);

    // ================================================================
    // Allocate host memory
    // ================================================================
    float* h_input       = new float[input_count];
    float* h_qkv_weight  = new float[qkv_weight_count];
    float* h_qkv_bias    = new float[FUSED];
    float* h_out_weight  = new float[out_weight_count];
    float* h_out_bias    = new float[D_MODEL];

    float* h_fused_ref   = new float[fused_count];
    float* h_trans_ref   = new float[fused_count];
    float* h_Q_ref       = new float[qkv_count];
    float* h_K_ref       = new float[qkv_count];
    float* h_V_ref       = new float[qkv_count];
    float* h_scores_ref  = new float[scores_count];
    float* h_context_ref = new float[context_count];
    float* h_combined_ref= new float[input_count];
    float* h_final_ref   = new float[input_count];

    float* h_Q_gpu       = new float[qkv_count];
    float* h_K_gpu       = new float[qkv_count];
    float* h_V_gpu       = new float[qkv_count];
    float* h_scores_gpu  = new float[scores_count];
    float* h_context_gpu = new float[context_count];
    float* h_combined_gpu= new float[input_count];
    float* h_final_gpu   = new float[input_count];

    // ================================================================
    // Fill deterministic data
    // ================================================================
    printf("\nFilling deterministic data...\n");
    for (long long i = 0; i < input_count; i++)
        h_input[i] = sinf(i * 0.01f) * 0.1f;

    if (bias_only) {
        memset(h_qkv_weight, 0, qkv_weight_count * sizeof(float));
        memset(h_out_weight, 0, out_weight_count * sizeof(float));
        for (int i = 0; i < FUSED; i++)
            h_qkv_bias[i] = sinf(i * 0.03f) * 0.05f;
        for (int i = 0; i < D_MODEL; i++)
            h_out_bias[i] = cosf(i * 0.04f) * 0.05f;
    } else {
        for (long long i = 0; i < qkv_weight_count; i++)
            h_qkv_weight[i] = cosf(i * 0.007f) * 0.1f;
        for (int i = 0; i < FUSED; i++)
            h_qkv_bias[i] = sinf(i * 0.03f) * 0.05f;
        for (long long i = 0; i < out_weight_count; i++)
            h_out_weight[i] = sinf(i * 0.011f) * 0.1f;
        for (int i = 0; i < D_MODEL; i++)
            h_out_bias[i] = cosf(i * 0.04f) * 0.05f;
    }

    // ================================================================
    // CPU reference: full pipeline
    // ================================================================
    printf("CPU step 1: QKV GEMM + bias [%d x %d x %d]...", T, FUSED, D_MODEL);
    fflush(stdout);
    cpu_gemm_bias(h_input, h_qkv_weight, h_qkv_bias, h_fused_ref, T, FUSED, D_MODEL);
    printf(" done\n");

    printf("CPU step 2: split_heads transpose [%d,%d,%d,%d] -> [%d,%d,%d,%d]...",
           BATCH, T, TOTAL_HEADS, D_HEAD, BATCH, TOTAL_HEADS, T, D_HEAD);
    fflush(stdout);
    cpu_transpose_0213(h_fused_ref, h_trans_ref, BATCH, T, TOTAL_HEADS, D_HEAD);
    printf(" done\n");

    printf("CPU step 3: split axis=1...");
    fflush(stdout);
    cpu_split_axis1(h_trans_ref, h_Q_ref, h_K_ref, h_V_ref,
                    BATCH, TOTAL_HEADS, NUM_HEADS, T, D_HEAD);
    printf(" done\n");

    printf("CPU step 4: Q@K^T scaled [%d batches, %dx%dx%d]...",
           NUM_HEADS, T, T, D_HEAD);
    fflush(stdout);
    cpu_batched_qk(h_Q_ref, h_K_ref, h_scores_ref, NUM_HEADS, T, D_HEAD, 0.125f);
    printf(" done\n");

    printf("CPU step 5: softmax [%d rows, %d cols]...", NUM_HEADS * T, T);
    fflush(stdout);
    cpu_softmax(h_scores_ref, NUM_HEADS * T, T);
    printf(" done\n");

    printf("CPU step 6: attn@V [%d batches, %dx%dx%d]...",
           NUM_HEADS, T, D_HEAD, T);
    fflush(stdout);
    cpu_batched_attn_v(h_scores_ref, h_V_ref, h_context_ref, NUM_HEADS, T, D_HEAD);
    printf(" done\n");

    printf("CPU step 7: combine_heads transpose [%d,%d,%d,%d] -> [%d,%d,%d,%d]...",
           BATCH, NUM_HEADS, T, D_HEAD, BATCH, T, NUM_HEADS, D_HEAD);
    fflush(stdout);
    cpu_transpose_0213(h_context_ref, h_combined_ref, BATCH, NUM_HEADS, T, D_HEAD);
    printf(" done\n");

    printf("CPU step 8: output projection [%d x %d x %d]...", T, D_MODEL, D_MODEL);
    fflush(stdout);
    cpu_gemm_bias(h_combined_ref, h_out_weight, h_out_bias, h_final_ref, T, D_MODEL, D_MODEL);
    printf(" done\n");

    printf("CPU step 9: residual add...");
    fflush(stdout);
    for (long long i = 0; i < input_count; i++)
        h_final_ref[i] += h_input[i];
    printf(" done\n");

    // ================================================================
    // GPU setup
    // ================================================================
    hipDeviceProp_t props;
    hipGetDeviceProperties(&props, 0);
    printf("\nGPU: %s (arch=%s, warpSize=%d)\n", props.name, props.gcnArchName, props.warpSize);

    hipStream_t stream;
    CHECK_HIP(hipStreamCreate(&stream));

    hipblasHandle_t handle;
    CHECK_HIPBLAS(hipblasCreate(&handle));
    CHECK_HIPBLAS(hipblasSetStream(handle, stream));

    float *d_input, *d_qkv_weight, *d_qkv_bias, *d_fused, *d_trans;
    float *d_Q, *d_K, *d_V, *d_scores, *d_context, *d_combined;
    float *d_out_weight, *d_out_bias, *d_final;

    CHECK_HIP(hipMalloc(&d_input,      input_count * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_qkv_weight, qkv_weight_count * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_qkv_bias,   FUSED * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_fused,      fused_count * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_trans,      fused_count * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_Q,          qkv_count * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_K,          qkv_count * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_V,          qkv_count * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_scores,     scores_count * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_context,    context_count * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_combined,   input_count * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_out_weight, out_weight_count * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_out_bias,   D_MODEL * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_final,      input_count * sizeof(float)));

    CHECK_HIP(hipMemcpyAsync(d_input,      h_input,      input_count * sizeof(float),      hipMemcpyHostToDevice, stream));
    CHECK_HIP(hipMemcpyAsync(d_qkv_weight, h_qkv_weight, qkv_weight_count * sizeof(float), hipMemcpyHostToDevice, stream));
    CHECK_HIP(hipMemcpyAsync(d_qkv_bias,   h_qkv_bias,   FUSED * sizeof(float),            hipMemcpyHostToDevice, stream));
    CHECK_HIP(hipMemcpyAsync(d_out_weight, h_out_weight, out_weight_count * sizeof(float),  hipMemcpyHostToDevice, stream));
    CHECK_HIP(hipMemcpyAsync(d_out_bias,   h_out_bias,   D_MODEL * sizeof(float),           hipMemcpyHostToDevice, stream));
    CHECK_HIP(hipStreamSynchronize(stream));

    // ================================================================
    // TEST 1: Full pipeline on same stream, NO intermediate syncs
    //         (replicates CT2 runtime behavior)
    // ================================================================
    printf("\n========================================================\n");
    printf("TEST 1: Full attention pipeline (same stream, no sync)\n");
    printf("========================================================\n");
    {
        // Step 1: QKV GEMM
        // CT2 pattern: cublasSgemm(handle, OP_T, OP_N, N, M, K, alpha, B, ldb, A, lda, beta, C, ldc)
        // For input[T,D_MODEL] x weight[FUSED,D_MODEL]^T -> fused[T,FUSED]:
        //   trans_a=false, trans_b=true => hipblas(OP_T, OP_N, FUSED, T, D_MODEL, ...)
        float alpha = 1.0f, beta = 0.0f;
        CHECK_HIPBLAS(hipblasSgemm(handle,
            HIPBLAS_OP_T, HIPBLAS_OP_N,
            FUSED, T, D_MODEL,
            &alpha,
            d_qkv_weight, D_MODEL,
            d_input, D_MODEL,
            &beta,
            d_fused, FUSED));

        // Step 1b: Add QKV bias
        {
            int threads = FUSED < 1024 ? FUSED : 1024;
            bias_add_kernel<<<T, threads, 0, stream>>>(d_fused, d_qkv_bias, T, FUSED);
        }

        // Step 2: split_heads transpose
        // [1, T, 18, 64] -> [1, 18, T, 64]
        {
            int rows = BATCH * T * TOTAL_HEADS;
            int cols_u4 = (D_HEAD * (int)sizeof(float)) / (int)sizeof(uint4);
            transpose_0213<<<rows, cols_u4, 0, stream>>>(
                reinterpret_cast<const uint4*>(d_fused),
                (index_t)cols_u4,
                (index_t)T,
                (index_t)TOTAL_HEADS,
                reinterpret_cast<uint4*>(d_trans));
        }

        // Step 3: Split axis=1 via thrust::gather
        gpu_split_thrust(d_trans, d_Q, d_K, d_V,
                         BATCH, TOTAL_HEADS, NUM_HEADS, T, D_HEAD,
                         stream);

        // Step 4: Q@K^T with scale
        // CT2 pattern for trans_a=false, trans_b=true, m=T, n=T, k=64, alpha=0.125:
        //   hipblasSgemmStridedBatched(handle, OP_T, OP_N, T, T, 64, 0.125,
        //     K, 64, T*64, Q, 64, T*64, 0, scores, T, T*T, 6)
        {
            float qk_alpha = 0.125f;
            float qk_beta = 0.0f;
            int stride_qk = T * D_HEAD;
            int stride_scores = T * T;
            CHECK_HIPBLAS(hipblasSgemmStridedBatched(handle,
                HIPBLAS_OP_T, HIPBLAS_OP_N,
                T, T, D_HEAD,
                &qk_alpha,
                d_K, D_HEAD, stride_qk,
                d_Q, D_HEAD, stride_qk,
                &qk_beta,
                d_scores, T, stride_scores,
                NUM_HEADS));
        }

        // Step 5: Softmax (per-row, in-place)
        {
            int softmax_rows = NUM_HEADS * T;
            int threads = T < 1024 ? T : 1024;
            // round threads up to next power of 2 for reduction
            int threads_p2 = 1;
            while (threads_p2 < threads) threads_p2 <<= 1;
            if (threads_p2 > 1024) threads_p2 = 1024;
            softmax_kernel<<<softmax_rows, threads_p2,
                            threads_p2 * sizeof(float), stream>>>(
                d_scores, softmax_rows, T);
        }

        // Step 6: attn@V
        // CT2 pattern for trans_a=false, trans_b=false, m=T, n=64, k=T, alpha=1.0:
        //   hipblasSgemmStridedBatched(handle, OP_N, OP_N, 64, T, T, 1.0,
        //     V, 64, T*64, attn, T, T*T, 0, context, 64, T*64, 6)
        {
            float av_alpha = 1.0f;
            float av_beta = 0.0f;
            CHECK_HIPBLAS(hipblasSgemmStridedBatched(handle,
                HIPBLAS_OP_N, HIPBLAS_OP_N,
                D_HEAD, T, T,
                &av_alpha,
                d_V, D_HEAD, T * D_HEAD,
                d_scores, T, T * T,
                &av_beta,
                d_context, D_HEAD, T * D_HEAD,
                NUM_HEADS));
        }

        // Step 7: combine_heads transpose
        // [1, 6, T, 64] -> [1, T, 6, 64]
        {
            int rows = BATCH * NUM_HEADS * T;
            int cols_u4 = (D_HEAD * (int)sizeof(float)) / (int)sizeof(uint4);
            transpose_0213<<<rows, cols_u4, 0, stream>>>(
                reinterpret_cast<const uint4*>(d_context),
                (index_t)cols_u4,
                (index_t)NUM_HEADS,
                (index_t)T,
                reinterpret_cast<uint4*>(d_combined));
        }

        // Step 8: Output projection GEMM
        // combined[T,384] x out_weight[384,384]^T + bias -> final[T,384]
        {
            float proj_alpha = 1.0f, proj_beta = 0.0f;
            CHECK_HIPBLAS(hipblasSgemm(handle,
                HIPBLAS_OP_T, HIPBLAS_OP_N,
                D_MODEL, T, D_MODEL,
                &proj_alpha,
                d_out_weight, D_MODEL,
                d_combined, D_MODEL,
                &proj_beta,
                d_final, D_MODEL));
        }

        // Step 8b: Add output bias
        {
            int threads = D_MODEL < 1024 ? D_MODEL : 1024;
            bias_add_kernel<<<T, threads, 0, stream>>>(d_final, d_out_bias, T, D_MODEL);
        }

        // Step 9: Residual add
        {
            int threads = 256;
            int blocks = ((int)input_count + threads - 1) / threads;
            residual_add_kernel<<<blocks, threads, 0, stream>>>(d_final, d_input, (int)input_count);
        }

        CHECK_HIP(hipStreamSynchronize(stream));

        // Read back intermediate results for diagnostics
        CHECK_HIP(hipMemcpy(h_Q_gpu, d_Q, qkv_count * sizeof(float), hipMemcpyDeviceToHost));
        CHECK_HIP(hipMemcpy(h_K_gpu, d_K, qkv_count * sizeof(float), hipMemcpyDeviceToHost));
        CHECK_HIP(hipMemcpy(h_V_gpu, d_V, qkv_count * sizeof(float), hipMemcpyDeviceToHost));
        CHECK_HIP(hipMemcpy(h_scores_gpu, d_scores, scores_count * sizeof(float), hipMemcpyDeviceToHost));
        CHECK_HIP(hipMemcpy(h_context_gpu, d_context, context_count * sizeof(float), hipMemcpyDeviceToHost));
        CHECK_HIP(hipMemcpy(h_combined_gpu, d_combined, input_count * sizeof(float), hipMemcpyDeviceToHost));
        CHECK_HIP(hipMemcpy(h_final_gpu, d_final, input_count * sizeof(float), hipMemcpyDeviceToHost));

        printf("\n  --- Intermediate comparisons (TEST 1) ---\n");
        report("Q (after split)", h_Q_ref, h_Q_gpu, (int)qkv_count, 1e-3f);
        report("K (after split)", h_K_ref, h_K_gpu, (int)qkv_count, 1e-3f);
        report("V (after split)", h_V_ref, h_V_gpu, (int)qkv_count, 1e-3f);
        report("Scores (Q@K^T)", h_scores_ref, h_scores_gpu, (int)scores_count, 1e-3f);
        report("Context (attn@V)", h_context_ref, h_context_gpu, (int)context_count, 1e-3f);
        report("Combined (after combine_heads)", h_combined_ref, h_combined_gpu, (int)input_count, 1e-3f);

        printf("\n  --- FINAL OUTPUT (after residual add) ---\n");
        report("Final output", h_final_ref, h_final_gpu, (int)input_count, 1e-3f);

        float final_corr = correlation(h_final_ref, h_final_gpu, (int)input_count);
        float final_maxd = max_abs_diff(h_final_ref, h_final_gpu, (int)input_count);
        int final_mis = count_mismatches(h_final_ref, h_final_gpu, (int)input_count, 1e-3f);

        bool pass = (final_corr > 0.999f);
        printf("\n  >> TEST 1: %s (corr=%.6f, max_diff=%.6e, mismatches=%d/%lld)\n",
               pass ? "PASS" : "FAIL", final_corr, final_maxd, final_mis, input_count);
        if (!pass) {
            printf("  First diffs (final):\n");
            print_first_diffs(h_final_ref, h_final_gpu, (int)input_count);
        }
    }

    // ================================================================
    // TEST 2: Full pipeline with hipDeviceSynchronize between each step
    //         (tests whether stream ordering is the problem)
    // ================================================================
    printf("\n========================================================\n");
    printf("TEST 2: Full attention pipeline (hipDeviceSynchronize between steps)\n");
    printf("========================================================\n");
    {
        CHECK_HIP(hipMemcpyAsync(d_input, h_input, input_count * sizeof(float),
                                 hipMemcpyHostToDevice, stream));
        CHECK_HIP(hipDeviceSynchronize());

        // Step 1: QKV GEMM
        float alpha = 1.0f, beta = 0.0f;
        CHECK_HIPBLAS(hipblasSgemm(handle,
            HIPBLAS_OP_T, HIPBLAS_OP_N,
            FUSED, T, D_MODEL,
            &alpha,
            d_qkv_weight, D_MODEL,
            d_input, D_MODEL,
            &beta,
            d_fused, FUSED));
        CHECK_HIP(hipDeviceSynchronize());

        // Step 1b: Add QKV bias
        {
            int threads = FUSED < 1024 ? FUSED : 1024;
            bias_add_kernel<<<T, threads, 0, stream>>>(d_fused, d_qkv_bias, T, FUSED);
        }
        CHECK_HIP(hipDeviceSynchronize());

        // Step 2: split_heads transpose
        {
            int rows = BATCH * T * TOTAL_HEADS;
            int cols_u4 = (D_HEAD * (int)sizeof(float)) / (int)sizeof(uint4);
            transpose_0213<<<rows, cols_u4, 0, stream>>>(
                reinterpret_cast<const uint4*>(d_fused),
                (index_t)cols_u4,
                (index_t)T,
                (index_t)TOTAL_HEADS,
                reinterpret_cast<uint4*>(d_trans));
        }
        CHECK_HIP(hipDeviceSynchronize());

        // Step 3: Split axis=1
        gpu_split_thrust(d_trans, d_Q, d_K, d_V,
                         BATCH, TOTAL_HEADS, NUM_HEADS, T, D_HEAD,
                         stream);
        CHECK_HIP(hipDeviceSynchronize());

        // Step 4: Q@K^T with scale
        {
            float qk_alpha = 0.125f;
            float qk_beta = 0.0f;
            CHECK_HIPBLAS(hipblasSgemmStridedBatched(handle,
                HIPBLAS_OP_T, HIPBLAS_OP_N,
                T, T, D_HEAD,
                &qk_alpha,
                d_K, D_HEAD, T * D_HEAD,
                d_Q, D_HEAD, T * D_HEAD,
                &qk_beta,
                d_scores, T, T * T,
                NUM_HEADS));
        }
        CHECK_HIP(hipDeviceSynchronize());

        // Step 5: Softmax
        {
            int softmax_rows = NUM_HEADS * T;
            int threads = T < 1024 ? T : 1024;
            int threads_p2 = 1;
            while (threads_p2 < threads) threads_p2 <<= 1;
            if (threads_p2 > 1024) threads_p2 = 1024;
            softmax_kernel<<<softmax_rows, threads_p2,
                            threads_p2 * sizeof(float), stream>>>(
                d_scores, softmax_rows, T);
        }
        CHECK_HIP(hipDeviceSynchronize());

        // Step 6: attn@V
        {
            float av_alpha = 1.0f;
            float av_beta = 0.0f;
            CHECK_HIPBLAS(hipblasSgemmStridedBatched(handle,
                HIPBLAS_OP_N, HIPBLAS_OP_N,
                D_HEAD, T, T,
                &av_alpha,
                d_V, D_HEAD, T * D_HEAD,
                d_scores, T, T * T,
                &av_beta,
                d_context, D_HEAD, T * D_HEAD,
                NUM_HEADS));
        }
        CHECK_HIP(hipDeviceSynchronize());

        // Step 7: combine_heads transpose
        {
            int rows = BATCH * NUM_HEADS * T;
            int cols_u4 = (D_HEAD * (int)sizeof(float)) / (int)sizeof(uint4);
            transpose_0213<<<rows, cols_u4, 0, stream>>>(
                reinterpret_cast<const uint4*>(d_context),
                (index_t)cols_u4,
                (index_t)NUM_HEADS,
                (index_t)T,
                reinterpret_cast<uint4*>(d_combined));
        }
        CHECK_HIP(hipDeviceSynchronize());

        // Step 8: Output projection
        {
            float proj_alpha = 1.0f, proj_beta = 0.0f;
            CHECK_HIPBLAS(hipblasSgemm(handle,
                HIPBLAS_OP_T, HIPBLAS_OP_N,
                D_MODEL, T, D_MODEL,
                &proj_alpha,
                d_out_weight, D_MODEL,
                d_combined, D_MODEL,
                &proj_beta,
                d_final, D_MODEL));
        }
        CHECK_HIP(hipDeviceSynchronize());

        // Step 8b: Add output bias
        {
            int threads = D_MODEL < 1024 ? D_MODEL : 1024;
            bias_add_kernel<<<T, threads, 0, stream>>>(d_final, d_out_bias, T, D_MODEL);
        }
        CHECK_HIP(hipDeviceSynchronize());

        // Step 9: Residual add
        {
            int threads = 256;
            int blocks = ((int)input_count + threads - 1) / threads;
            residual_add_kernel<<<blocks, threads, 0, stream>>>(d_final, d_input, (int)input_count);
        }
        CHECK_HIP(hipDeviceSynchronize());

        CHECK_HIP(hipMemcpy(h_final_gpu, d_final, input_count * sizeof(float), hipMemcpyDeviceToHost));

        printf("\n  --- FINAL OUTPUT (after residual add) ---\n");
        report("Final output", h_final_ref, h_final_gpu, (int)input_count, 1e-3f);

        float final_corr = correlation(h_final_ref, h_final_gpu, (int)input_count);
        float final_maxd = max_abs_diff(h_final_ref, h_final_gpu, (int)input_count);
        int final_mis = count_mismatches(h_final_ref, h_final_gpu, (int)input_count, 1e-3f);

        bool pass = (final_corr > 0.999f);
        printf("\n  >> TEST 2: %s (corr=%.6f, max_diff=%.6e, mismatches=%d/%lld)\n",
               pass ? "PASS" : "FAIL", final_corr, final_maxd, final_mis, input_count);
        if (!pass) {
            printf("  First diffs (final):\n");
            print_first_diffs(h_final_ref, h_final_gpu, (int)input_count);
        }
    }

    // ================================================================
    // TEST 3: Step-by-step verification (check each intermediate result)
    //         Most useful for debugging — shows exactly where divergence starts
    // ================================================================
    printf("\n========================================================\n");
    printf("TEST 3: Step-by-step verification (each intermediate checked)\n");
    printf("========================================================\n");
    {
        CHECK_HIP(hipMemcpyAsync(d_input, h_input, input_count * sizeof(float),
                                 hipMemcpyHostToDevice, stream));
        CHECK_HIP(hipStreamSynchronize(stream));

        // Step 1: QKV GEMM + bias
        printf("\n  Step 1: QKV GEMM + bias\n");
        {
            float alpha = 1.0f, beta = 0.0f;
            CHECK_HIPBLAS(hipblasSgemm(handle,
                HIPBLAS_OP_T, HIPBLAS_OP_N,
                FUSED, T, D_MODEL,
                &alpha,
                d_qkv_weight, D_MODEL,
                d_input, D_MODEL,
                &beta,
                d_fused, FUSED));
            int threads = FUSED < 1024 ? FUSED : 1024;
            bias_add_kernel<<<T, threads, 0, stream>>>(d_fused, d_qkv_bias, T, FUSED);
            CHECK_HIP(hipStreamSynchronize(stream));
        }
        {
            float* h_fused_gpu = new float[fused_count];
            CHECK_HIP(hipMemcpy(h_fused_gpu, d_fused, fused_count * sizeof(float), hipMemcpyDeviceToHost));
            report("QKV fused", h_fused_ref, h_fused_gpu, (int)fused_count, 1e-3f);
            float c = correlation(h_fused_ref, h_fused_gpu, (int)fused_count);
            printf("  >> Step 1: %s (corr=%.6f)\n", c > 0.999f ? "PASS" : "FAIL", c);
            if (c <= 0.999f) {
                printf("  First diffs:\n");
                print_first_diffs(h_fused_ref, h_fused_gpu, (int)fused_count);
            }
            delete[] h_fused_gpu;
        }

        // Step 2: split_heads transpose
        printf("\n  Step 2: split_heads transpose\n");
        {
            int rows = BATCH * T * TOTAL_HEADS;
            int cols_u4 = (D_HEAD * (int)sizeof(float)) / (int)sizeof(uint4);
            transpose_0213<<<rows, cols_u4, 0, stream>>>(
                reinterpret_cast<const uint4*>(d_fused),
                (index_t)cols_u4,
                (index_t)T,
                (index_t)TOTAL_HEADS,
                reinterpret_cast<uint4*>(d_trans));
            CHECK_HIP(hipStreamSynchronize(stream));
        }
        {
            float* h_trans_gpu = new float[fused_count];
            CHECK_HIP(hipMemcpy(h_trans_gpu, d_trans, fused_count * sizeof(float), hipMemcpyDeviceToHost));
            report("Transposed", h_trans_ref, h_trans_gpu, (int)fused_count, 1e-3f);
            float c = correlation(h_trans_ref, h_trans_gpu, (int)fused_count);
            printf("  >> Step 2: %s (corr=%.6f)\n", c > 0.999f ? "PASS" : "FAIL", c);
            if (c <= 0.999f) {
                printf("  First diffs:\n");
                print_first_diffs(h_trans_ref, h_trans_gpu, (int)fused_count);
            }
            delete[] h_trans_gpu;
        }

        // Step 3: Split
        printf("\n  Step 3: Split axis=1\n");
        gpu_split_thrust(d_trans, d_Q, d_K, d_V,
                         BATCH, TOTAL_HEADS, NUM_HEADS, T, D_HEAD,
                         stream);
        CHECK_HIP(hipStreamSynchronize(stream));
        CHECK_HIP(hipMemcpy(h_Q_gpu, d_Q, qkv_count * sizeof(float), hipMemcpyDeviceToHost));
        CHECK_HIP(hipMemcpy(h_K_gpu, d_K, qkv_count * sizeof(float), hipMemcpyDeviceToHost));
        CHECK_HIP(hipMemcpy(h_V_gpu, d_V, qkv_count * sizeof(float), hipMemcpyDeviceToHost));
        report("Q", h_Q_ref, h_Q_gpu, (int)qkv_count, 1e-3f);
        report("K", h_K_ref, h_K_gpu, (int)qkv_count, 1e-3f);
        report("V", h_V_ref, h_V_gpu, (int)qkv_count, 1e-3f);
        {
            float cq = correlation(h_Q_ref, h_Q_gpu, (int)qkv_count);
            float ck = correlation(h_K_ref, h_K_gpu, (int)qkv_count);
            float cv = correlation(h_V_ref, h_V_gpu, (int)qkv_count);
            bool pass = (cq > 0.999f && ck > 0.999f && cv > 0.999f);
            printf("  >> Step 3: %s (Q=%.6f K=%.6f V=%.6f)\n",
                   pass ? "PASS" : "FAIL", cq, ck, cv);
        }

        // Step 4: Q@K^T
        printf("\n  Step 4: Q@K^T scaled\n");
        {
            float qk_alpha = 0.125f;
            float qk_beta = 0.0f;
            CHECK_HIPBLAS(hipblasSgemmStridedBatched(handle,
                HIPBLAS_OP_T, HIPBLAS_OP_N,
                T, T, D_HEAD,
                &qk_alpha,
                d_K, D_HEAD, T * D_HEAD,
                d_Q, D_HEAD, T * D_HEAD,
                &qk_beta,
                d_scores, T, T * T,
                NUM_HEADS));
            CHECK_HIP(hipStreamSynchronize(stream));
        }
        CHECK_HIP(hipMemcpy(h_scores_gpu, d_scores, scores_count * sizeof(float), hipMemcpyDeviceToHost));
        report("Scores (pre-softmax)", h_scores_ref, h_scores_gpu, (int)scores_count, 1e-3f);
        {
            // Note: h_scores_ref has already been softmaxed by now in the CPU pipeline.
            // We need a pre-softmax CPU reference. Recompute it.
            float* h_scores_pre = new float[scores_count];
            cpu_batched_qk(h_Q_ref, h_K_ref, h_scores_pre, NUM_HEADS, T, D_HEAD, 0.125f);
            report("Scores (pre-softmax, vs fresh CPU)", h_scores_pre, h_scores_gpu, (int)scores_count, 1e-3f);
            float c = correlation(h_scores_pre, h_scores_gpu, (int)scores_count);
            printf("  >> Step 4: %s (corr=%.6f)\n", c > 0.999f ? "PASS" : "FAIL", c);
            if (c <= 0.999f) {
                printf("  First diffs:\n");
                print_first_diffs(h_scores_pre, h_scores_gpu, (int)scores_count);
            }
            delete[] h_scores_pre;
        }

        // Step 5: Softmax
        printf("\n  Step 5: Softmax\n");
        {
            int softmax_rows = NUM_HEADS * T;
            int threads = T < 1024 ? T : 1024;
            int threads_p2 = 1;
            while (threads_p2 < threads) threads_p2 <<= 1;
            if (threads_p2 > 1024) threads_p2 = 1024;
            softmax_kernel<<<softmax_rows, threads_p2,
                            threads_p2 * sizeof(float), stream>>>(
                d_scores, softmax_rows, T);
            CHECK_HIP(hipStreamSynchronize(stream));
        }
        CHECK_HIP(hipMemcpy(h_scores_gpu, d_scores, scores_count * sizeof(float), hipMemcpyDeviceToHost));
        report("Scores (post-softmax)", h_scores_ref, h_scores_gpu, (int)scores_count, 1e-3f);
        {
            float c = correlation(h_scores_ref, h_scores_gpu, (int)scores_count);
            printf("  >> Step 5: %s (corr=%.6f)\n", c > 0.999f ? "PASS" : "FAIL", c);
            if (c <= 0.999f) {
                printf("  First diffs:\n");
                print_first_diffs(h_scores_ref, h_scores_gpu, (int)scores_count);
            }
            // Sanity: check row sums
            float min_sum = 1e9f, max_sum = -1e9f;
            int bad_rows = 0;
            for (int r = 0; r < NUM_HEADS * T; r++) {
                float s = 0;
                for (int c2 = 0; c2 < T; c2++)
                    s += h_scores_gpu[r * T + c2];
                if (s < min_sum) min_sum = s;
                if (s > max_sum) max_sum = s;
                if (fabsf(s - 1.0f) > 1e-3f) bad_rows++;
            }
            printf("  Softmax row sums: min=%.6f max=%.6f bad_rows(tol=1e-3)=%d/%d\n",
                   min_sum, max_sum, bad_rows, NUM_HEADS * T);
        }

        // Step 6: attn@V
        printf("\n  Step 6: attn@V\n");
        {
            float av_alpha = 1.0f;
            float av_beta = 0.0f;
            CHECK_HIPBLAS(hipblasSgemmStridedBatched(handle,
                HIPBLAS_OP_N, HIPBLAS_OP_N,
                D_HEAD, T, T,
                &av_alpha,
                d_V, D_HEAD, T * D_HEAD,
                d_scores, T, T * T,
                &av_beta,
                d_context, D_HEAD, T * D_HEAD,
                NUM_HEADS));
            CHECK_HIP(hipStreamSynchronize(stream));
        }
        CHECK_HIP(hipMemcpy(h_context_gpu, d_context, context_count * sizeof(float), hipMemcpyDeviceToHost));
        report("Context (attn@V)", h_context_ref, h_context_gpu, (int)context_count, 1e-3f);
        {
            float c = correlation(h_context_ref, h_context_gpu, (int)context_count);
            printf("  >> Step 6: %s (corr=%.6f)\n", c > 0.999f ? "PASS" : "FAIL", c);
            if (c <= 0.999f) {
                printf("  First diffs:\n");
                print_first_diffs(h_context_ref, h_context_gpu, (int)context_count);
            }
        }

        // Step 7: combine_heads transpose
        printf("\n  Step 7: combine_heads transpose\n");
        {
            int rows = BATCH * NUM_HEADS * T;
            int cols_u4 = (D_HEAD * (int)sizeof(float)) / (int)sizeof(uint4);
            transpose_0213<<<rows, cols_u4, 0, stream>>>(
                reinterpret_cast<const uint4*>(d_context),
                (index_t)cols_u4,
                (index_t)NUM_HEADS,
                (index_t)T,
                reinterpret_cast<uint4*>(d_combined));
            CHECK_HIP(hipStreamSynchronize(stream));
        }
        CHECK_HIP(hipMemcpy(h_combined_gpu, d_combined, input_count * sizeof(float), hipMemcpyDeviceToHost));
        report("Combined", h_combined_ref, h_combined_gpu, (int)input_count, 1e-3f);
        {
            float c = correlation(h_combined_ref, h_combined_gpu, (int)input_count);
            printf("  >> Step 7: %s (corr=%.6f)\n", c > 0.999f ? "PASS" : "FAIL", c);
            if (c <= 0.999f) {
                printf("  First diffs:\n");
                print_first_diffs(h_combined_ref, h_combined_gpu, (int)input_count);
            }
        }

        // Step 8: Output projection + bias
        printf("\n  Step 8: Output projection + bias\n");
        {
            float proj_alpha = 1.0f, proj_beta = 0.0f;
            CHECK_HIPBLAS(hipblasSgemm(handle,
                HIPBLAS_OP_T, HIPBLAS_OP_N,
                D_MODEL, T, D_MODEL,
                &proj_alpha,
                d_out_weight, D_MODEL,
                d_combined, D_MODEL,
                &proj_beta,
                d_final, D_MODEL));
            int threads = D_MODEL < 1024 ? D_MODEL : 1024;
            bias_add_kernel<<<T, threads, 0, stream>>>(d_final, d_out_bias, T, D_MODEL);
            CHECK_HIP(hipStreamSynchronize(stream));
        }
        {
            float* h_proj_ref = new float[input_count];
            cpu_gemm_bias(h_combined_ref, h_out_weight, h_out_bias, h_proj_ref, T, D_MODEL, D_MODEL);
            float* h_proj_gpu = new float[input_count];
            CHECK_HIP(hipMemcpy(h_proj_gpu, d_final, input_count * sizeof(float), hipMemcpyDeviceToHost));
            report("Projection (pre-residual)", h_proj_ref, h_proj_gpu, (int)input_count, 1e-3f);
            float c = correlation(h_proj_ref, h_proj_gpu, (int)input_count);
            printf("  >> Step 8: %s (corr=%.6f)\n", c > 0.999f ? "PASS" : "FAIL", c);
            if (c <= 0.999f) {
                printf("  First diffs:\n");
                print_first_diffs(h_proj_ref, h_proj_gpu, (int)input_count);
            }
            delete[] h_proj_ref;
            delete[] h_proj_gpu;
        }

        // Step 9: Residual add
        printf("\n  Step 9: Residual add\n");
        {
            int threads = 256;
            int blocks = ((int)input_count + threads - 1) / threads;
            residual_add_kernel<<<blocks, threads, 0, stream>>>(d_final, d_input, (int)input_count);
            CHECK_HIP(hipStreamSynchronize(stream));
        }
        CHECK_HIP(hipMemcpy(h_final_gpu, d_final, input_count * sizeof(float), hipMemcpyDeviceToHost));
        report("Final (after residual)", h_final_ref, h_final_gpu, (int)input_count, 1e-3f);
        {
            float c = correlation(h_final_ref, h_final_gpu, (int)input_count);
            float maxd = max_abs_diff(h_final_ref, h_final_gpu, (int)input_count);
            printf("  >> Step 9: %s (corr=%.6f, max_diff=%.6e)\n",
                   c > 0.999f ? "PASS" : "FAIL", c, maxd);
            if (c <= 0.999f) {
                printf("  First diffs:\n");
                print_first_diffs(h_final_ref, h_final_gpu, (int)input_count);
            }
        }
    }

    // ================================================================
    // Summary
    // ================================================================
    printf("\n========================================================\n");
    printf("DIAGNOSTIC KEY\n");
    printf("========================================================\n");
    printf("TEST 1 = full pipeline, same stream, no sync (CT2 behavior)\n");
    printf("TEST 2 = full pipeline, hipDeviceSynchronize between every step\n");
    printf("TEST 3 = step-by-step with intermediate checks\n");
    printf("\n");
    printf("If TEST 1+2+3 all PASS -> standalone attention chain works;\n");
    printf("  bug is in CT2 build/runtime (memory allocator, compiled code, etc.)\n");
    printf("If TEST 1 FAIL, TEST 2 PASS -> stream ordering broken on gfx1010\n");
    printf("If TEST 3 shows a specific step FAIL -> that step has the bug\n");
    printf("If ONLY bias-only mode fails -> bug is input-dependent\n");

    // ================================================================
    // Cleanup
    // ================================================================
    hipFree(d_input); hipFree(d_qkv_weight); hipFree(d_qkv_bias);
    hipFree(d_fused); hipFree(d_trans);
    hipFree(d_Q); hipFree(d_K); hipFree(d_V);
    hipFree(d_scores); hipFree(d_context); hipFree(d_combined);
    hipFree(d_out_weight); hipFree(d_out_bias); hipFree(d_final);
    hipblasDestroy(handle);
    hipStreamDestroy(stream);

    delete[] h_input; delete[] h_qkv_weight; delete[] h_qkv_bias;
    delete[] h_out_weight; delete[] h_out_bias;
    delete[] h_fused_ref; delete[] h_trans_ref;
    delete[] h_Q_ref; delete[] h_K_ref; delete[] h_V_ref;
    delete[] h_scores_ref; delete[] h_context_ref;
    delete[] h_combined_ref; delete[] h_final_ref;
    delete[] h_Q_gpu; delete[] h_K_gpu; delete[] h_V_gpu;
    delete[] h_scores_gpu; delete[] h_context_gpu;
    delete[] h_combined_gpu; delete[] h_final_gpu;

    printf("\nDone.\n");
    return 0;
}
