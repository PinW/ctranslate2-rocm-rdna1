// test_chain_stream.hip — Self-attention op chain test for gfx1010
//
// Tests four hypotheses for WHY self-attention composition fails:
//   A) thrust::gather + inner_dim_offset_map + uint4 standalone
//   A2) Same as A but with thrust::hip::par instead of par_nosync
//   B) transpose_0213 -> split chain on same stream (no sync between)
//   C) GEMM -> transpose -> split chain on same stream (no sync between)
//   D) Same as C but with hipDeviceSynchronize() between each op
//
// Compile:
//   hipcc -O2 test_chain_stream.hip -o test_chain_stream.exe -lhipblas --offload-arch=gfx1010
//
// Run:
//   test_chain_stream.exe          (full Whisper tiny shapes, time=1500)
//   test_chain_stream.exe 32       (quick test with time=32)

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
// CPU reference implementations
// ================================================================

void cpu_gemm(const float* input, const float* weight, float* output,
              int M, int N, int K) {
    // output[M,N] = input[M,K] * weight[N,K]^T
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0;
            for (int k = 0; k < K; k++)
                sum += (double)input[i * K + k] * weight[j * K + k];
            output[i * N + j] = (float)sum;
        }
    }
}

void cpu_transpose_0213(const float* in, float* out,
                        int d0, int d1, int d2, int d3) {
    // [d0, d1, d2, d3] -> [d0, d2, d1, d3]
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
    // in[d0, total_heads, time, d_head] -> Q/K/V[d0, split_heads, time, d_head]
    int inner = time * d_head;
    for (int b = 0; b < d0; b++)
        for (int h = 0; h < split_heads; h++)
            for (int t = 0; t < time; t++)
                for (int d = 0; d < d_head; d++) {
                    int out_idx = ((b * split_heads + h) * time + t) * d_head + d;
                    int base = b * total_heads * inner;
                    Q[out_idx] = in[base + ((h + 0*split_heads) * time + t) * d_head + d];
                    K[out_idx] = in[base + ((h + 1*split_heads) * time + t) * d_head + d];
                    V[out_idx] = in[base + ((h + 2*split_heads) * time + t) * d_head + d];
                }
}

// ================================================================
// GPU split using thrust::gather (replicates CT2 Split::compute)
// ================================================================

void gpu_split_thrust(const float* d_in, float* d_Q, float* d_K, float* d_V,
                      int d0, int total_heads, int split_heads,
                      int time, int d_head,
                      hipStream_t stream, bool nosync) {
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
        if (nosync) {
            thrust::gather(thrust::hip::par_nosync.on(stream),
                           map_ids, map_ids + out_u4,
                           reinterpret_cast<const uint4*>(d_in),
                           reinterpret_cast<uint4*>(outputs[s]));
        } else {
            thrust::gather(thrust::hip::par.on(stream),
                           map_ids, map_ids + out_u4,
                           reinterpret_cast<const uint4*>(d_in),
                           reinterpret_cast<uint4*>(outputs[s]));
        }
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
    const int FUSED = 3 * NUM_HEADS * D_HEAD;    // 1152
    const int TOTAL_HEADS = 3 * NUM_HEADS;       // 18

    int T = 1500;
    if (argc > 1) T = atoi(argv[1]);

    printf("=== Self-Attention Chain Test for gfx1010 ===\n");
    printf("Shapes: batch=%d  time=%d  d_model=%d  heads=%d  d_head=%d\n",
           BATCH, T, D_MODEL, NUM_HEADS, D_HEAD);
    printf("GEMM: [%d,%d] x [%d,%d]^T -> [%d,%d]\n",
           T, D_MODEL, FUSED, D_MODEL, T, FUSED);
    printf("Transpose: [%d,%d,%d,%d] -> [%d,%d,%d,%d]\n",
           BATCH, T, TOTAL_HEADS, D_HEAD,
           BATCH, TOTAL_HEADS, T, D_HEAD);
    printf("Split axis=1: [%d,%d,%d,%d] -> 3x [%d,%d,%d,%d]\n",
           BATCH, TOTAL_HEADS, T, D_HEAD,
           BATCH, NUM_HEADS, T, D_HEAD);

    int dh_bytes = D_HEAD * (int)sizeof(float);
    int inner_bytes = T * D_HEAD * (int)sizeof(float);
    int out_bytes = BATCH * NUM_HEADS * T * D_HEAD * (int)sizeof(float);
    printf("\nuint4 alignment check:\n");
    printf("  d_head bytes=%d  %%16=%d  %s\n",
           dh_bytes, dh_bytes % 16, dh_bytes % 16 == 0 ? "OK" : "NOT ALIGNED");
    printf("  inner bytes=%d  %%16=%d  %s\n",
           inner_bytes, inner_bytes % 16, inner_bytes % 16 == 0 ? "OK" : "NOT ALIGNED");
    printf("  output bytes=%d  %%16=%d  %s\n",
           out_bytes, out_bytes % 16, out_bytes % 16 == 0 ? "OK" : "NOT ALIGNED");

    int gemm_count = T * FUSED;
    int qkv_count = BATCH * NUM_HEADS * T * D_HEAD;
    int input_count = T * D_MODEL;
    int weight_count = FUSED * D_MODEL;

    // Allocate host memory
    float* h_input     = new float[input_count];
    float* h_weight    = new float[weight_count];
    float* h_gemm_ref  = new float[gemm_count];
    float* h_trans_ref = new float[gemm_count];
    float* h_Q_ref     = new float[qkv_count];
    float* h_K_ref     = new float[qkv_count];
    float* h_V_ref     = new float[qkv_count];
    float* h_Q_gpu     = new float[qkv_count];
    float* h_K_gpu     = new float[qkv_count];
    float* h_V_gpu     = new float[qkv_count];

    // Deterministic data (small values to avoid overflow)
    printf("\nFilling input data...\n");
    for (int i = 0; i < input_count; i++)
        h_input[i] = sinf(i * 0.01f) * 0.1f;
    for (int i = 0; i < weight_count; i++)
        h_weight[i] = cosf(i * 0.007f) * 0.1f;

    // CPU reference
    printf("Computing CPU GEMM [%d x %d x %d]...", T, FUSED, D_MODEL);
    fflush(stdout);
    cpu_gemm(h_input, h_weight, h_gemm_ref, T, FUSED, D_MODEL);
    printf(" done\n");

    printf("Computing CPU transpose...");
    fflush(stdout);
    cpu_transpose_0213(h_gemm_ref, h_trans_ref, BATCH, T, TOTAL_HEADS, D_HEAD);
    printf(" done\n");

    printf("Computing CPU split...");
    fflush(stdout);
    cpu_split_axis1(h_trans_ref, h_Q_ref, h_K_ref, h_V_ref,
                    BATCH, TOTAL_HEADS, NUM_HEADS, T, D_HEAD);
    printf(" done\n");

    // GPU setup
    hipStream_t stream;
    CHECK_HIP(hipStreamCreate(&stream));

    hipblasHandle_t handle;
    CHECK_HIPBLAS(hipblasCreate(&handle));
    CHECK_HIPBLAS(hipblasSetStream(handle, stream));

    float *d_input, *d_weight, *d_gemm, *d_trans, *d_Q, *d_K, *d_V;
    CHECK_HIP(hipMalloc(&d_input,  input_count  * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_weight, weight_count * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_gemm,   gemm_count   * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_trans,  gemm_count   * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_Q,      qkv_count    * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_K,      qkv_count    * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_V,      qkv_count    * sizeof(float)));

    // ================================================================
    // TEST A: Standalone split (thrust::gather + uint4 + par_nosync)
    // ================================================================
    printf("\n=== TEST A: Standalone split (par_nosync) ===\n");
    printf("  Tests: thrust::gather + inner_dim_offset_map + uint4 reinterpret\n");
    {
        CHECK_HIP(hipMemcpyAsync(d_trans, h_trans_ref,
                                 gemm_count * sizeof(float),
                                 hipMemcpyHostToDevice, stream));
        CHECK_HIP(hipStreamSynchronize(stream));

        gpu_split_thrust(d_trans, d_Q, d_K, d_V,
                         BATCH, TOTAL_HEADS, NUM_HEADS, T, D_HEAD,
                         stream, true);

        CHECK_HIP(hipMemcpyAsync(h_Q_gpu, d_Q, qkv_count * sizeof(float),
                                 hipMemcpyDeviceToHost, stream));
        CHECK_HIP(hipMemcpyAsync(h_K_gpu, d_K, qkv_count * sizeof(float),
                                 hipMemcpyDeviceToHost, stream));
        CHECK_HIP(hipMemcpyAsync(h_V_gpu, d_V, qkv_count * sizeof(float),
                                 hipMemcpyDeviceToHost, stream));
        CHECK_HIP(hipStreamSynchronize(stream));

        report("Q", h_Q_ref, h_Q_gpu, qkv_count, 1e-5f);
        report("K", h_K_ref, h_K_gpu, qkv_count, 1e-5f);
        report("V", h_V_ref, h_V_gpu, qkv_count, 1e-5f);

        int mis = count_mismatches(h_Q_ref, h_Q_gpu, qkv_count, 1e-5f)
                + count_mismatches(h_K_ref, h_K_gpu, qkv_count, 1e-5f)
                + count_mismatches(h_V_ref, h_V_gpu, qkv_count, 1e-5f);
        printf("  >> %s\n", mis == 0 ? "PASS" : "FAIL");
        if (mis > 0) {
            printf("  First diffs (Q):\n");
            print_first_diffs(h_Q_ref, h_Q_gpu, qkv_count);
        }
    }

    // ================================================================
    // TEST A2: Same split but with thrust::hip::par (synchronizing)
    // ================================================================
    printf("\n=== TEST A2: Standalone split (par — synchronizing) ===\n");
    printf("  If A fails but A2 passes: par_nosync is the problem\n");
    {
        CHECK_HIP(hipMemcpyAsync(d_trans, h_trans_ref,
                                 gemm_count * sizeof(float),
                                 hipMemcpyHostToDevice, stream));
        CHECK_HIP(hipStreamSynchronize(stream));

        gpu_split_thrust(d_trans, d_Q, d_K, d_V,
                         BATCH, TOTAL_HEADS, NUM_HEADS, T, D_HEAD,
                         stream, false);

        CHECK_HIP(hipMemcpyAsync(h_Q_gpu, d_Q, qkv_count * sizeof(float),
                                 hipMemcpyDeviceToHost, stream));
        CHECK_HIP(hipMemcpyAsync(h_K_gpu, d_K, qkv_count * sizeof(float),
                                 hipMemcpyDeviceToHost, stream));
        CHECK_HIP(hipMemcpyAsync(h_V_gpu, d_V, qkv_count * sizeof(float),
                                 hipMemcpyDeviceToHost, stream));
        CHECK_HIP(hipStreamSynchronize(stream));

        report("Q", h_Q_ref, h_Q_gpu, qkv_count, 1e-5f);
        report("K", h_K_ref, h_K_gpu, qkv_count, 1e-5f);
        report("V", h_V_ref, h_V_gpu, qkv_count, 1e-5f);

        int mis = count_mismatches(h_Q_ref, h_Q_gpu, qkv_count, 1e-5f)
                + count_mismatches(h_K_ref, h_K_gpu, qkv_count, 1e-5f)
                + count_mismatches(h_V_ref, h_V_gpu, qkv_count, 1e-5f);
        printf("  >> %s\n", mis == 0 ? "PASS" : "FAIL");
    }

    // ================================================================
    // TEST B: transpose -> split chain (same stream, par_nosync)
    // ================================================================
    printf("\n=== TEST B: transpose -> split chain (same stream, no sync) ===\n");
    printf("  Tests: stream ordering between custom kernel and thrust\n");
    {
        CHECK_HIP(hipMemcpyAsync(d_gemm, h_gemm_ref,
                                 gemm_count * sizeof(float),
                                 hipMemcpyHostToDevice, stream));
        CHECK_HIP(hipStreamSynchronize(stream));

        // Transpose: [1, T, 18, 64] -> [1, 18, T, 64]
        int rows = BATCH * T * TOTAL_HEADS;
        int cols_u4 = (D_HEAD * (int)sizeof(float)) / (int)sizeof(uint4);
        int threads = cols_u4;
        transpose_0213<<<rows, threads, 0, stream>>>(
            reinterpret_cast<const uint4*>(d_gemm),
            (index_t)cols_u4,
            (index_t)T,             // stride1 = dims[1]
            (index_t)TOTAL_HEADS,   // stride2 = dims[2]
            reinterpret_cast<uint4*>(d_trans));

        // Split immediately after on same stream — NO sync
        gpu_split_thrust(d_trans, d_Q, d_K, d_V,
                         BATCH, TOTAL_HEADS, NUM_HEADS, T, D_HEAD,
                         stream, true);

        CHECK_HIP(hipMemcpyAsync(h_Q_gpu, d_Q, qkv_count * sizeof(float),
                                 hipMemcpyDeviceToHost, stream));
        CHECK_HIP(hipMemcpyAsync(h_K_gpu, d_K, qkv_count * sizeof(float),
                                 hipMemcpyDeviceToHost, stream));
        CHECK_HIP(hipMemcpyAsync(h_V_gpu, d_V, qkv_count * sizeof(float),
                                 hipMemcpyDeviceToHost, stream));
        CHECK_HIP(hipStreamSynchronize(stream));

        report("Q", h_Q_ref, h_Q_gpu, qkv_count, 1e-5f);
        report("K", h_K_ref, h_K_gpu, qkv_count, 1e-5f);
        report("V", h_V_ref, h_V_gpu, qkv_count, 1e-5f);

        int mis = count_mismatches(h_Q_ref, h_Q_gpu, qkv_count, 1e-5f)
                + count_mismatches(h_K_ref, h_K_gpu, qkv_count, 1e-5f)
                + count_mismatches(h_V_ref, h_V_gpu, qkv_count, 1e-5f);
        printf("  >> %s\n", mis == 0 ? "PASS" : "FAIL");
        if (mis > 0) {
            printf("  First diffs (Q):\n");
            print_first_diffs(h_Q_ref, h_Q_gpu, qkv_count);
        }
    }

    // ================================================================
    // TEST C: Full chain GEMM -> transpose -> split (same stream, no sync)
    // ================================================================
    printf("\n=== TEST C: GEMM -> transpose -> split (same stream, no sync) ===\n");
    printf("  Tests: full self-attention pre-processing chain\n");
    {
        CHECK_HIP(hipMemcpyAsync(d_input, h_input,
                                 input_count * sizeof(float),
                                 hipMemcpyHostToDevice, stream));
        CHECK_HIP(hipMemcpyAsync(d_weight, h_weight,
                                 weight_count * sizeof(float),
                                 hipMemcpyHostToDevice, stream));
        CHECK_HIP(hipStreamSynchronize(stream));

        // GEMM: output[T,FUSED] = input[T,D_MODEL] x weight[FUSED,D_MODEL]^T
        // CT2 swaps A/B for column-major: hipblasSgemm(OP_T, OP_N, N, M, K, ...)
        float alpha = 1.0f, beta = 0.0f;
        CHECK_HIPBLAS(hipblasSgemm(handle,
            HIPBLAS_OP_T, HIPBLAS_OP_N,
            FUSED, T, D_MODEL,
            &alpha,
            d_weight, D_MODEL,
            d_input, D_MODEL,
            &beta,
            d_gemm, FUSED));

        // Transpose on same stream — NO sync after GEMM
        int rows = BATCH * T * TOTAL_HEADS;
        int cols_u4 = (D_HEAD * (int)sizeof(float)) / (int)sizeof(uint4);
        transpose_0213<<<rows, cols_u4, 0, stream>>>(
            reinterpret_cast<const uint4*>(d_gemm),
            (index_t)cols_u4,
            (index_t)T,
            (index_t)TOTAL_HEADS,
            reinterpret_cast<uint4*>(d_trans));

        // Split on same stream — NO sync after transpose
        gpu_split_thrust(d_trans, d_Q, d_K, d_V,
                         BATCH, TOTAL_HEADS, NUM_HEADS, T, D_HEAD,
                         stream, true);

        CHECK_HIP(hipMemcpyAsync(h_Q_gpu, d_Q, qkv_count * sizeof(float),
                                 hipMemcpyDeviceToHost, stream));
        CHECK_HIP(hipMemcpyAsync(h_K_gpu, d_K, qkv_count * sizeof(float),
                                 hipMemcpyDeviceToHost, stream));
        CHECK_HIP(hipMemcpyAsync(h_V_gpu, d_V, qkv_count * sizeof(float),
                                 hipMemcpyDeviceToHost, stream));
        CHECK_HIP(hipStreamSynchronize(stream));

        // Use larger tolerance for GEMM (floating point accumulation differences)
        report("Q", h_Q_ref, h_Q_gpu, qkv_count, 1e-3f);
        report("K", h_K_ref, h_K_gpu, qkv_count, 1e-3f);
        report("V", h_V_ref, h_V_gpu, qkv_count, 1e-3f);

        float cq = correlation(h_Q_ref, h_Q_gpu, qkv_count);
        float ck = correlation(h_K_ref, h_K_gpu, qkv_count);
        float cv = correlation(h_V_ref, h_V_gpu, qkv_count);
        bool pass = (cq > 0.999f && ck > 0.999f && cv > 0.999f);
        printf("  >> %s (corr threshold > 0.999)\n", pass ? "PASS" : "FAIL");
        if (!pass) {
            printf("  First diffs (Q):\n");
            print_first_diffs(h_Q_ref, h_Q_gpu, qkv_count);
        }
    }

    // ================================================================
    // TEST D: Full chain with hipDeviceSynchronize between each op
    // ================================================================
    printf("\n=== TEST D: GEMM -> SYNC -> transpose -> SYNC -> split (explicit sync) ===\n");
    printf("  If C fails but D passes: stream ordering is broken\n");
    {
        CHECK_HIP(hipMemcpyAsync(d_input, h_input,
                                 input_count * sizeof(float),
                                 hipMemcpyHostToDevice, stream));
        CHECK_HIP(hipMemcpyAsync(d_weight, h_weight,
                                 weight_count * sizeof(float),
                                 hipMemcpyHostToDevice, stream));
        CHECK_HIP(hipDeviceSynchronize());

        float alpha = 1.0f, beta = 0.0f;
        CHECK_HIPBLAS(hipblasSgemm(handle,
            HIPBLAS_OP_T, HIPBLAS_OP_N,
            FUSED, T, D_MODEL,
            &alpha,
            d_weight, D_MODEL,
            d_input, D_MODEL,
            &beta,
            d_gemm, FUSED));
        CHECK_HIP(hipDeviceSynchronize());  // <<< SYNC after GEMM

        int rows = BATCH * T * TOTAL_HEADS;
        int cols_u4 = (D_HEAD * (int)sizeof(float)) / (int)sizeof(uint4);
        transpose_0213<<<rows, cols_u4, 0, stream>>>(
            reinterpret_cast<const uint4*>(d_gemm),
            (index_t)cols_u4,
            (index_t)T,
            (index_t)TOTAL_HEADS,
            reinterpret_cast<uint4*>(d_trans));
        CHECK_HIP(hipDeviceSynchronize());  // <<< SYNC after transpose

        gpu_split_thrust(d_trans, d_Q, d_K, d_V,
                         BATCH, TOTAL_HEADS, NUM_HEADS, T, D_HEAD,
                         stream, true);
        CHECK_HIP(hipDeviceSynchronize());  // <<< SYNC after split

        CHECK_HIP(hipMemcpy(h_Q_gpu, d_Q, qkv_count * sizeof(float),
                            hipMemcpyDeviceToHost));
        CHECK_HIP(hipMemcpy(h_K_gpu, d_K, qkv_count * sizeof(float),
                            hipMemcpyDeviceToHost));
        CHECK_HIP(hipMemcpy(h_V_gpu, d_V, qkv_count * sizeof(float),
                            hipMemcpyDeviceToHost));

        report("Q", h_Q_ref, h_Q_gpu, qkv_count, 1e-3f);
        report("K", h_K_ref, h_K_gpu, qkv_count, 1e-3f);
        report("V", h_V_ref, h_V_gpu, qkv_count, 1e-3f);

        float cq = correlation(h_Q_ref, h_Q_gpu, qkv_count);
        float ck = correlation(h_K_ref, h_K_gpu, qkv_count);
        float cv = correlation(h_V_ref, h_V_gpu, qkv_count);
        bool pass = (cq > 0.999f && ck > 0.999f && cv > 0.999f);
        printf("  >> %s (corr threshold > 0.999)\n", pass ? "PASS" : "FAIL");
        if (!pass) {
            printf("  First diffs (Q):\n");
            print_first_diffs(h_Q_ref, h_Q_gpu, qkv_count);
        }
    }

    // ================================================================
    // Summary
    // ================================================================
    printf("\n=== DIAGNOSTIC KEY ===\n");
    printf("A  fails            -> thrust::gather + uint4 broken on gfx1010\n");
    printf("A  fails, A2 passes -> par_nosync specifically is broken\n");
    printf("A  passes, B fails  -> stream ordering broken (kernel -> thrust)\n");
    printf("B  passes, C fails  -> stream ordering broken (hipBLAS -> kernel)\n");
    printf("C  fails, D passes  -> stream ordering broken (sync fixes it)\n");
    printf("All pass            -> bug is in CT2 build/allocator, not ops\n");

    // Cleanup
    hipFree(d_input); hipFree(d_weight); hipFree(d_gemm);
    hipFree(d_trans); hipFree(d_Q); hipFree(d_K); hipFree(d_V);
    hipblasDestroy(handle);
    hipStreamDestroy(stream);
    delete[] h_input;  delete[] h_weight;
    delete[] h_gemm_ref; delete[] h_trans_ref;
    delete[] h_Q_ref; delete[] h_K_ref; delete[] h_V_ref;
    delete[] h_Q_gpu; delete[] h_K_gpu; delete[] h_V_gpu;

    printf("\nDone.\n");
    return 0;
}
