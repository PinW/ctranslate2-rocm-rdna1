#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

using index_t = unsigned int;

// Exact replica of CTranslate2's transpose_0213 kernel
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

// CPU reference: transpose perm={0,2,1,3}
void cpu_transpose_0213(const float* in, float* out,
                        int d0, int d1, int d2, int d3) {
    for (int i0 = 0; i0 < d0; i0++)
        for (int i1 = 0; i1 < d1; i1++)
            for (int i2 = 0; i2 < d2; i2++)
                for (int i3 = 0; i3 < d3; i3++) {
                    int in_idx = ((i0 * d1 + i1) * d2 + i2) * d3 + i3;
                    int out_idx = ((i0 * d2 + i2) * d1 + i1) * d3 + i3;
                    out[out_idx] = in[in_idx];
                }
}

struct TestCase {
    const char* name;
    int d0, d1, d2, d3;
};

int run_test(const TestCase& tc) {
    int total = tc.d0 * tc.d1 * tc.d2 * tc.d3;
    size_t bytes = total * sizeof(float);

    float* h_in = (float*)malloc(bytes);
    float* h_out_gpu_uint4 = (float*)malloc(bytes);
    float* h_out_gpu_float = (float*)malloc(bytes);
    float* h_out_cpu = (float*)malloc(bytes);

    // Fill with known pattern: index-based values
    for (int i = 0; i < total; i++)
        h_in[i] = (float)(i % 10000) * 0.001f;

    cpu_transpose_0213(h_in, h_out_cpu, tc.d0, tc.d1, tc.d2, tc.d3);

    float *d_in, *d_out;
    hipMalloc(&d_in, bytes);
    hipMalloc(&d_out, bytes);
    hipMemcpy(d_in, h_in, bytes, hipMemcpyHostToDevice);

    int rows = tc.d0 * tc.d1 * tc.d2;

    // Test 1: uint4 path (matching CTranslate2 optimization)
    bool can_uint4 = (tc.d3 * sizeof(float)) % sizeof(uint4) == 0;
    if (can_uint4) {
        int cols_uint4 = (tc.d3 * sizeof(float)) / sizeof(uint4);
        int threads = cols_uint4 < 1024 ? cols_uint4 : 1024;

        hipMemset(d_out, 0, bytes);
        transpose_0213<<<rows, threads, 0, 0>>>(
            reinterpret_cast<const uint4*>(d_in),
            (index_t)cols_uint4,
            (index_t)tc.d1,
            (index_t)tc.d2,
            reinterpret_cast<uint4*>(d_out));
        hipDeviceSynchronize();
        hipMemcpy(h_out_gpu_uint4, d_out, bytes, hipMemcpyDeviceToHost);

        double sum_sq_diff = 0, max_diff = 0;
        int mismatches = 0;
        for (int i = 0; i < total; i++) {
            double diff = fabs(h_out_gpu_uint4[i] - h_out_cpu[i]);
            sum_sq_diff += diff * diff;
            if (diff > max_diff) max_diff = diff;
            if (diff > 1e-6) mismatches++;
        }
        double rmse = sqrt(sum_sq_diff / total);
        printf("  uint4 path: max_diff=%.6f, rmse=%.8f, mismatches=%d/%d %s\n",
               max_diff, rmse, mismatches, total, mismatches == 0 ? "[PASS]" : "[FAIL]");
        if (mismatches > 0 && mismatches < 20) {
            for (int i = 0; i < total && mismatches > 0; i++) {
                if (fabs(h_out_gpu_uint4[i] - h_out_cpu[i]) > 1e-6) {
                    printf("    idx=%d: gpu=%.6f cpu=%.6f\n", i, h_out_gpu_uint4[i], h_out_cpu[i]);
                    mismatches--;
                }
            }
        }
    }

    // Test 2: float path (non-vectorized)
    {
        int cols_float = tc.d3;
        int threads = cols_float < 1024 ? cols_float : 1024;

        hipMemset(d_out, 0, bytes);
        transpose_0213<<<rows, threads, 0, 0>>>(
            d_in,
            (index_t)cols_float,
            (index_t)tc.d1,
            (index_t)tc.d2,
            d_out);
        hipDeviceSynchronize();
        hipMemcpy(h_out_gpu_float, d_out, bytes, hipMemcpyDeviceToHost);

        double sum_sq_diff = 0, max_diff = 0;
        int mismatches = 0;
        for (int i = 0; i < total; i++) {
            double diff = fabs(h_out_gpu_float[i] - h_out_cpu[i]);
            sum_sq_diff += diff * diff;
            if (diff > max_diff) max_diff = diff;
            if (diff > 1e-6) mismatches++;
        }
        double rmse = sqrt(sum_sq_diff / total);
        printf("  float path: max_diff=%.6f, rmse=%.8f, mismatches=%d/%d %s\n",
               max_diff, rmse, mismatches, total, mismatches == 0 ? "[PASS]" : "[FAIL]");
    }

    // Compare uint4 vs float paths
    if (can_uint4) {
        int mismatches = 0;
        double max_diff = 0;
        for (int i = 0; i < total; i++) {
            double diff = fabs(h_out_gpu_uint4[i] - h_out_gpu_float[i]);
            if (diff > max_diff) max_diff = diff;
            if (diff > 1e-6) mismatches++;
        }
        printf("  uint4 vs float: max_diff=%.6f, mismatches=%d/%d %s\n",
               max_diff, mismatches, total, mismatches == 0 ? "[MATCH]" : "[MISMATCH]");
    }

    hipFree(d_in);
    hipFree(d_out);
    free(h_in);
    free(h_out_gpu_uint4);
    free(h_out_gpu_float);
    free(h_out_cpu);

    return 0;
}

int main() {
    printf("============================================================\n");
    printf("TEST: transpose_0213 kernel (exact CTranslate2 replica)\n");
    printf("============================================================\n");

    hipDeviceProp_t props;
    hipGetDeviceProperties(&props, 0);
    printf("GPU: %s (gfx%x, warpSize=%d)\n\n", props.name, props.gcnArchName[3] ? 0 : 0, props.warpSize);
    printf("GPU: %s\n\n", props.gcnArchName);

    TestCase tests[] = {
        // Exact Whisper tiny encoder split_heads shape
        {"split_heads [1, 1500, 18, 64]", 1, 1500, 18, 64},
        // Exact Whisper tiny encoder combine_heads shape
        {"combine_heads [1, 6, 1500, 64]", 1, 6, 1500, 64},
        // Small sanity check
        {"small [1, 4, 3, 8]", 1, 4, 3, 8},
        // Medium test
        {"medium [2, 100, 6, 64]", 2, 100, 6, 64},
        // Non-uint4 case (d3=3, not divisible by 4)
        {"non-uint4 [1, 4, 3, 3]", 1, 4, 3, 3},
        // Large batch
        {"large batch [4, 1500, 6, 64]", 4, 1500, 6, 64},
    };

    int num_tests = sizeof(tests) / sizeof(tests[0]);
    for (int t = 0; t < num_tests; t++) {
        printf("--- %s ---\n", tests[t].name);
        run_test(tests[t]);
        printf("\n");
    }

    printf("============================================================\n");
    printf("DONE\n");
    printf("============================================================\n");
    return 0;
}
