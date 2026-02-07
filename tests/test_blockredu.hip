#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#include <hipcub/block/block_reduce.hpp>
#include <stdio.h>
#include <math.h>

// Test 1: BlockReduce::Sum with 512 threads, 384 elements (matches layer norm config)
#define BLOCK_SIZE 512

__global__ void test_block_reduce_sum(const float* input, float* output, int N) {
    typedef hipcub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    float val = 0.0f;
    if (threadIdx.x < N) {
        val = input[blockIdx.x * N + threadIdx.x];
    }

    float sum = BlockReduce(temp_storage).Sum(val);

    if (threadIdx.x == 0) {
        output[blockIdx.x] = sum;
    }
}

// Test 2: Full layer norm reduction (mean + variance) - exactly as CTranslate2 does it
__global__ void test_layernorm_reduce(const float* X, float* means, float* vars, int N) {
    typedef hipcub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage m_temp;
    __shared__ typename BlockReduce::TempStorage v_temp;
    __shared__ float s_mean;
    __shared__ float s_var;

    int i = blockIdx.x;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
        float val = X[i * N + j];
        sum1 += val;
        sum2 += val * val;
    }

    sum1 = BlockReduce(m_temp).Sum(sum1);
    sum2 = BlockReduce(v_temp).Sum(sum2);

    if (threadIdx.x == 0) {
        float scale = 1.0f / (float)N;
        sum1 *= scale;
        sum2 = fmaxf(sum2 * scale - sum1 * sum1, 0.0f);
        s_mean = sum1;
        s_var = sum2;
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        means[i] = s_mean;
        vars[i] = s_var;
    }
}

// Test 3: Custom block_reduce from helpers.h (used by softmax)
#define C10_WARP_SIZE 32

template <typename Reduction, typename T>
__device__ __forceinline__ T custom_block_reduce(T* smem, T val, const Reduction& r, T defaultVal) {
    __syncthreads();
    smem[threadIdx.x] = val;
    __syncthreads();

    T warpVal = defaultVal;
    unsigned long long mask = (((unsigned long long)1) << (blockDim.x / C10_WARP_SIZE)) - 1;
    if (threadIdx.x < C10_WARP_SIZE) {
        unsigned int lane = threadIdx.x % C10_WARP_SIZE;
        if (lane < blockDim.x / C10_WARP_SIZE) {
            for (unsigned int i = 0; i < C10_WARP_SIZE; ++i) {
                warpVal = r(warpVal, smem[lane * C10_WARP_SIZE + i]);
            }
            __syncthreads();  // was __syncwarp(mask) in original
            smem[lane] = warpVal;
        }
    }

    __syncthreads();

    T blockVal = defaultVal;
    if (threadIdx.x == 0) {
        for (unsigned int i = 0; i < blockDim.x / C10_WARP_SIZE; ++i) {
            blockVal = r(blockVal, smem[i]);
        }
        smem[0] = blockVal;
    }

    __syncthreads();
    return smem[0];
}

struct AddOp {
    __device__ float operator()(float a, float b) const { return a + b; }
};

struct MaxOp {
    __device__ float operator()(float a, float b) const { return a > b ? a : b; }
};

__global__ void test_custom_reduce_sum(const float* input, float* output, int N) {
    extern __shared__ float smem[];
    float val = 0.0f;
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
        val += input[blockIdx.x * N + j];
    }
    float result = custom_block_reduce(smem, val, AddOp(), 0.0f);
    if (threadIdx.x == 0) output[blockIdx.x] = result;
}

__global__ void test_custom_reduce_max(const float* input, float* output, int N) {
    extern __shared__ float smem[];
    float val = -1e30f;
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
        float v = input[blockIdx.x * N + j];
        if (v > val) val = v;
    }
    float result = custom_block_reduce(smem, val, MaxOp(), -1e30f);
    if (threadIdx.x == 0) output[blockIdx.x] = result;
}

int main() {
    printf("=== Test hipcub::BlockReduce and custom block_reduce on gfx1010 ===\n\n");

    const int N = 384;  // whisper tiny hidden size
    const int num_rows = 8;

    // Create test data: row i has values [1, 2, 3, ..., N] * (i+1)
    float* h_input = new float[num_rows * N];
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < N; j++) {
            h_input[i * N + j] = (float)(j + 1) * (float)(i + 1);
        }
    }

    float* d_input;
    float* d_output;
    float* d_means;
    float* d_vars;
    hipMalloc(&d_input, num_rows * N * sizeof(float));
    hipMalloc(&d_output, num_rows * sizeof(float));
    hipMalloc(&d_means, num_rows * sizeof(float));
    hipMalloc(&d_vars, num_rows * sizeof(float));

    hipMemcpy(d_input, h_input, num_rows * N * sizeof(float), hipMemcpyHostToDevice);

    // Expected sum for row i: sum(1..N) * (i+1) = N*(N+1)/2 * (i+1)
    float expected_sum_base = (float)N * (float)(N + 1) / 2.0f;

    // Test 1: hipcub::BlockReduce::Sum
    printf("Test 1: hipcub::BlockReduce::Sum (512 threads, %d elements)\n", N);
    test_block_reduce_sum<<<num_rows, BLOCK_SIZE>>>(d_input, d_output, N);
    hipDeviceSynchronize();

    float h_output[8];
    hipMemcpy(h_output, d_output, num_rows * sizeof(float), hipMemcpyDeviceToHost);

    int pass1 = 1;
    for (int i = 0; i < num_rows; i++) {
        float expected = expected_sum_base * (float)(i + 1);
        float err = fabsf(h_output[i] - expected) / expected;
        const char* status = (err < 1e-5) ? "PASS" : "FAIL";
        if (err >= 1e-5) pass1 = 0;
        printf("  row %d: got=%.2f expected=%.2f rel_err=%.2e %s\n",
               i, h_output[i], expected, err, status);
    }
    printf("  %s\n\n", pass1 ? "ALL PASS" : "SOME FAILED");

    // Test 2: Layer norm mean/variance
    printf("Test 2: Layer norm reduction (mean + variance)\n");
    test_layernorm_reduce<<<num_rows, BLOCK_SIZE>>>(d_input, d_means, d_vars, N);
    hipDeviceSynchronize();

    float h_means[8], h_vars[8];
    hipMemcpy(h_means, d_means, num_rows * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(h_vars, d_vars, num_rows * sizeof(float), hipMemcpyDeviceToHost);

    int pass2 = 1;
    for (int i = 0; i < num_rows; i++) {
        float scale = (float)(i + 1);
        float exp_mean = expected_sum_base * scale / (float)N;
        // Variance of [scale, 2*scale, ..., N*scale] = scale^2 * Var(1..N)
        // Var(1..N) = (N^2-1)/12
        float exp_var = scale * scale * ((float)(N * N) - 1.0f) / 12.0f;
        float mean_err = fabsf(h_means[i] - exp_mean) / exp_mean;
        float var_err = fabsf(h_vars[i] - exp_var) / exp_var;
        const char* status = (mean_err < 1e-4 && var_err < 1e-3) ? "PASS" : "FAIL";
        if (mean_err >= 1e-4 || var_err >= 1e-3) pass2 = 0;
        printf("  row %d: mean=%.2f(exp=%.2f,err=%.2e) var=%.2f(exp=%.2f,err=%.2e) %s\n",
               i, h_means[i], exp_mean, mean_err, h_vars[i], exp_var, var_err, status);
    }
    printf("  %s\n\n", pass2 ? "ALL PASS" : "SOME FAILED");

    // Test 3: Custom block_reduce (softmax-style)
    printf("Test 3: Custom block_reduce Sum (softmax-style, %d threads)\n", BLOCK_SIZE);
    test_custom_reduce_sum<<<num_rows, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_input, d_output, N);
    hipDeviceSynchronize();

    hipMemcpy(h_output, d_output, num_rows * sizeof(float), hipMemcpyDeviceToHost);

    int pass3 = 1;
    for (int i = 0; i < num_rows; i++) {
        float expected = expected_sum_base * (float)(i + 1);
        float err = fabsf(h_output[i] - expected) / expected;
        const char* status = (err < 1e-5) ? "PASS" : "FAIL";
        if (err >= 1e-5) pass3 = 0;
        printf("  row %d: got=%.2f expected=%.2f rel_err=%.2e %s\n",
               i, h_output[i], expected, err, status);
    }
    printf("  %s\n\n", pass3 ? "ALL PASS" : "SOME FAILED");

    // Test 4: Custom block_reduce Max
    printf("Test 4: Custom block_reduce Max\n");
    test_custom_reduce_max<<<num_rows, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_input, d_output, N);
    hipDeviceSynchronize();

    hipMemcpy(h_output, d_output, num_rows * sizeof(float), hipMemcpyDeviceToHost);

    int pass4 = 1;
    for (int i = 0; i < num_rows; i++) {
        float expected = (float)N * (float)(i + 1);
        const char* status = (h_output[i] == expected) ? "PASS" : "FAIL";
        if (h_output[i] != expected) pass4 = 0;
        printf("  row %d: got=%.2f expected=%.2f %s\n",
               i, h_output[i], expected, status);
    }
    printf("  %s\n\n", pass4 ? "ALL PASS" : "SOME FAILED");

    printf("=== SUMMARY ===\n");
    printf("hipcub::BlockReduce Sum: %s\n", pass1 ? "PASS" : "FAIL");
    printf("Layer norm reduction:     %s\n", pass2 ? "PASS" : "FAIL");
    printf("Custom block_reduce Sum:  %s\n", pass3 ? "PASS" : "FAIL");
    printf("Custom block_reduce Max:  %s\n", pass4 ? "PASS" : "FAIL");

    hipFree(d_input);
    hipFree(d_output);
    hipFree(d_means);
    hipFree(d_vars);
    delete[] h_input;

    return (pass1 && pass2 && pass3 && pass4) ? 0 : 1;
}
