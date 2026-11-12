// 注意: この版は、K>=5 で CPU と GPU の結果の検証が通らない(GPU のバグではなく、式 sinf(x)*y+cosf(y) が
//       誤差を増幅するため)。過程の記録として残している。誤差が増えない形に直した版は vecheavy_stable.cu。
//       詳細は README の「注意・未解決」と diag_precision.cu。
//
// M4: 演算強度を上げた比較実験
// データ量はM3と同じ。1要素あたりの計算を100回の数学演算に増やす。
// → 転送時間は同じだが計算だけ重くなる = GPUの得意領域

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>

#ifndef K
#define K 200            // 1要素あたりの反復回数（演算強度）-DK=..で変更可
#endif

// GPU版: 各スレッドが1要素に対してK回の数学演算
__global__ void heavy_gpu(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = a[i];
    float y = b[i];
    for (int k = 0; k < K; ++k) {
        x = sinf(x) * y + cosf(y);   // sin/cos = GPUで1命令級の重い演算
    }
    c[i] = x;
}

// CPU版: 全く同じ計算をforループで
void heavy_cpu(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; ++i) {
        float x = a[i];
        float y = b[i];
        for (int k = 0; k < K; ++k) {
            x = sinf(x) * y + cosf(y);
        }
        c[i] = x;
    }
}

double elapsed_ms(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
}

int main() {
    int n = 1 << 24;
    size_t bytes = n * sizeof(float);

    float *h_a = new float[n], *h_b = new float[n], *h_c = new float[n];
    float *h_ref = new float[n];
    for (int i = 0; i < n; ++i) { h_a[i] = i * 1e-4f; h_b[i] = i * 1e-5f + 1; }

    // ========== CPU ==========
    auto t0 = std::chrono::steady_clock::now();
    heavy_cpu(h_a, h_b, h_ref, n);
    double cpu_ms = elapsed_ms(t0);

    // ========== GPU ==========
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes); cudaMalloc(&d_b, bytes); cudaMalloc(&d_c, bytes);

    t0 = std::chrono::steady_clock::now();
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    double h2d_ms = elapsed_ms(t0);

    t0 = std::chrono::steady_clock::now();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    heavy_gpu<<<blocks, threads>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
    double kernel_ms = elapsed_ms(t0);

    t0 = std::chrono::steady_clock::now();
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    double d2h_ms = elapsed_ms(t0);

    // 検証 (浮動小数点なので微小誤差許容)
    int errors = 0;
    for (int i = 0; i < n; ++i)
        if (fabsf(h_c[i] - h_ref[i]) > 1e-4f) ++errors;

    printf("n = %d, K = %d\n", n, K);
    printf("CPU        : %10.2f ms\n", cpu_ms);
    printf("GPU kernel : %10.2f ms\n", kernel_ms);
    printf("GPU memcpy : %10.2f ms  (H2D %.2f + D2H %.2f)\n",
           h2d_ms + d2h_ms, h2d_ms, d2h_ms);
    printf("GPU total  : %10.2f ms\n", h2d_ms + kernel_ms + d2h_ms);
    printf("speedup(kernel only): x%.1f\n", cpu_ms / kernel_ms);
    printf("speedup(total)      : x%.1f\n", cpu_ms / (h2d_ms + kernel_ms + d2h_ms));
    printf("errors=%d\n", errors);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    delete[] h_a; delete[] h_b; delete[] h_c; delete[] h_ref;
}
