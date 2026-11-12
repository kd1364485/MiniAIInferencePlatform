// M3: ベクトル加算をCPUとGPUで速度比較
// 「GPUの方が速いか」は条件次第 — 計測してみる

#include <chrono>
#include <cstdio>
#include <cuda_runtime.h>

__global__ void vec_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

// CPU版: 普通のforループ
void vec_add_cpu(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; ++i) c[i] = a[i] + b[i];
}

double elapsed_ms(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
}

int main() {
    int n = 1 << 24;                    // 1677万要素 (前回の16倍)
    size_t bytes = n * sizeof(float);

    float *h_a = new float[n], *h_b = new float[n], *h_c = new float[n];
    for (int i = 0; i < n; ++i) { h_a[i] = i; h_b[i] = i * 2; }

    // ========== CPU版 ==========
    auto t0 = std::chrono::steady_clock::now();
    vec_add_cpu(h_a, h_b, h_c, n);
    double cpu_ms = elapsed_ms(t0);

    // ========== GPU版 ==========
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes); cudaMalloc(&d_b, bytes); cudaMalloc(&d_c, bytes);

    t0 = std::chrono::steady_clock::now();
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    double h2d_ms = elapsed_ms(t0);     // 転送(H→D)だけの時間

    t0 = std::chrono::steady_clock::now();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    vec_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
    double kernel_ms = elapsed_ms(t0);  // カーネル計算だけの時間

    t0 = std::chrono::steady_clock::now();
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    double d2h_ms = elapsed_ms(t0);     // 転送(D→H)だけの時間

    // 検証
    int errors = 0;
    for (int i = 0; i < n; ++i)
        if (h_c[i] != h_a[i] + h_b[i]) ++errors;

    printf("n = %d (%.1f MB x3)\n", n, bytes / 1e6);
    printf("CPU        : %8.2f ms\n", cpu_ms);
    printf("GPU kernel : %8.2f ms\n", kernel_ms);
    printf("GPU memcpy : %8.2f ms  (H2D %.2f + D2H %.2f)\n",
           h2d_ms + d2h_ms, h2d_ms, d2h_ms);
    printf("GPU total  : %8.2f ms\n", h2d_ms + kernel_ms + d2h_ms);
    printf("errors=%d\n", errors);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    delete[] h_a; delete[] h_b; delete[] h_c;
}
