// M5: 行列積 C = A * B を CPU vs GPU で比較
// 演算強度が高い(O(N^3)の計算 / O(N^2)のデータ)＝GPUの本領
// 新要素: 2次元スレッド配置 (blockIdx.x/y, threadIdx.x/y)

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>

// GPU版: 1スレッドが C[i][j] 1要素を計算(内側でN回の乗算)
// スレッド座標 = (row, col) で出力行列の位置と一致
__global__ void matmul_gpu(const float* A, const float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // 行(i)
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // 列(j)
    if (row >= n || col >= n) return;

    float sum = 0;
    for (int k = 0; k < n; ++k) {
        sum += A[row * n + k] * B[k * n + col];       // 行×列の内積
    }
    C[row * n + col] = sum;
}

// CPU版: 教科書通りの3重ループ
void matmul_cpu(const float* A, const float* B, float* C, int n) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            float sum = 0;
            for (int k = 0; k < n; ++k)
                sum += A[i * n + k] * B[k * n + j];
            C[i * n + j] = sum;
        }
}

double elapsed_ms(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
}

int main() {
    int n = 512;                        // 512x512行列
    size_t bytes = n * n * sizeof(float);

    float *h_a = new float[n*n], *h_b = new float[n*n], *h_c = new float[n*n];
    float *h_ref = new float[n*n];
    // 適当な初期値(検証しやすい小さめの値)
    for (int i = 0; i < n*n; ++i) {
        h_a[i] = (i % 13) * 0.1f;
        h_b[i] = (i % 7)  * 0.1f;
    }

    // ========== CPU ==========
    auto t0 = std::chrono::steady_clock::now();
    matmul_cpu(h_a, h_b, h_ref, n);
    double cpu_ms = elapsed_ms(t0);

    // ========== GPU ==========
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes); cudaMalloc(&d_b, bytes); cudaMalloc(&d_c, bytes);
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // 2D配置: 1ブロック=16x16スレッド、グリッド=(32,32)ブロックで512x512をカバー
    dim3 threads(16, 16);               // 256スレッド/ブロック
    dim3 blocks((n + threads.x - 1) / threads.x,
                (n + threads.y - 1) / threads.y);

    t0 = std::chrono::steady_clock::now();
    matmul_gpu<<<blocks, threads>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
    double kernel_ms = elapsed_ms(t0);

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // 検証(浮動小数点なので相対誤差で判定)
    int errors = 0;
    for (int i = 0; i < n * n; ++i) {
        float diff = fabsf(h_c[i] - h_ref[i]);
        float rel  = diff / (fabsf(h_ref[i]) + 1e-6f);
        if (rel > 1e-4f) ++errors;
    }

    printf("n = %d (%dx%d)\n", n, n, n);
    printf("CPU        : %10.2f ms\n", cpu_ms);
    printf("GPU kernel : %10.2f ms\n", kernel_ms);
    printf("speedup    : x%.1f\n", cpu_ms / kernel_ms);
    printf("errors=%d\n", errors);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    delete[] h_a; delete[] h_b; delete[] h_c; delete[] h_ref;
}
