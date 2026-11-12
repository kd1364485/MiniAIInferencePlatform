// M6: 行列積のタイル化 (shared memoryでglobal読みを削減)
// naive版: 全スレッドが global memory から重複して読む
// tiled版: ブロックが TILE×TILE を shared に畳み込んで再利用

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>

#define TILE 16     // ブロック=16x16スレッド = 16x16タイル

// 前回のnaive版(比較用に残す)
__global__ void matmul_naive(const float* A, const float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n || col >= n) return;
    float sum = 0;
    for (int k = 0; k < n; ++k)
        sum += A[row * n + k] * B[k * n + col];
    C[row * n + col] = sum;
}

// タイル化版
__global__ void matmul_tiled(const float* A, const float* B, float* C, int n) {
    // ブロック内共有メモリ(全スレッドから見える、globalより速い)
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    int tx  = threadIdx.x;
    int ty  = threadIdx.y;

    float sum = 0;
    // k方向をTILE幅で刻んでループ(タイルスキャン)
    for (int t = 0; t < n / TILE; ++t) {
        // このスレッドがタイル内で担当する1要素をglobal→sharedへコピー
        As[ty][tx] = A[row * n + (t * TILE + tx)];
        Bs[ty][tx] = B[(t * TILE + ty) * n + col];
        __syncthreads();        // 全スレッドが書き終わるまで待ち合わせ

        // sharedメモリから読んで部分内積(globalより速い)
        for (int k = 0; k < TILE; ++k)
            sum += As[ty][k] * Bs[k][tx];
        __syncthreads();        // 次のタイル読み込み前にもう一度待ち合わせ
    }
    C[row * n + col] = sum;
}

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
#ifndef N_SIZE
#define N_SIZE 512
#endif
    int n = N_SIZE;
    size_t bytes = n * n * sizeof(float);

    float *h_a = new float[n*n], *h_b = new float[n*n], *h_c = new float[n*n];
    float *h_ref = new float[n*n];
    for (int i = 0; i < n*n; ++i) {
        h_a[i] = (i % 13) * 0.1f;
        h_b[i] = (i % 7)  * 0.1f;
    }

    // CPU (正解データ作成兼ベースライン)
    auto t0 = std::chrono::steady_clock::now();
    matmul_cpu(h_a, h_b, h_ref, n);
    double cpu_ms = elapsed_ms(t0);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes); cudaMalloc(&d_b, bytes); cudaMalloc(&d_c, bytes);
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    dim3 threads(TILE, TILE);
    dim3 blocks(n / TILE, n / TILE);

    // ウォームアップ (JITを除外するため1回空打ち)
    matmul_naive<<<blocks, threads>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();

    // naive版の計測
    t0 = std::chrono::steady_clock::now();
    matmul_naive<<<blocks, threads>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
    double naive_ms = elapsed_ms(t0);

    // tiled版の計測
    t0 = std::chrono::steady_clock::now();
    matmul_tiled<<<blocks, threads>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
    double tiled_ms = elapsed_ms(t0);

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < n * n; ++i) {
        float rel = fabsf(h_c[i] - h_ref[i]) / (fabsf(h_ref[i]) + 1e-6f);
        if (rel > 1e-4f) ++errors;
    }

    printf("n = %d, TILE = %d\n", n, TILE);
    printf("CPU          : %8.2f ms\n", cpu_ms);
    printf("GPU naive    : %8.2f ms\n", naive_ms);
    printf("GPU tiled    : %8.2f ms\n", tiled_ms);
    printf("naive->tiled : x%.2f\n", naive_ms / tiled_ms);
    printf("errors=%d\n", errors);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    delete[] h_a; delete[] h_b; delete[] h_c; delete[] h_ref;
}
