// M1: ベクトル加算 C[i] = A[i] + B[i] をGPUで並列計算する
// CUDAプログラムの3ステップ: (1)データをGPUへ送る (2)カーネル実行 (3)結果をCPUへ戻す

#include <cstdio>
#include <cuda_runtime.h>

// __global__ = 「GPUで実行される関数」。CPU側から<<<>>>で起動する
__global__ void vec_add(const float* a, const float* b, float* c, int n) {
    // このスレッドが担当する配列インデックスを計算
    // blockIdx.x  = 自分が何番目のブロックか
    // blockDim.x  = 1ブロックのスレッド数
    // threadIdx.x = ブロック内で自分が何番目か
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {            // はみ出し防止(スレッド数 > n のとき)
        c[i] = a[i] + b[i]; // 1スレッドが1要素だけ計算
    }
}

int main() {
    int n = 1 << 20;               // 約100万要素
    size_t bytes = n * sizeof(float);

    // --- CPU側メモリ ---
    float *h_a = new float[n], *h_b = new float[n], *h_c = new float[n];
    for (int i = 0; i < n; ++i) { h_a[i] = i; h_b[i] = i * 2; }

    // --- GPU側メモリ確保 (cudaMalloc = GPU版malloc) ---
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // --- (1) CPU→GPU コピー (cudaMemcpy H2D) ---
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // --- (2) カーネル起動: <<<ブロック数, 1ブロックのスレッド数>>> ---
    int threads = 256;                              // 1ブロック256スレッド
    int blocks = (n + threads - 1) / threads;       // 切り上げで必要ブロック数
    vec_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();                        // GPUの終了をCPUが待つ

    // --- (3) GPU→CPU コピー (cudaMemcpy D2H) ---
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // --- 検証: 全要素 c[i] == a[i] + b[i] か ---
    int errors = 0;
    for (int i = 0; i < n; ++i) {
        if (h_c[i] != h_a[i] + h_b[i]) ++errors;
    }
    printf("n=%d  blocks=%d  threads=%d  errors=%d\n", n, blocks, threads, errors);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    delete[] h_a; delete[] h_b; delete[] h_c;
}
