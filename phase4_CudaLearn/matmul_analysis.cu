// matmul_analysis.cu — 「タイル化(shared memory)が、naive の約 1.2 倍にとどまる」理由を測る
//
// 仮説:
//   (1) 16x16 のタイル化は、global メモリの読みを 16 分の 1 に減らすが、1 回の掛け算(FMA)ごとに
//       shared メモリから 2 回読む(As[ty][k] と Bs[k][tx])。shared メモリの読みの帯域が頭打ちになる。
//   (2) 元の naive 版も、B は連続して読まれ(coalesced)、A は同じ行を読む(ブロードキャスト)ので、
//       新しい GPU の L1 / L2 キャッシュに、かなり助けられている。差が小さいのは、そのため。
//   (3) 大きく速くするには、1 スレッドが複数の出力(4x4)を計算して、shared からの読みを減らす
//       (レジスタ・タイル化)。FMA 16 回に対して、読みは 8 回(0.5 回/FMA)になる。
//
// 比べるもの: naive / tiled16 / tiled32 / regtile(4x4) / cuBLAS(FP32 のみ)
// 時間は cudaEvent で測る(GPU 側の時間。launch の前後を chrono で挟むより正確)。
// 使い方: ./matmul_analysis [N=2048]    (N は 64 の倍数)
//
// ビルド: nvcc -O3 -arch=native matmul_analysis.cu -o matmul_analysis -lcublas

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CHECK(call)                                                                   \
    do {                                                                              \
        cudaError_t e = (call);                                                       \
        if (e != cudaSuccess) {                                                       \
            std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
            std::exit(1);                                                             \
        }                                                                             \
    } while (0)

// ---- 1. naive: 1 スレッドが出力 1 要素 ----
__global__ void k_naive(const float* A, const float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n || col >= n) return;
    float sum = 0;
    for (int k = 0; k < n; ++k) sum += A[row * n + k] * B[k * n + col];
    C[row * n + col] = sum;
}

// ---- 2. タイル化: T x T のタイルを shared に置いて再利用する ----
template <int T>
__global__ void k_tiled(const float* A, const float* B, float* C, int n) {
    __shared__ float As[T][T];
    __shared__ float Bs[T][T];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * T + ty, col = blockIdx.x * T + tx;
    float sum = 0;
    for (int t = 0; t < n / T; ++t) {
        As[ty][tx] = A[row * n + t * T + tx];
        Bs[ty][tx] = B[(t * T + ty) * n + col];
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < T; ++k) sum += As[ty][k] * Bs[k][tx];   // FMA 1 回に、shared の読みが 2 回
        __syncthreads();
    }
    C[row * n + col] = sum;
}

// ---- 3. レジスタ・タイル化: 64x64 のブロック、256 スレッド(16x16)、1 スレッドが 4x4 の出力 ----
//   FMA 16 回(4x4)に対して、shared の読みは 4(A)+4(B)=8 回。0.5 回/FMA(タイル化の 2 回/FMA の 4 分の 1)
constexpr int BM = 64, BN = 64, BK = 16;
__global__ void k_regtile(const float* A, const float* B, float* C, int n) {
    __shared__ float As[BM][BK + 1];   // +1: 異なる行を同時に読んだときの bank 衝突を避ける
    __shared__ __align__(16) float Bs[BK][BN];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * 16 + tx;
    const int by = blockIdx.y * BM, bx = blockIdx.x * BN;
    float acc[4][4] = {};

    for (int t = 0; t < n; t += BK) {
        // A のタイル(64x16 = 1024 個)と B のタイル(16x64 = 1024 個)を、256 スレッドで 4 個ずつ読む
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int idx = tid + i * 256;
            int ar = idx / BK, ac = idx % BK;
            As[ar][ac] = A[(by + ar) * n + t + ac];
            int br = idx / BN, bc = idx % BN;
            Bs[br][bc] = B[(t + br) * n + bx + bc];
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a[4], b[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) a[i] = As[ty * 4 + i][k];
            float4 bv = *reinterpret_cast<float4*>(&Bs[k][tx * 4]);
            b[0] = bv.x; b[1] = bv.y; b[2] = bv.z; b[3] = bv.w;
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float4 out = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
        *reinterpret_cast<float4*>(&C[(by + ty * 4 + i) * n + bx + tx * 4]) = out;
    }
}

// cudaEvent で、fn を iters 回呼んだ平均と最小(ms)を測る。ウォームアップ 3 回
struct Timing { double mean_ms, min_ms; };
static Timing measure(int iters, const std::function<void()>& fn) {
    for (int i = 0; i < 3; ++i) fn();
    CHECK(cudaDeviceSynchronize());
    cudaEvent_t ev_start, ev_stop;   // (CHECK マクロの内部変数 e と、名前をぶつけない)
    CHECK(cudaEventCreate(&ev_start));
    CHECK(cudaEventCreate(&ev_stop));
    double sum = 0, best = 1e30;
    for (int i = 0; i < iters; ++i) {
        CHECK(cudaEventRecord(ev_start));
        fn();
        CHECK(cudaEventRecord(ev_stop));
        CHECK(cudaEventSynchronize(ev_stop));
        float ms = 0;
        CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
        sum += ms;
        best = std::min<double>(best, ms);
    }
    CHECK(cudaEventDestroy(ev_start));
    CHECK(cudaEventDestroy(ev_stop));
    return {sum / iters, best};
}

int main(int argc, char** argv) {
    const int n = argc > 1 ? std::atoi(argv[1]) : 2048;
    if (n % 64 != 0) { std::printf("N は 64 の倍数にしてください\n"); return 1; }
    const size_t bytes = (size_t)n * n * sizeof(float);
    const double flops = 2.0 * n * n * n;

    cudaDeviceProp prop{};
    CHECK(cudaGetDeviceProperties(&prop, 0));
    int clock_khz = 0;
    CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0));
    // FP32 の理論ピーク = SM 数 x 128 FMA/clk(Ampere)x 2 FLOP x クロック
    const double peak_max = prop.multiProcessorCount * 128.0 * 2.0 * clock_khz * 1e3 / 1e12;   // TFLOPS(最大クロック)
    std::printf("GPU: %s, SM=%d, 最大クロック=%.0f MHz, FP32 理論ピーク(最大クロック)=%.1f TFLOPS\n", prop.name,
                prop.multiProcessorCount, clock_khz / 1000.0, peak_max);
    std::printf("N=%d  (FLOP = %.2f G)\n\n", n, flops / 1e9);

    std::vector<float> hA((size_t)n * n), hB((size_t)n * n);
    for (size_t i = 0; i < hA.size(); ++i) { hA[i] = (i % 13) * 0.1f; hB[i] = (i % 7) * 0.1f; }
    float *dA, *dB, *dC, *dRef;
    CHECK(cudaMalloc(&dA, bytes)); CHECK(cudaMalloc(&dB, bytes));
    CHECK(cudaMalloc(&dC, bytes)); CHECK(cudaMalloc(&dRef, bytes));
    CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));

    cublasHandle_t h;
    cublasCreate(&h);
    const float alpha = 1.0f, beta = 0.0f;
    // 行優先の C = A*B を、列優先の cuBLAS で計算する: C^T = B^T * A^T
    auto run_cublas = [&](float* out) {
        cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, dB, n, dA, n, &beta, out, n);
    };

    // 正解の基準: cuBLAS を、FP32 のまま(Tensor Core を使わない)で計算する
    cublasSetMathMode(h, CUBLAS_PEDANTIC_MATH);
    run_cublas(dRef);
    CHECK(cudaDeviceSynchronize());
    std::vector<float> ref((size_t)n * n), got((size_t)n * n);
    CHECK(cudaMemcpy(ref.data(), dRef, bytes, cudaMemcpyDeviceToHost));

    auto max_rel_err = [&](float* d) {
        CHECK(cudaMemcpy(got.data(), d, bytes, cudaMemcpyDeviceToHost));
        double worst = 0;
        for (size_t i = 0; i < got.size(); ++i)
            worst = std::max(worst, std::fabs((double)got[i] - ref[i]) / (std::fabs((double)ref[i]) + 1e-6));
        return worst;
    };

    const int iters = n <= 1024 ? 50 : (n <= 2048 ? 20 : 6);
    struct Row { const char* name; Timing t; double err; };
    std::vector<Row> rows;
    auto bench = [&](const char* name, const std::function<void()>& fn) {
        CHECK(cudaMemset(dC, 0, bytes));
        fn();
        CHECK(cudaDeviceSynchronize());
        double err = max_rel_err(dC);
        rows.push_back({name, measure(iters, fn), err});
    };

    dim3 t16(16, 16), g16(n / 16, n / 16);
    dim3 t32(32, 32), g32(n / 32, n / 32);
    dim3 tr(16, 16), gr(n / BN, n / BM);
    bench("naive (16x16 スレッド)",           [&] { k_naive<<<g16, t16>>>(dA, dB, dC, n); });
    bench("tiled16 (shared, 16x16)",          [&] { k_tiled<16><<<g16, t16>>>(dA, dB, dC, n); });
    bench("tiled32 (shared, 32x32)",          [&] { k_tiled<32><<<g32, t32>>>(dA, dB, dC, n); });
    bench("regtile (1 スレッド = 4x4 出力)",   [&] { k_regtile<<<gr, tr>>>(dA, dB, dC, n); });
    bench("cuBLAS Sgemm (FP32 のみ)",          [&] { run_cublas(dC); });

    cublasSetMathMode(h, CUBLAS_DEFAULT_MATH);
    CHECK(cudaMemset(dC, 0, bytes));
    run_cublas(dC);
    CHECK(cudaDeviceSynchronize());
    double err_default = max_rel_err(dC);
    Timing t_default = measure(iters, [&] { run_cublas(dC); });

    std::printf("%-30s %10s %10s %10s %9s %9s\n", "カーネル", "平均 ms", "最小 ms", "GFLOPS", "対naive", "最大相対誤差");
    const double naive_min = rows[0].t.min_ms;
    for (auto& r : rows) {
        std::printf("%-30s %10.3f %10.3f %10.0f %8.2fx %12.2e\n", r.name, r.t.mean_ms, r.t.min_ms,
                    flops / (r.t.min_ms * 1e-3) / 1e9, naive_min / r.t.min_ms, r.err);
    }
    std::printf("%-30s %10.3f %10.3f %10.0f %8.2fx %12.2e   ← 参考: 既定(TF32 が使われる可能性)\n",
                "cuBLAS Sgemm (既定の設定)", t_default.mean_ms, t_default.min_ms,
                flops / (t_default.min_ms * 1e-3) / 1e9, naive_min / t_default.min_ms, err_default);
    std::printf("\n(GFLOPS は最小時間から。理論ピークの何 %% かは、実行中の SM クロックで変わる。\n"
                " 最大クロックでのピーク %.0f GFLOPS に対して: naive %.1f%% / tiled16 %.1f%% / regtile %.1f%% / cuBLAS %.1f%%)\n",
                peak_max * 1000, 100.0 * flops / (rows[0].t.min_ms * 1e-3) / 1e9 / (peak_max * 1000),
                100.0 * flops / (rows[1].t.min_ms * 1e-3) / 1e9 / (peak_max * 1000),
                100.0 * flops / (rows[3].t.min_ms * 1e-3) / 1e9 / (peak_max * 1000),
                100.0 * flops / (rows[4].t.min_ms * 1e-3) / 1e9 / (peak_max * 1000));

    cublasDestroy(h);
    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dRef);
    return 0;
}
