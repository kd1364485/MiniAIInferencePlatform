// vecheavy_bench の検証失敗の切り分け用(元ソースは変更しない)
// 同じ入力・同じ式を CPU float / GPU float / CPU double で計算し、double を基準に誤差を比べる
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

__global__ void heavy_gpu(const float* a, const float* b, float* c, int n, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = a[i], y = b[i];
    for (int k = 0; k < K; ++k) x = sinf(x) * y + cosf(y);
    c[i] = x;
}

int main() {
    const int n = 1 << 20;
    std::vector<float> a(n), b(n), cf(n), g(n);
    for (int i = 0; i < n; ++i) { a[i] = i * 1e-4f; b[i] = i * 1e-5f + 1; }
    float *da, *db, *dc;
    cudaMalloc(&da, n * 4); cudaMalloc(&db, n * 4); cudaMalloc(&dc, n * 4);
    cudaMemcpy(da, a.data(), n * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b.data(), n * 4, cudaMemcpyHostToDevice);

    printf("n=%d  (y range %.1f .. %.1f)\n", n, b[0], b[n - 1]);
    printf("%4s | %-28s | %-28s | %-28s\n", "K", "CPUfloat vs double", "GPUfloat vs double", "CPUfloat vs GPUfloat");
    printf("%4s | %8s %8s %8s | %8s %8s %8s | %8s %8s %8s\n", "", "maxabs", "med", ">1e-4", "maxabs", "med", ">1e-4", "maxabs", "med", ">1e-4");
    int Ks[] = {1, 2, 3, 5, 10, 20, 200};
    for (int K : Ks) {
        std::vector<double> ref(n);
        for (int i = 0; i < n; ++i) {
            float x = a[i], y = b[i];
            for (int k = 0; k < K; ++k) x = sinf(x) * y + cosf(y);
            cf[i] = x;
            double xd = a[i], yd = b[i];
            for (int k = 0; k < K; ++k) xd = sin(xd) * yd + cos(yd);
            ref[i] = xd;
        }
        heavy_gpu<<<(n + 255) / 256, 256>>>(da, db, dc, n, K);
        cudaMemcpy(g.data(), dc, n * 4, cudaMemcpyDeviceToHost);
        auto stat = [&](auto f, double* mx, double* med, double* frac) {
            std::vector<double> e(n);
            long bad = 0;
            for (int i = 0; i < n; ++i) { e[i] = f(i); if (e[i] > 1e-4) ++bad; }
            *frac = 100.0 * bad / n;
            *mx = *std::max_element(e.begin(), e.end());
            std::nth_element(e.begin(), e.begin() + n / 2, e.end());
            *med = e[n / 2];
        };
        double m1, d1, f1, m2, d2, f2, m3, d3, f3;
        stat([&](int i) { return fabs(cf[i] - ref[i]); }, &m1, &d1, &f1);
        stat([&](int i) { return fabs(g[i] - ref[i]); }, &m2, &d2, &f2);
        stat([&](int i) { return fabs((double)cf[i] - g[i]); }, &m3, &d3, &f3);
        printf("%4d | %8.1e %8.1e %7.1f%% | %8.1e %8.1e %7.1f%% | %8.1e %8.1e %7.1f%%\n", K, m1, d1, f1, m2, d2, f2, m3, d3, f3);
    }

    // y の大きさ別(K=5, CPU float と GPU float の差)
    int K = 5;
    heavy_gpu<<<(n + 255) / 256, 256>>>(da, db, dc, n, K);
    cudaMemcpy(g.data(), dc, n * 4, cudaMemcpyDeviceToHost);
    printf("\nK=5: CPU float vs GPU float の差を y の範囲別に(中央値)\n");
    for (int lo = 0; lo < n; lo += n / 8) {
        std::vector<double> e;
        for (int i = lo; i < lo + n / 8; ++i) {
            float x = a[i], y = b[i];
            for (int k = 0; k < K; ++k) x = sinf(x) * y + cosf(y);
            e.push_back(fabs((double)x - g[i]));
        }
        std::nth_element(e.begin(), e.begin() + e.size() / 2, e.end());
        printf("  y=%6.1f..%6.1f  median diff = %.2e\n", b[lo], b[lo + n / 8 - 1], e[e.size() / 2]);
    }
}
