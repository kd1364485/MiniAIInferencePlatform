// M5: voxel_downsample の CPU 版と CUDA 版を、点の数を変えて比べる(ROS は使わない)。
//   点群は sensor_sim と同じ疑似 LiDAR(地面 + 車 8 台)。rings(縦の本数)と azimuth(1 周の方向の数)で点の数を変える。
//   各大きさで、空打ち 5 回のあと 50 回測り、p50 を出す。結果が CPU 版と同じかも確かめる(点の順番は違うので並べ替えて比べる)。
// 使い方: voxel_bench [leaf=0.2] [iterations=50] [interval_ms=0]
//   interval_ms > 0 なら、1 回ごとにその時間だけ休む(パイプラインの 10 Hz を真似る。GPU が合間に休むかを見る)
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <map>
#include <tuple>
#include <vector>

#include "pcl_pipeline/filters.hpp"
#include "pcl_pipeline/voxel_cuda.hpp"

using namespace pcl_pipeline;

namespace {

double now_ms() {
  return std::chrono::duration<double, std::milli>(
           std::chrono::steady_clock::now().time_since_epoch())
    .count();
}

double p50(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

std::vector<Box> cars() {
  std::vector<Box> b;
  const float lanes[4] = {-7.f, -3.5f, 3.5f, 7.f};
  for (int i = 0; i < 8; ++i) {
    const float x = -30.f + 8.f * i, y = lanes[i % 4];
    b.push_back({x - 2.25f, y - 0.9f, -1.8f, x + 2.25f, y + 0.9f, -0.3f});
  }
  return b;
}

// 2 つの結果が同じか。重心は必ず自分のボクセルの中にあるので、重心の座標からボクセルの番号を求め、
// 同じボクセルどうしで座標の差を比べる。(座標で並べ替えて比べると、x がほぼ同じ 2 点の順番が
// CPU と GPU の丸めの違いで入れ替わり、そこから後ろが全部ずれて「違う」と出てしまった)
bool same(const std::vector<Point> & a, const std::vector<Point> & b, float leaf,
          double * max_diff) {
  auto key = [leaf](const Point & p) {
    return std::make_tuple(static_cast<int>(std::floor(p.x / leaf)),
                           static_cast<int>(std::floor(p.y / leaf)),
                           static_cast<int>(std::floor(p.z / leaf)));
  };
  if (a.size() != b.size()) return false;
  std::map<std::tuple<int, int, int>, Point> ref;
  for (const auto & p : a) ref[key(p)] = p;
  if (ref.size() != a.size()) return false;  // 同じボクセルが 2 つあるのはおかしい
  double m = 0;
  for (const auto & q : b) {
    auto it = ref.find(key(q));
    if (it == ref.end()) return false;
    const Point & p = it->second;
    m = std::max({m, std::fabs(double(p.x) - q.x), std::fabs(double(p.y) - q.y),
                  std::fabs(double(p.z) - q.z)});
  }
  *max_diff = m;
  return m < 1e-3;
}

}  // namespace

int main(int argc, char ** argv) {
  const float leaf = argc > 1 ? std::strtof(argv[1], nullptr) : 0.2f;
  const int iters = argc > 2 ? std::atoi(argv[2]) : 50;
  const int interval_ms = argc > 3 ? std::atoi(argv[3]) : 0;
  if (!VoxelCuda::available()) {
    std::fprintf(stderr, "CUDA device not found (WSL: LD_LIBRARY_PATH=/usr/lib/wsl/lib ?)\n");
    return 1;
  }
  VoxelCuda gpu;
  const struct {
    int rings, az;
  } sizes[] = {{4, 1800}, {8, 1800}, {16, 1800}, {32, 1800}, {64, 1800},
               {128, 1800}, {256, 1800}, {256, 3600}, {512, 3600}};

  std::printf("leaf=%.2f m, iterations=%d, interval=%d ms, p50 [ms]\n", leaf, iters,
              interval_ms);
  std::printf("%8s %8s | %8s | %8s %7s %8s %7s | %7s | %s\n", "points", "voxels", "cpu",
              "gpu_all", "h2d", "compute", "d2h", "speedup", "match (max diff)");
  for (const auto & s : sizes) {
    SimParams sp;
    sp.rings = s.rings;
    sp.azimuth_steps = s.az;
    const auto pts = simulate_scan(sp, cars(), 1);

    std::vector<double> cpu_ms, gpu_ms, h2d, comp, d2h;
    std::vector<Point> cpu_out, gpu_out;
    for (int it = 0; it < iters + 5; ++it) {
      if (interval_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
      double t0 = now_ms();
      cpu_out = voxel_downsample(pts, leaf);
      const double cpu_t = now_ms() - t0;
      if (interval_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
      VoxelCudaTiming t;
      const double tg = now_ms();  // GPU の時間は、休んだあとの呼び出しから測る
      gpu_out = gpu.run(pts, leaf, &t);
      const double gpu_t = now_ms() - tg;
      if (it < 5) continue;  // 空打ち(GPU の確保・初回の起動、キャッシュ)
      cpu_ms.push_back(cpu_t);
      gpu_ms.push_back(gpu_t);  // 呼び出しから戻るまで(CPU 側の時計)
      h2d.push_back(t.h2d_ms);
      comp.push_back(t.compute_ms);
      d2h.push_back(t.d2h_ms);
    }
    double diff = 0;
    const bool ok = same(cpu_out, gpu_out, leaf, &diff);
    std::printf("%8zu %8zu | %8.3f | %8.3f %7.3f %8.3f %7.3f | %6.2fx | %s (%.1e)\n", pts.size(),
                cpu_out.size(), p50(cpu_ms), p50(gpu_ms), p50(h2d), p50(comp), p50(d2h),
                p50(cpu_ms) / p50(gpu_ms), ok ? "OK" : "MISMATCH", diff);
  }
  return 0;
}
