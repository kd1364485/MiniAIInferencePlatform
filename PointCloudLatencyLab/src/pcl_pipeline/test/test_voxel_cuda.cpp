// M6: voxel の CUDA 版が CPU 版と同じ結果を返すか(GPU がない環境ではスキップ)
#include <gtest/gtest.h>

#include <cmath>
#include <map>
#include <tuple>
#include <vector>

#include "pcl_pipeline/filters.hpp"
#include "pcl_pipeline/voxel_cuda.hpp"

using namespace pcl_pipeline;

TEST(VoxelCuda, MatchesCpu) {
  if (!VoxelCuda::available()) GTEST_SKIP() << "no CUDA device";
  SimParams p;  // 32 本 × 1800 = 約 3.4 万点(パイプラインと同じ)
  const auto pts = simulate_scan(p, {{8, -1, -1.8f, 12, 1, -0.3f}}, 3);
  const float leaf = 0.2f;
  const auto cpu = voxel_downsample(pts, leaf);
  VoxelCuda gpu;
  for (int round = 0; round < 2; ++round) {  // 2 回目は確保済みのメモリを使い回す
    const auto out = gpu.run(pts, leaf);
    ASSERT_EQ(out.size(), cpu.size());
    // 重心は自分のボクセルの中にあるので、ボクセルの番号で対応を取る
    auto key = [leaf](const Point & q) {
      return std::make_tuple(int(std::floor(q.x / leaf)), int(std::floor(q.y / leaf)),
                             int(std::floor(q.z / leaf)));
    };
    std::map<std::tuple<int, int, int>, Point> ref;
    for (const auto & q : cpu) ref[key(q)] = q;
    for (const auto & q : out) {
      auto it = ref.find(key(q));
      ASSERT_NE(it, ref.end());
      EXPECT_NEAR(q.x, it->second.x, 1e-4);
      EXPECT_NEAR(q.y, it->second.y, 1e-4);
      EXPECT_NEAR(q.z, it->second.z, 1e-4);
    }
  }
}

TEST(VoxelCuda, EmptyInput) {
  if (!VoxelCuda::available()) GTEST_SKIP() << "no CUDA device";
  VoxelCuda gpu;
  EXPECT_TRUE(gpu.run({}, 0.2f).empty());
}
