// M6: 点群の処理(filters.cpp)の単体テスト。答えがわかる小さな入力で、各関数が正しいかを確かめる。
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "pcl_pipeline/filters.hpp"

using namespace pcl_pipeline;

namespace {

// 中心 (cx, cy, cz)、一辺 size の立方体の中に n 点をばらまく
std::vector<Point> blob(float cx, float cy, float cz, float size, int n, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> u(-size / 2, size / 2);
  std::vector<Point> v;
  for (int i = 0; i < n; ++i) v.push_back({cx + u(rng), cy + u(rng), cz + u(rng), 1.f});
  return v;
}

// z = -1.8 の平面(地面)に、x, y が [-r, r] の格子点を step 間隔で並べる
std::vector<Point> ground(float r, float step) {
  std::vector<Point> v;
  for (float x = -r; x <= r; x += step)
    for (float y = -r; y <= r; y += step) v.push_back({x, y, -1.8f, 10.f});
  return v;
}

}  // namespace

TEST(CropBox, KeepsInsideAndRemovesEgo) {
  const Box keep{-10, -10, -3, 10, 10, 3};
  const Box ego{-1, -1, -3, 1, 1, 3};
  const std::vector<Point> in = {
    {0, 0, 0, 0},     // 自車の中 → 消える
    {5, 0, 0, 0},     // 残る
    {20, 0, 0, 0},    // 範囲の外 → 消える
    {5, 5, 2.9f, 0},  // 残る(端の近く)
    {5, 5, 3.1f, 0},  // 高さが範囲の外 → 消える
  };
  const auto out = crop_box(in, keep, ego);
  ASSERT_EQ(out.size(), 2u);
  EXPECT_FLOAT_EQ(out[0].x, 5);
  EXPECT_FLOAT_EQ(out[1].z, 2.9f);
}

TEST(VoxelDownsample, OnePointPerVoxelAtCentroid) {
  // 同じボクセル(一辺 1 m の [0,1)^3)に 3 点、別のボクセルに 1 点
  const std::vector<Point> in = {
    {0.1f, 0.1f, 0.1f, 3}, {0.3f, 0.5f, 0.2f, 6}, {0.8f, 0.3f, 0.6f, 9}, {2.5f, 0.5f, 0.5f, 1}};
  auto out = voxel_downsample(in, 1.0f);
  ASSERT_EQ(out.size(), 2u);
  std::sort(out.begin(), out.end(), [](const Point & a, const Point & b) { return a.x < b.x; });
  EXPECT_NEAR(out[0].x, 0.4f, 1e-6);
  EXPECT_NEAR(out[0].y, 0.3f, 1e-6);
  EXPECT_NEAR(out[0].z, 0.3f, 1e-6);
  EXPECT_NEAR(out[0].intensity, 6.f, 1e-6);
  EXPECT_NEAR(out[1].x, 2.5f, 1e-6);
}

TEST(VoxelDownsample, NegativeCoordinatesUseFloor) {
  // -0.1 と 0.1 は、一辺 1 m なら別のボクセル(floor で -1 と 0)。切り捨て(0 と 0)だと誤って 1 つになる
  const std::vector<Point> in = {{-0.1f, 0, 0, 0}, {0.1f, 0, 0, 0}};
  EXPECT_EQ(voxel_downsample(in, 1.0f).size(), 2u);
}

TEST(RemoveGround, RemovesPlaneKeepsObstacle) {
  auto in = ground(10.f, 0.5f);
  const size_t n_ground = in.size();
  const auto car = blob(5, 0, -1.0f, 1.0f, 200, 1);  // 地面から 0.3〜1.3 m 上
  in.insert(in.end(), car.begin(), car.end());
  Plane plane;
  const auto out = remove_ground(in, 0.2f, 50, 7, &plane);
  EXPECT_EQ(out.size(), car.size());
  EXPECT_GT(n_ground, 1000u);
  EXPECT_NEAR(std::fabs(plane.c), 1.0f, 1e-3);   // 水平
  EXPECT_NEAR(-plane.d / plane.c, -1.8f, 1e-3);  // z = -1.8
}

TEST(EuclideanCluster, SeparatesTwoBlobsAndDropsSmallOnes) {
  auto in = blob(0, 0, 0, 1.0f, 300, 1);
  const auto b = blob(5, 0, 0, 1.0f, 300, 2);
  in.insert(in.end(), b.begin(), b.end());
  in.push_back({20, 20, 0, 0});  // 1 点だけ → min_points 未満で消える
  const auto clusters = euclidean_cluster(in, 0.5f, 5, 5000);
  ASSERT_EQ(clusters.size(), 2u);
  for (const auto & c : clusters) {
    EXPECT_EQ(c.num_points, 300);
    EXPECT_LT(c.bounds.max_x - c.bounds.min_x, 1.01f);
  }
}

TEST(EuclideanCluster, MaxPointsFiltersHugeCluster) {
  const auto in = blob(0, 0, 0, 1.0f, 300, 1);
  EXPECT_TRUE(euclidean_cluster(in, 0.5f, 5, 100).empty());
}

TEST(SimulateScan, GroundAndCarsAreHit) {
  SimParams p;
  p.rings = 16;
  p.azimuth_steps = 360;
  p.noise_stddev = 0.f;
  const std::vector<Box> cars = {{8, -1, -1.8f, 12, 1, -0.3f}};
  const auto pts = simulate_scan(p, cars, 1);
  ASSERT_FALSE(pts.empty());
  size_t on_ground = 0, on_car = 0;
  for (const auto & q : pts) {
    if (q.intensity == 10.f) {
      ++on_ground;
      EXPECT_NEAR(q.z, -1.8f, 1e-3);
    } else {
      ++on_car;
      EXPECT_GE(q.x, 8.f - 1e-3);  // 車の手前の面より手前には出ない
    }
  }
  EXPECT_GT(on_ground, 0u);
  EXPECT_GT(on_car, 0u);
}
