// 点群の処理そのもの(ROS に依存しない)。ノードはこれを呼ぶだけにして、
// M5 の CUDA 版との比較や M6 の gtest で、同じ関数をそのまま使えるようにする。
#pragma once

#include <cstdint>
#include <vector>

namespace pcl_pipeline {

// PointCloud2 の 1 点(x, y, z, intensity の float32 × 4 = 16 バイト)と同じ並び
struct Point {
  float x, y, z, intensity;
};
static_assert(sizeof(Point) == 16, "Point must match the PointCloud2 layout");

// 軸に平行な箱(min <= p <= max)
struct Box {
  float min_x, min_y, min_z, max_x, max_y, max_z;
  bool contains(const Point & p) const {
    return p.x >= min_x && p.x <= max_x && p.y >= min_y && p.y <= max_y && p.z >= min_z &&
           p.z <= max_z;
  }
};

// 回転式 LiDAR を真似る。センサーが原点、地面は z = -sensor_height の平面。
struct SimParams {
  int rings = 32;              // 縦の本数
  int azimuth_steps = 1800;    // 1 周の水平方向の数(0.2 度刻み)
  float min_elev_deg = -25.f;  // 一番下のビームの角度
  float max_elev_deg = 15.f;   // 一番上のビームの角度
  float sensor_height = 1.8f;  // 地面からの高さ [m]
  float max_range = 100.f;     // これより遠いと返ってこない [m]
  float noise_stddev = 0.02f;  // 距離のノイズ(標準偏差) [m]
};

// 地面と箱に光線を当て、当たった点を返す(返ってこない光線は点にならない)
std::vector<Point> simulate_scan(const SimParams & params, const std::vector<Box> & obstacles,
                                 uint32_t seed);

// keep の中にあり、ego(自車の範囲)の外にある点だけ残す
std::vector<Point> crop_box(const std::vector<Point> & in, const Box & keep, const Box & ego);

// 一辺 leaf の立方体ごとに、中の点を重心 1 点にまとめる
std::vector<Point> voxel_downsample(const std::vector<Point> & in, float leaf);

// 平面 a x + b y + c z + d = 0((a, b, c) は単位ベクトル)
struct Plane {
  float a = 0.f, b = 0.f, c = 1.f, d = 0.f;
};

// RANSAC で水平に近い平面(地面)を探し、平面から dist_thresh より離れた点だけ返す
std::vector<Point> remove_ground(const std::vector<Point> & in, float dist_thresh, int iterations,
                                 uint32_t seed, Plane * plane_out = nullptr);

struct Cluster {
  Box bounds;
  int num_points;
};

// 距離 tolerance 以内でつながる点を 1 つの塊にする。点の数が [min_points, max_points] の塊だけ返す
std::vector<Cluster> euclidean_cluster(const std::vector<Point> & in, float tolerance,
                                       int min_points, int max_points);

}  // namespace pcl_pipeline
