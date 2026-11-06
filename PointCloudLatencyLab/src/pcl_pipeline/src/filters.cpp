#include "pcl_pipeline/filters.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <unordered_map>

namespace pcl_pipeline {

namespace {

constexpr float kPi = 3.14159265358979f;

// 光線(原点から dir の向き)と箱の交点までの距離。当たらなければ +inf(スラブ法)
float ray_box(float dx, float dy, float dz, const Box & b) {
  float t_near = 0.f;
  float t_far = std::numeric_limits<float>::infinity();
  const float o[3] = {0.f, 0.f, 0.f};
  const float d[3] = {dx, dy, dz};
  const float lo[3] = {b.min_x, b.min_y, b.min_z};
  const float hi[3] = {b.max_x, b.max_y, b.max_z};
  for (int i = 0; i < 3; ++i) {
    if (std::fabs(d[i]) < 1e-9f) {
      if (o[i] < lo[i] || o[i] > hi[i]) return std::numeric_limits<float>::infinity();
      continue;
    }
    float t0 = (lo[i] - o[i]) / d[i];
    float t1 = (hi[i] - o[i]) / d[i];
    if (t0 > t1) std::swap(t0, t1);
    t_near = std::max(t_near, t0);
    t_far = std::min(t_far, t1);
    if (t_near > t_far) return std::numeric_limits<float>::infinity();
  }
  return t_near > 0.f ? t_near : std::numeric_limits<float>::infinity();
}

// 3 つの整数座標を 1 つの 64 bit の鍵にする(各 21 bit。±約 100 万セルまで)
inline uint64_t cell_key(int32_t ix, int32_t iy, int32_t iz) {
  constexpr int32_t kOffset = 1 << 20;
  constexpr uint64_t kMask = (1u << 21) - 1;
  return ((static_cast<uint64_t>(ix + kOffset) & kMask) << 42) |
         ((static_cast<uint64_t>(iy + kOffset) & kMask) << 21) |
         (static_cast<uint64_t>(iz + kOffset) & kMask);
}

inline int32_t cell_of(float v, float size) { return static_cast<int32_t>(std::floor(v / size)); }

}  // namespace

std::vector<Point> simulate_scan(const SimParams & p, const std::vector<Box> & obstacles,
                                 uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> noise(0.f, p.noise_stddev);
  std::vector<Point> out;
  out.reserve(static_cast<size_t>(p.rings) * p.azimuth_steps);

  for (int r = 0; r < p.rings; ++r) {
    const float elev_deg =
      p.rings == 1 ? p.min_elev_deg
                   : p.min_elev_deg + (p.max_elev_deg - p.min_elev_deg) * r / (p.rings - 1);
    const float elev = elev_deg * kPi / 180.f;
    const float ce = std::cos(elev), se = std::sin(elev);
    for (int a = 0; a < p.azimuth_steps; ++a) {
      const float az = 2.f * kPi * a / p.azimuth_steps;
      const float dx = ce * std::cos(az), dy = ce * std::sin(az), dz = se;

      float t = std::numeric_limits<float>::infinity();
      float intensity = 0.f;
      if (dz < 0.f) {  // 下向きの光線は地面に当たる
        t = -p.sensor_height / dz;
        intensity = 10.f;
      }
      for (const auto & b : obstacles) {
        const float tb = ray_box(dx, dy, dz, b);
        if (tb < t) {
          t = tb;
          intensity = 100.f;
        }
      }
      if (t > p.max_range) continue;  // 返ってこない
      const float range = t + noise(rng);
      out.push_back({dx * range, dy * range, dz * range, intensity});
    }
  }
  return out;
}

std::vector<Point> crop_box(const std::vector<Point> & in, const Box & keep, const Box & ego) {
  std::vector<Point> out;
  out.reserve(in.size());
  for (const auto & pt : in) {
    if (keep.contains(pt) && !ego.contains(pt)) out.push_back(pt);
  }
  return out;
}

std::vector<Point> voxel_downsample(const std::vector<Point> & in, float leaf) {
  struct Acc {
    double x = 0, y = 0, z = 0, i = 0;
    uint32_t n = 0;
  };
  std::unordered_map<uint64_t, Acc> cells;
  cells.reserve(in.size());
  for (const auto & pt : in) {
    auto & a = cells[cell_key(cell_of(pt.x, leaf), cell_of(pt.y, leaf), cell_of(pt.z, leaf))];
    a.x += pt.x;
    a.y += pt.y;
    a.z += pt.z;
    a.i += pt.intensity;
    ++a.n;
  }
  std::vector<Point> out;
  out.reserve(cells.size());
  for (const auto & kv : cells) {
    const auto & a = kv.second;
    out.push_back({static_cast<float>(a.x / a.n), static_cast<float>(a.y / a.n),
                   static_cast<float>(a.z / a.n), static_cast<float>(a.i / a.n)});
  }
  return out;
}

std::vector<Point> remove_ground(const std::vector<Point> & in, float dist_thresh, int iterations,
                                 uint32_t seed, Plane * plane_out) {
  if (in.size() < 3) return in;
  std::mt19937 rng(seed);
  std::uniform_int_distribution<size_t> pick(0, in.size() - 1);

  Plane best;
  size_t best_inliers = 0;
  for (int it = 0; it < iterations; ++it) {
    const Point & p0 = in[pick(rng)];
    const Point & p1 = in[pick(rng)];
    const Point & p2 = in[pick(rng)];
    // 法線 = (p1 - p0) × (p2 - p0)
    const float ux = p1.x - p0.x, uy = p1.y - p0.y, uz = p1.z - p0.z;
    const float vx = p2.x - p0.x, vy = p2.y - p0.y, vz = p2.z - p0.z;
    float a = uy * vz - uz * vy, b = uz * vx - ux * vz, c = ux * vy - uy * vx;
    const float len = std::sqrt(a * a + b * b + c * c);
    if (len < 1e-6f) continue;  // 3 点が一直線
    a /= len;
    b /= len;
    c /= len;
    if (std::fabs(c) < 0.9f) continue;  // 水平に近い平面だけを地面の候補にする(約 25 度まで)
    const float d = -(a * p0.x + b * p0.y + c * p0.z);
    size_t inliers = 0;
    for (const auto & q : in) {
      if (std::fabs(a * q.x + b * q.y + c * q.z + d) < dist_thresh) ++inliers;
    }
    if (inliers > best_inliers) {
      best_inliers = inliers;
      best = {a, b, c, d};
    }
  }
  if (plane_out) *plane_out = best;
  if (best_inliers == 0) return in;  // 地面が見つからない

  std::vector<Point> out;
  out.reserve(in.size() - best_inliers);
  for (const auto & q : in) {
    if (std::fabs(best.a * q.x + best.b * q.y + best.c * q.z + best.d) >= dist_thresh) {
      out.push_back(q);
    }
  }
  return out;
}

std::vector<Cluster> euclidean_cluster(const std::vector<Point> & in, float tolerance,
                                       int min_points, int max_points) {
  // 一辺 tolerance の格子に点を入れておくと、近くの点は周り 27 マスだけ探せばよい
  std::unordered_map<uint64_t, std::vector<uint32_t>> grid;
  grid.reserve(in.size());
  for (uint32_t i = 0; i < in.size(); ++i) {
    const auto & p = in[i];
    grid[cell_key(cell_of(p.x, tolerance), cell_of(p.y, tolerance), cell_of(p.z, tolerance))]
      .push_back(i);
  }

  const float tol2 = tolerance * tolerance;
  std::vector<uint8_t> visited(in.size(), 0);
  std::vector<uint32_t> queue;
  std::vector<Cluster> out;

  for (uint32_t seed = 0; seed < in.size(); ++seed) {
    if (visited[seed]) continue;
    visited[seed] = 1;
    queue.clear();
    queue.push_back(seed);
    for (size_t head = 0; head < queue.size(); ++head) {  // 幅優先探索
      const auto & p = in[queue[head]];
      const int32_t cx = cell_of(p.x, tolerance), cy = cell_of(p.y, tolerance),
                    cz = cell_of(p.z, tolerance);
      for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
          for (int dz = -1; dz <= 1; ++dz) {
            auto it = grid.find(cell_key(cx + dx, cy + dy, cz + dz));
            if (it == grid.end()) continue;
            for (uint32_t j : it->second) {
              if (visited[j]) continue;
              const auto & q = in[j];
              const float ex = q.x - p.x, ey = q.y - p.y, ez = q.z - p.z;
              if (ex * ex + ey * ey + ez * ez <= tol2) {
                visited[j] = 1;
                queue.push_back(j);
              }
            }
          }
        }
      }
    }
    const int n = static_cast<int>(queue.size());
    if (n < min_points || n > max_points) continue;
    Box b{std::numeric_limits<float>::max(),    std::numeric_limits<float>::max(),
          std::numeric_limits<float>::max(),    std::numeric_limits<float>::lowest(),
          std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest()};
    for (uint32_t k : queue) {
      const auto & q = in[k];
      b.min_x = std::min(b.min_x, q.x);
      b.min_y = std::min(b.min_y, q.y);
      b.min_z = std::min(b.min_z, q.z);
      b.max_x = std::max(b.max_x, q.x);
      b.max_y = std::max(b.max_y, q.y);
      b.max_z = std::max(b.max_z, q.z);
    }
    out.push_back({b, n});
  }
  return out;
}

}  // namespace pcl_pipeline
