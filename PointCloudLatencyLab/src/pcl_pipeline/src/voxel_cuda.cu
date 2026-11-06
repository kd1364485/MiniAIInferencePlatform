#include "pcl_pipeline/voxel_cuda.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>

namespace pcl_pipeline {

namespace {

void check(cudaError_t e, const char * what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

// CPU 版(filters.cpp の cell_key)と同じ鍵: 各軸 21 bit
__device__ inline uint64_t cell_key(int32_t ix, int32_t iy, int32_t iz) {
  constexpr int32_t kOffset = 1 << 20;
  constexpr uint64_t kMask = (1u << 21) - 1;
  return ((static_cast<uint64_t>(ix + kOffset) & kMask) << 42) |
         ((static_cast<uint64_t>(iy + kOffset) & kMask) << 21) |
         (static_cast<uint64_t>(iz + kOffset) & kMask);
}

// ボクセルごとの合計(足し合わせの単位)
struct Acc {
  float x, y, z, i;
  uint32_t n;
};

struct AccSum {
  __device__ Acc operator()(const Acc & a, const Acc & b) const {
    return {a.x + b.x, a.y + b.y, a.z + b.z, a.i + b.i, a.n + b.n};
  }
};

__global__ void make_keys(const Point * pts, int n, float inv_leaf, uint64_t * keys,
                          uint32_t * idx) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= n) return;
  const Point p = pts[k];
  keys[k] = cell_key(static_cast<int32_t>(floorf(p.x * inv_leaf)),
                     static_cast<int32_t>(floorf(p.y * inv_leaf)),
                     static_cast<int32_t>(floorf(p.z * inv_leaf)));
  idx[k] = static_cast<uint32_t>(k);
}

// 並べ替えた順に点を並べ直し、1 点ぶんの Acc にする
__global__ void gather(const Point * pts, const uint32_t * idx_sorted, int n, Acc * vals) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= n) return;
  const Point p = pts[idx_sorted[k]];
  vals[k] = {p.x, p.y, p.z, p.intensity, 1u};
}

__global__ void centroids(const Acc * sums, const int * num_runs, Point * out) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= *num_runs) return;
  const Acc a = sums[k];
  const float inv = 1.f / static_cast<float>(a.n);
  out[k] = {a.x * inv, a.y * inv, a.z * inv, a.i * inv};
}

}  // namespace

struct VoxelCuda::Impl {
  size_t capacity = 0;
  Point * pts = nullptr;
  uint64_t * keys = nullptr;
  uint64_t * keys_sorted = nullptr;
  uint32_t * idx = nullptr;
  uint32_t * idx_sorted = nullptr;
  Acc * vals = nullptr;
  uint64_t * unique_keys = nullptr;
  Acc * sums = nullptr;
  Point * out = nullptr;
  int * num_runs = nullptr;
  void * temp = nullptr;
  size_t temp_bytes = 0;
  cudaStream_t stream = nullptr;
  cudaEvent_t ev[4] = {};

  Impl() {
    check(cudaStreamCreate(&stream), "cudaStreamCreate");
    for (auto & e : ev) check(cudaEventCreate(&e), "cudaEventCreate");
    check(cudaMalloc(&num_runs, sizeof(int)), "cudaMalloc");
  }
  ~Impl() {
    release();
    cudaFree(num_runs);
    for (auto & e : ev) cudaEventDestroy(e);
    cudaStreamDestroy(stream);
  }
  void release() {
    for (void * p : {static_cast<void *>(pts), static_cast<void *>(keys),
                     static_cast<void *>(keys_sorted), static_cast<void *>(idx),
                     static_cast<void *>(idx_sorted), static_cast<void *>(vals),
                     static_cast<void *>(unique_keys), static_cast<void *>(sums),
                     static_cast<void *>(out), temp}) {
      cudaFree(p);
    }
    pts = nullptr;
    keys = keys_sorted = unique_keys = nullptr;
    idx = idx_sorted = nullptr;
    vals = sums = nullptr;
    out = nullptr;
    temp = nullptr;
    capacity = temp_bytes = 0;
  }
  // 点の数が今までより多いときだけ確保し直す(1.5 倍の余裕を持たせる)
  void reserve(size_t n) {
    if (n <= capacity) return;
    release();
    capacity = n + n / 2;
    check(cudaMalloc(&pts, capacity * sizeof(Point)), "cudaMalloc pts");
    check(cudaMalloc(&keys, capacity * sizeof(uint64_t)), "cudaMalloc keys");
    check(cudaMalloc(&keys_sorted, capacity * sizeof(uint64_t)), "cudaMalloc keys_sorted");
    check(cudaMalloc(&idx, capacity * sizeof(uint32_t)), "cudaMalloc idx");
    check(cudaMalloc(&idx_sorted, capacity * sizeof(uint32_t)), "cudaMalloc idx_sorted");
    check(cudaMalloc(&vals, capacity * sizeof(Acc)), "cudaMalloc vals");
    check(cudaMalloc(&unique_keys, capacity * sizeof(uint64_t)), "cudaMalloc unique_keys");
    check(cudaMalloc(&sums, capacity * sizeof(Acc)), "cudaMalloc sums");
    check(cudaMalloc(&out, capacity * sizeof(Point)), "cudaMalloc out");
    // 作業領域: 並べ替えと足し合わせで要る大きさの、大きいほう
    size_t sort_bytes = 0, reduce_bytes = 0;
    const int cap = static_cast<int>(capacity);
    cub::DeviceRadixSort::SortPairs(nullptr, sort_bytes, keys, keys_sorted, idx, idx_sorted, cap,
                                    0, 63);
    cub::DeviceReduce::ReduceByKey(nullptr, reduce_bytes, keys_sorted, unique_keys, vals, sums,
                                   num_runs, AccSum(), cap);
    temp_bytes = std::max(sort_bytes, reduce_bytes);
    check(cudaMalloc(&temp, temp_bytes), "cudaMalloc temp");
  }
};

VoxelCuda::VoxelCuda() : impl_(std::make_unique<Impl>()) {}
VoxelCuda::~VoxelCuda() = default;

bool VoxelCuda::available() {
  int n = 0;
  return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

std::vector<Point> VoxelCuda::run(const std::vector<Point> & in, float leaf,
                                  VoxelCudaTiming * timing) {
  const int n = static_cast<int>(in.size());
  if (n == 0) return {};
  auto & m = *impl_;
  m.reserve(in.size());
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;

  cudaEventRecord(m.ev[0], m.stream);
  check(cudaMemcpyAsync(m.pts, in.data(), n * sizeof(Point), cudaMemcpyHostToDevice, m.stream),
        "H2D");
  cudaEventRecord(m.ev[1], m.stream);

  make_keys<<<blocks, threads, 0, m.stream>>>(m.pts, n, 1.f / leaf, m.keys, m.idx);
  size_t bytes = m.temp_bytes;
  // 鍵は 63 bit しか使わない(21 bit × 3)ので、そこまでだけ並べ替える
  check(cub::DeviceRadixSort::SortPairs(m.temp, bytes, m.keys, m.keys_sorted, m.idx,
                                        m.idx_sorted, n, 0, 63, m.stream),
        "SortPairs");
  gather<<<blocks, threads, 0, m.stream>>>(m.pts, m.idx_sorted, n, m.vals);
  bytes = m.temp_bytes;
  check(cub::DeviceReduce::ReduceByKey(m.temp, bytes, m.keys_sorted, m.unique_keys, m.vals,
                                       m.sums, m.num_runs, AccSum(), n, m.stream),
        "ReduceByKey");
  centroids<<<blocks, threads, 0, m.stream>>>(m.sums, m.num_runs, m.out);
  check(cudaGetLastError(), "kernel launch");
  cudaEventRecord(m.ev[2], m.stream);

  int runs = 0;
  check(cudaMemcpyAsync(&runs, m.num_runs, sizeof(int), cudaMemcpyDeviceToHost, m.stream),
        "D2H num_runs");
  check(cudaStreamSynchronize(m.stream), "sync");  // 結果の数がわからないと、戻す大きさが決まらない
  std::vector<Point> out(runs);
  check(cudaMemcpyAsync(out.data(), m.out, runs * sizeof(Point), cudaMemcpyDeviceToHost,
                        m.stream),
        "D2H out");
  cudaEventRecord(m.ev[3], m.stream);
  check(cudaStreamSynchronize(m.stream), "sync");

  if (timing) {
    float a = 0, b = 0, c = 0;
    cudaEventElapsedTime(&a, m.ev[0], m.ev[1]);
    cudaEventElapsedTime(&b, m.ev[1], m.ev[2]);
    cudaEventElapsedTime(&c, m.ev[2], m.ev[3]);
    *timing = {a, b, c, static_cast<double>(a) + b + c};
  }
  return out;
}

}  // namespace pcl_pipeline
