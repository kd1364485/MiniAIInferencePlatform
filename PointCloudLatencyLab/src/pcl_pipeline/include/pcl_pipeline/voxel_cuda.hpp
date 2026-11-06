// M5: voxel_downsample の CUDA 版。CPU 版(filters.cpp)と同じ結果(点の順番は違う)を返す。
// やり方: 1 点ごとにボクセルの鍵を作る → 鍵で並べ替え(CUB の基数ソート)→ 同じ鍵が続く区間を足し合わせる
// (CUB の ReduceByKey)→ 点の数で割って重心にする。
// GPU のメモリと作業領域は、フレームごとに確保し直さないよう、オブジェクトが持ち続ける。
#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "pcl_pipeline/filters.hpp"

namespace pcl_pipeline {

// 1 回の実行の内訳 [ms](CUDA のイベントで測る)
struct VoxelCudaTiming {
  double h2d_ms = 0;      // 点群を GPU へ送る
  double compute_ms = 0;  // 鍵・並べ替え・足し合わせ・割り算
  double d2h_ms = 0;      // 結果の数と、結果を CPU へ戻す
  double total_ms = 0;    // 上の合計(呼び出しから戻るまで)
};

class VoxelCuda {
public:
  VoxelCuda();
  ~VoxelCuda();
  VoxelCuda(const VoxelCuda &) = delete;
  VoxelCuda & operator=(const VoxelCuda &) = delete;

  std::vector<Point> run(const std::vector<Point> & in, float leaf,
                         VoxelCudaTiming * timing = nullptr);

  // GPU が使えるか(ドライバが見つからない環境では false)
  static bool available();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace pcl_pipeline
