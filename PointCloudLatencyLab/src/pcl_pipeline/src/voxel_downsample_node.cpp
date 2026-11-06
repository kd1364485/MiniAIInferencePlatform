// 間引き: 一辺 leaf_size の立方体ごとに、中の点を重心 1 点にまとめる
#include <memory>
#include <vector>

#include "pcl_pipeline/filters.hpp"
#include "pcl_pipeline/node_util.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#ifdef PCL_PIPELINE_WITH_CUDA
#include "pcl_pipeline/voxel_cuda.hpp"
#endif

namespace pcl_pipeline {

class VoxelDownsample : public rclcpp::Node {
public:
  explicit VoxelDownsample(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : Node("voxel_downsample", options), stats_(this) {
    p_.leaf = static_cast<float>(declare_parameter("leaf_size", 0.2));
    // M5: use_cuda:=true なら CUDA 版を使う(CUDA なしでビルドしたときは CPU 版のまま)
    if (declare_parameter("use_cuda", false)) {
#ifdef PCL_PIPELINE_WITH_CUDA
      if (VoxelCuda::available()) {
        cuda_ = std::make_unique<VoxelCuda>();
      } else {
        RCLCPP_WARN(get_logger(), "use_cuda=true but no CUDA device; using CPU");
      }
#else
      RCLCPP_WARN(get_logger(), "use_cuda=true but built without CUDA; using CPU");
#endif
    }
    RCLCPP_INFO(get_logger(), "voxel on %s", use_gpu() ? "GPU (CUDA)" : "CPU");
    pub_ = create_publisher<PointCloud2>("downsampled/points", cloud_qos(this));
    sub_ = create_subscription<PointCloud2>(
      "cropped/points", cloud_qos(this), [this](PointCloud2::ConstSharedPtr msg) { on_cloud(*msg); });
  }

private:
  void on_cloud(const PointCloud2 & msg) {
    Stopwatch sw;
    const auto in = from_msg(msg);
    const auto out = run(in);
    auto out_msg = to_msg(out, msg.header);  // header(センサーの時刻)はそのまま引き継ぐ
    stats_.add(sw.ms(), msg.header.stamp, in.size(), out.size());
    pub_->publish(std::move(out_msg));
  }

  bool use_gpu() const {
#ifdef PCL_PIPELINE_WITH_CUDA
    return cuda_ != nullptr;
#else
    return false;
#endif
  }
  std::vector<Point> run(const std::vector<Point> & in) {
#ifdef PCL_PIPELINE_WITH_CUDA
    if (cuda_) return cuda_->run(in, p_.leaf);
#endif
    return voxel_downsample(in, p_.leaf);
  }

  struct Params {
    float leaf;
  } p_;
#ifdef PCL_PIPELINE_WITH_CUDA
  std::unique_ptr<VoxelCuda> cuda_;
#endif
  rclcpp::Publisher<PointCloud2>::SharedPtr pub_;
  rclcpp::Subscription<PointCloud2>::SharedPtr sub_;
  StageStats stats_;
};

}  // namespace pcl_pipeline

// 部品(component)として登録する。1 プロセスにまとめるとき(M3)も、単独の実行ファイルとしても使える
RCLCPP_COMPONENTS_REGISTER_NODE(pcl_pipeline::VoxelDownsample)
