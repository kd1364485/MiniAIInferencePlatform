// 間引き: 一辺 leaf_size の立方体ごとに、中の点を重心 1 点にまとめる
#include <memory>
#include <vector>

#include "pcl_pipeline/filters.hpp"
#include "pcl_pipeline/node_util.hpp"
#include "rclcpp_components/register_node_macro.hpp"

namespace pcl_pipeline {

class VoxelDownsample : public rclcpp::Node {
public:
  explicit VoxelDownsample(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : Node("voxel_downsample", options), stats_(this) {
    p_.leaf = static_cast<float>(declare_parameter("leaf_size", 0.2));
    pub_ = create_publisher<PointCloud2>("downsampled/points", cloud_qos(this));
    sub_ = create_subscription<PointCloud2>(
      "cropped/points", cloud_qos(this), [this](PointCloud2::ConstSharedPtr msg) { on_cloud(*msg); });
  }

private:
  void on_cloud(const PointCloud2 & msg) {
    Stopwatch sw;
    const auto in = from_msg(msg);
    const auto out = voxel_downsample(in, p_.leaf);
    auto out_msg = to_msg(out, msg.header);  // header(センサーの時刻)はそのまま引き継ぐ
    stats_.add(sw.ms(), msg.header.stamp, in.size(), out.size());
    pub_->publish(std::move(out_msg));
  }

  struct Params {
    float leaf;
  } p_;
  rclcpp::Publisher<PointCloud2>::SharedPtr pub_;
  rclcpp::Subscription<PointCloud2>::SharedPtr sub_;
  StageStats stats_;
};

}  // namespace pcl_pipeline

// 部品(component)として登録する。1 プロセスにまとめるとき(M3)も、単独の実行ファイルとしても使える
RCLCPP_COMPONENTS_REGISTER_NODE(pcl_pipeline::VoxelDownsample)
