// 地面除去: RANSAC で水平に近い平面を探し、そこから distance_threshold 以内の点を除く
#include <memory>
#include <vector>

#include "pcl_pipeline/filters.hpp"
#include "pcl_pipeline/node_util.hpp"
#include "rclcpp_components/register_node_macro.hpp"

namespace pcl_pipeline {

class GroundRemoval : public rclcpp::Node {
public:
  explicit GroundRemoval(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : Node("ground_removal", options), stats_(this) {
    p_.dist = static_cast<float>(declare_parameter("distance_threshold", 0.2));
    p_.iterations = static_cast<int>(declare_parameter("ransac_iterations", 50));
    pub_ = create_publisher<PointCloud2>("ground_removed/points", cloud_qos(this));
    sub_ = create_subscription<PointCloud2>(
      "downsampled/points", cloud_qos(this), [this](PointCloud2::ConstSharedPtr msg) { on_cloud(*msg); });
  }

private:
  void on_cloud(const PointCloud2 & msg) {
    Stopwatch sw;
    const auto in = from_msg(msg);
    const auto out = remove_ground(in, p_.dist, p_.iterations, seq_++);
    auto out_msg = to_msg(out, msg.header);  // header(センサーの時刻)はそのまま引き継ぐ
    stats_.add(sw.ms(), msg.header.stamp, in.size(), out.size());
    pub_->publish(std::move(out_msg));
  }

  struct Params {
    float dist;
    int iterations;
  } p_;
  uint32_t seq_ = 0;
  rclcpp::Publisher<PointCloud2>::SharedPtr pub_;
  rclcpp::Subscription<PointCloud2>::SharedPtr sub_;
  StageStats stats_;
};

}  // namespace pcl_pipeline

// 部品(component)として登録する。1 プロセスにまとめるとき(M3)も、単独の実行ファイルとしても使える
RCLCPP_COMPONENTS_REGISTER_NODE(pcl_pipeline::GroundRemoval)
