// 切り出し: 前後左右 ±50 m、高さ -2.5〜1.0 m(センサー基準)の範囲だけ残し、自車の範囲を除く
#include <memory>
#include <vector>

#include "pcl_pipeline/filters.hpp"
#include "pcl_pipeline/node_util.hpp"
#include "rclcpp_components/register_node_macro.hpp"

namespace pcl_pipeline {

class CropBox : public rclcpp::Node {
public:
  explicit CropBox(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : Node("crop_box", options), stats_(this) {
    p_.keep = {static_cast<float>(declare_parameter("min_x", -50.0)),
               static_cast<float>(declare_parameter("min_y", -50.0)),
               static_cast<float>(declare_parameter("min_z", -2.5)),
               static_cast<float>(declare_parameter("max_x", 50.0)),
               static_cast<float>(declare_parameter("max_y", 50.0)),
               static_cast<float>(declare_parameter("max_z", 1.0))};
    p_.ego = {-2.5f, -1.2f, -3.f, 2.5f, 1.2f, 1.f};  // 自車の範囲(センサーの真下の車体)
    pub_ = create_publisher<PointCloud2>("cropped/points", cloud_qos(this));
    sub_ = create_subscription<PointCloud2>(
      "sensor/points", cloud_qos(this), [this](PointCloud2::ConstSharedPtr msg) { on_cloud(*msg); });
  }

private:
  void on_cloud(const PointCloud2 & msg) {
    Stopwatch sw;
    const auto in = from_msg(msg);
    const auto out = crop_box(in, p_.keep, p_.ego);
    auto out_msg = to_msg(out, msg.header);  // header(センサーの時刻)はそのまま引き継ぐ
    stats_.add(sw.ms(), msg.header.stamp, in.size(), out.size());
    pub_->publish(std::move(out_msg));
  }

  struct Params {
    Box keep, ego;
  } p_;
  rclcpp::Publisher<PointCloud2>::SharedPtr pub_;
  rclcpp::Subscription<PointCloud2>::SharedPtr sub_;
  StageStats stats_;
};

}  // namespace pcl_pipeline

// 部品(component)として登録する。1 プロセスにまとめるとき(M3)も、単独の実行ファイルとしても使える
RCLCPP_COMPONENTS_REGISTER_NODE(pcl_pipeline::CropBox)
