// 受け取るだけのノード(M2 の比較用)。RViz2 の「描画」を除いて、「受け取る相手が 1 つ増える」ことだけの影響を見る。
#include <memory>
#include <string>

#include "pcl_pipeline/node_util.hpp"
#include "rclcpp_components/register_node_macro.hpp"

namespace pcl_pipeline {

class Sink : public rclcpp::Node {
public:
  explicit Sink(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : Node("sink", options) {
    const auto topic = declare_parameter("topic", std::string("sensor/points"));
    sub_ = create_subscription<PointCloud2>(topic, cloud_qos(this),
                                            [this](PointCloud2::ConstSharedPtr) { ++count_; });
  }
  ~Sink() override { RCLCPP_INFO(get_logger(), "received %zu messages", count_); }

private:
  rclcpp::Subscription<PointCloud2>::SharedPtr sub_;
  size_t count_ = 0;
};

}  // namespace pcl_pipeline

// 部品(component)として登録する。1 プロセスにまとめるとき(M3)も、単独の実行ファイルとしても使える
RCLCPP_COMPONENTS_REGISTER_NODE(pcl_pipeline::Sink)
