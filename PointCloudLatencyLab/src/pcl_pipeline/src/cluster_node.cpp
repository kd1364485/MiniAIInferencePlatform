// クラスタリング: 地面を除いた点を、距離 cluster_tolerance 以内でつながる塊に分け、塊ごとの箱を
// MarkerArray で出す(RViz2 で表示)。パイプラインの最後なので、ここでの since_sensor が端から端までの遅延。
#include <memory>
#include <vector>

#include "pcl_pipeline/filters.hpp"
#include "pcl_pipeline/node_util.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

namespace pcl_pipeline {

using visualization_msgs::msg::Marker;
using visualization_msgs::msg::MarkerArray;

class ClusterNode : public rclcpp::Node {
public:
  explicit ClusterNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : Node("cluster", options), stats_(this) {
    tolerance_ = static_cast<float>(declare_parameter("cluster_tolerance", 0.6));
    min_points_ = static_cast<int>(declare_parameter("min_points", 5));
    max_points_ = static_cast<int>(declare_parameter("max_points", 5000));
    // 箱は数が少なく、取りこぼすと RViz に古い箱が残るので reliable(既定の QoS)
    pub_ = create_publisher<MarkerArray>("objects/markers", 10);
    sub_ = create_subscription<PointCloud2>(
      "ground_removed/points", cloud_qos(this),
      [this](PointCloud2::ConstSharedPtr msg) { on_cloud(*msg); });
  }

private:
  void on_cloud(const PointCloud2 & msg) {
    Stopwatch sw;
    const auto in = from_msg(msg);
    const auto clusters = euclidean_cluster(in, tolerance_, min_points_, max_points_);

    auto out = std::make_unique<MarkerArray>();
    Marker clear;
    clear.header = msg.header;
    clear.action = Marker::DELETEALL;  // 前のフレームの箱を消す
    out->markers.push_back(clear);
    int id = 0;
    for (const auto & c : clusters) {
      Marker m;
      m.header = msg.header;
      m.ns = "clusters";
      m.id = id++;
      m.type = Marker::CUBE;
      m.action = Marker::ADD;
      const auto & b = c.bounds;
      m.pose.position.x = (b.min_x + b.max_x) / 2;
      m.pose.position.y = (b.min_y + b.max_y) / 2;
      m.pose.position.z = (b.min_z + b.max_z) / 2;
      m.pose.orientation.w = 1.0;
      m.scale.x = std::max(b.max_x - b.min_x, 0.1f);
      m.scale.y = std::max(b.max_y - b.min_y, 0.1f);
      m.scale.z = std::max(b.max_z - b.min_z, 0.1f);
      m.color.r = 0.1f;
      m.color.g = 0.8f;
      m.color.b = 0.3f;
      m.color.a = 0.4f;
      out->markers.push_back(m);
    }
    stats_.add(sw.ms(), msg.header.stamp, in.size(), clusters.size());
    pub_->publish(std::move(out));
  }

  float tolerance_;
  int min_points_, max_points_;
  rclcpp::Publisher<MarkerArray>::SharedPtr pub_;
  rclcpp::Subscription<PointCloud2>::SharedPtr sub_;
  StageStats stats_;
};

}  // namespace pcl_pipeline

// 部品(component)として登録する。1 プロセスにまとめるとき(M3)も、単独の実行ファイルとしても使える
RCLCPP_COMPONENTS_REGISTER_NODE(pcl_pipeline::ClusterNode)
