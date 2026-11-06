// 疑似 LiDAR: 地面と、車線を走る箱(車の大きさ)に光線を当てた点群を、一定の周期で出す。
// header.stamp は「点群ができて送る直前」の時刻。後ろの段の遅延は、ここからの経過時間で測る。
#include <cmath>
#include <memory>
#include <vector>

#include "pcl_pipeline/filters.hpp"
#include "pcl_pipeline/node_util.hpp"
#include "rclcpp_components/register_node_macro.hpp"

namespace pcl_pipeline {

class SensorSim : public rclcpp::Node {
public:
  explicit SensorSim(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : Node("sensor_sim", options), stats_(this) {
    params_.rings = declare_parameter("rings", 32);
    params_.azimuth_steps = declare_parameter("azimuth_steps", 1800);
    params_.max_range = declare_parameter("max_range", 100.0);
    params_.noise_stddev = declare_parameter("noise_stddev", 0.02);
    num_cars_ = declare_parameter("num_cars", 8);
    frame_id_ = declare_parameter("frame_id", std::string("base_link"));
    const double rate_hz = declare_parameter("rate_hz", 10.0);

    pub_ = create_publisher<PointCloud2>("sensor/points", cloud_qos(this));
    start_ = now();
    timer_ = create_wall_timer(std::chrono::duration<double>(1.0 / rate_hz), [this] { tick(); });
    RCLCPP_INFO(get_logger(), "rings=%d azimuth_steps=%d rate=%.1f Hz cars=%d", params_.rings,
                params_.azimuth_steps, rate_hz, num_cars_);
  }

private:
  // 車線(y = ±3.5, ±7 m)を、それぞれの速さで往復せずに流れる箱。80 m の区間を繰り返す
  std::vector<Box> cars(double t) const {
    static const float lanes[4] = {-7.f, -3.5f, 3.5f, 7.f};
    const float ground = -params_.sensor_height;
    std::vector<Box> boxes;
    for (int i = 0; i < num_cars_; ++i) {
      const float y = lanes[i % 4];
      const float speed = 5.f + 2.f * (i % 3);
      const float dir = y > 0 ? -1.f : 1.f;
      const float s = std::fmod(static_cast<float>(i) * 23.f + speed * static_cast<float>(t), 80.f);
      const float x = dir > 0 ? -40.f + s : 40.f - s;
      boxes.push_back({x - 2.25f, y - 0.9f, ground, x + 2.25f, y + 0.9f, ground + 1.5f});
    }
    return boxes;
  }

  void tick() {
    Stopwatch sw;
    const double t = (now() - start_).seconds();
    auto pts = simulate_scan(params_, cars(t), seq_++);
    std_msgs::msg::Header h;
    h.frame_id = frame_id_;
    h.stamp = now();
    auto msg = to_msg(pts, h);
    stats_.add(sw.ms(), h.stamp, 0, pts.size());
    pub_->publish(std::move(msg));
  }

  SimParams params_;
  int num_cars_;
  std::string frame_id_;
  rclcpp::Time start_;
  uint32_t seq_ = 0;
  rclcpp::Publisher<PointCloud2>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  StageStats stats_;
};

}  // namespace pcl_pipeline

// 部品(component)として登録する。1 プロセスにまとめるとき(M3)も、単独の実行ファイルとしても使える
RCLCPP_COMPONENTS_REGISTER_NODE(pcl_pipeline::SensorSim)
