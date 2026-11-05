// M0: colcon でのビルド確認を兼ねた、最小の pub / sub。
// pub は送った時刻を Header.stamp に入れて 100 Hz で送り、sub は受け取った時刻との差(片道の遅延)を数える。
// 使い方: ros2 run m0_hello m0_hello pub   /   ros2 run m0_hello m0_hello sub
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/header.hpp"

using namespace std::chrono_literals;
using std_msgs::msg::Header;

class Pub : public rclcpp::Node {
public:
  Pub() : Node("m0_pub") {
    pub_ = create_publisher<Header>("m0/ping", 10);
    timer_ = create_wall_timer(10ms, [this] {
      Header h;
      h.frame_id = std::to_string(seq_++);
      h.stamp = now();  // 送る直前の時刻
      pub_->publish(h);
    });
  }
private:
  rclcpp::Publisher<Header>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  uint64_t seq_ = 0;
};

class Sub : public rclcpp::Node {
public:
  Sub() : Node("m0_sub") {
    sub_ = create_subscription<Header>("m0/ping", 10, [this](const Header & h) {
      lat_us_.push_back((now() - rclcpp::Time(h.stamp)).nanoseconds() / 1000.0);
    });
    timer_ = create_wall_timer(1s, [this] { report(); });
  }
private:
  void report() {
    if (lat_us_.empty()) { RCLCPP_INFO(get_logger(), "no messages yet"); return; }
    std::sort(lat_us_.begin(), lat_us_.end());
    auto pct = [&](double p) { return lat_us_[static_cast<size_t>(p * (lat_us_.size() - 1))]; };
    RCLCPP_INFO(get_logger(), "received=%zu  latency p50=%.1f us  p99=%.1f us  max=%.1f us",
                lat_us_.size(), pct(0.5), pct(0.99), lat_us_.back());
    lat_us_.clear();
  }
  rclcpp::Subscription<Header>::SharedPtr sub_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::vector<double> lat_us_;
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  const std::string mode = argc > 1 ? argv[1] : "";
  if (mode == "pub") {
    rclcpp::spin(std::make_shared<Pub>());
  } else if (mode == "sub") {
    rclcpp::spin(std::make_shared<Sub>());
  } else {
    std::fprintf(stderr, "usage: m0_hello pub|sub\n");
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
