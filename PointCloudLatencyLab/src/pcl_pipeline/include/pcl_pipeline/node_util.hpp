// ノードで共通に使うもの: PointCloud2 との変換、QoS、処理時間の集計
#pragma once

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "pcl_pipeline/filters.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"

namespace pcl_pipeline {

using sensor_msgs::msg::PointCloud2;

// 点群のトピックの QoS。既定は「遅れたら捨てる」(best_effort、最新 5 件)。
// パラメータ qos_reliable:=true で reliable(落ちたら送り直す)にする(M4 で比べる)
inline rclcpp::QoS cloud_qos(rclcpp::Node * node) {
  if (!node->has_parameter("qos_reliable")) node->declare_parameter("qos_reliable", false);
  auto qos = rclcpp::SensorDataQoS().keep_last(5);
  if (node->get_parameter("qos_reliable").as_bool()) qos.reliable();
  return qos;
}

// std::vector<Point> → PointCloud2。並びが同じなので 1 回の memcpy で済む
inline std::unique_ptr<PointCloud2> to_msg(const std::vector<Point> & pts,
                                           const std_msgs::msg::Header & header) {
  auto msg = std::make_unique<PointCloud2>();
  msg->header = header;
  sensor_msgs::PointCloud2Modifier mod(*msg);
  mod.setPointCloud2Fields(4, "x", 1, sensor_msgs::msg::PointField::FLOAT32, "y", 1,
                           sensor_msgs::msg::PointField::FLOAT32, "z", 1,
                           sensor_msgs::msg::PointField::FLOAT32, "intensity", 1,
                           sensor_msgs::msg::PointField::FLOAT32);
  mod.resize(pts.size());
  msg->height = 1;
  msg->width = static_cast<uint32_t>(pts.size());
  msg->is_dense = true;
  if (!pts.empty()) std::memcpy(msg->data.data(), pts.data(), pts.size() * sizeof(Point));
  return msg;
}

// PointCloud2 → std::vector<Point>。自分のノードが出した形なら memcpy、それ以外は 1 点ずつ読む
inline std::vector<Point> from_msg(const PointCloud2 & msg) {
  const size_t n = static_cast<size_t>(msg.width) * msg.height;
  std::vector<Point> pts(n);
  const bool same_layout = msg.point_step == sizeof(Point) && msg.fields.size() == 4 &&
                           msg.fields[0].name == "x" && msg.fields[0].offset == 0 &&
                           msg.fields[1].offset == 4 && msg.fields[2].offset == 8 &&
                           msg.fields[3].name == "intensity" && msg.fields[3].offset == 12 &&
                           msg.row_step == msg.width * msg.point_step;
  if (same_layout) {
    if (n) std::memcpy(pts.data(), msg.data.data(), n * sizeof(Point));
    return pts;
  }
  sensor_msgs::PointCloud2ConstIterator<float> x(msg, "x"), y(msg, "y"), z(msg, "z");
  for (size_t i = 0; i < n; ++i, ++x, ++y, ++z) pts[i] = {*x, *y, *z, 0.f};
  return pts;
}

// 各段の処理時間と、センサーの時刻(header.stamp)からの経過時間を集め、report_every 回ごとにログに出す。
// パラメータ csv_dir を指定すると、1 フレームごとに <csv_dir>/<ノード名>.csv へ 1 行書く(M2):
//   stamp_ns  センサーで付けた時刻(フレームの ID を兼ねる)
//   recv_ns   受け取って処理を始めた時刻(= done_ns - 処理時間)
//   done_ns   処理が終わり、送る直前の時刻
// 時刻はどれも同じ PC の ROS の時計(system time)なので、ノードをまたいで引き算できる。
class StageStats {
public:
  StageStats(rclcpp::Node * node, int report_every = 50)
  : node_(node), report_every_(report_every) {
    const auto dir = node_->declare_parameter("csv_dir", std::string(""));
    if (!dir.empty()) {
      const auto path = dir + "/" + node_->get_name() + ".csv";
      csv_.open(path);
      if (!csv_) {
        RCLCPP_ERROR(node_->get_logger(), "cannot open %s", path.c_str());
      } else {
        csv_ << "stamp_ns,recv_ns,done_ns,in_pts,out_pts\n";
      }
    }
  }

  void add(double proc_ms, const builtin_interfaces::msg::Time & stamp, size_t in_pts,
           size_t out_pts) {
    const int64_t done_ns = node_->now().nanoseconds();
    const int64_t stamp_ns = rclcpp::Time(stamp).nanoseconds();
    if (csv_.is_open()) {
      csv_ << stamp_ns << ',' << done_ns - static_cast<int64_t>(proc_ms * 1e6) << ',' << done_ns
           << ',' << in_pts << ',' << out_pts << '\n';
    }
    proc_ms_.push_back(proc_ms);
    since_sensor_ms_.push_back((done_ns - stamp_ns) / 1e6);
    in_sum_ += in_pts;
    out_sum_ += out_pts;
    if (static_cast<int>(proc_ms_.size()) >= report_every_) report();
  }

private:
  static double avg(const std::vector<double> & v) {
    double s = 0;
    for (double x : v) s += x;
    return v.empty() ? 0 : s / v.size();
  }
  void report() {
    const auto n = proc_ms_.size();
    RCLCPP_INFO(node_->get_logger(),
                "frames=%zu  points in=%zu out=%zu  proc avg=%.2f max=%.2f ms  "
                "since_sensor avg=%.2f max=%.2f ms",
                n, in_sum_ / n, out_sum_ / n, avg(proc_ms_),
                *std::max_element(proc_ms_.begin(), proc_ms_.end()), avg(since_sensor_ms_),
                *std::max_element(since_sensor_ms_.begin(), since_sensor_ms_.end()));
    proc_ms_.clear();
    since_sensor_ms_.clear();
    in_sum_ = out_sum_ = 0;
    if (csv_.is_open()) csv_.flush();
  }

  rclcpp::Node * node_;
  int report_every_;
  std::vector<double> proc_ms_, since_sensor_ms_;
  size_t in_sum_ = 0, out_sum_ = 0;
  std::ofstream csv_;
};

// 経過時間 [ms] を測る小さな道具
class Stopwatch {
public:
  Stopwatch() : t0_(std::chrono::steady_clock::now()) {}
  double ms() const {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0_)
      .count();
  }

private:
  std::chrono::steady_clock::time_point t0_;
};

}  // namespace pcl_pipeline
