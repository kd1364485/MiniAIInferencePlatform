#!/bin/bash
# M0: talker -> listener が届くかを、DDS の探索範囲(ROS_AUTOMATIC_DISCOVERY_RANGE)ごとに確かめる
# 使い方: bash tools/m0_pubsub_check.sh [SUBNET|LOCALHOST|...]   (省略時は両方)
set -u
set +u; source /opt/ros/jazzy/setup.bash; set -u
export RCUTILS_LOGGING_BUFFERED_STREAM=0
for range in ${1:-SUBNET LOCALHOST}; do
  export ROS_AUTOMATIC_DISCOVERY_RANGE=$range
  timeout 12 ros2 run demo_nodes_cpp talker > /dev/null 2>&1 &
  tp=$!
  sleep 2
  n=$(timeout 6 ros2 run demo_nodes_cpp listener 2>&1 | grep -c "I heard")
  kill $tp 2>/dev/null; wait $tp 2>/dev/null
  echo "range=$range  received=$n (in ~6 s, talker publishes 1 Hz)"
done
