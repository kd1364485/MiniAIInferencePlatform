# PointCloudLatencyLab を、ROS 2 Jazzy の公式イメージでビルド・テストする(CPU 版。CUDA は入れない)。
#   docker build -t pointcloud-latency-lab .
#   docker run --rm pointcloud-latency-lab                     # パイプラインを 1 プロセスで動かし、ログを出す
#   docker run --rm pointcloud-latency-lab colcon test-result --verbose   # ビルド時のテスト結果を見る
FROM ros:jazzy-ros-base

SHELL ["/bin/bash", "-c"]
# Cyclone DDS(このプロジェクトの RMW)。他の依存は rosdep で入れる
RUN apt-get update \
 && apt-get install -y --no-install-recommends ros-jazzy-rmw-cyclonedds-cpp \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /ws
COPY src/ src/
RUN apt-get update \
 && rosdep install --from-paths src --ignore-src -y --skip-keys rviz2 \
 && rm -rf /var/lib/apt/lists/*

ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    RCUTILS_LOGGING_BUFFERED_STREAM=0
RUN source /opt/ros/jazzy/setup.bash \
 && colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release \
 && colcon test --event-handlers console_cohesion+ \
 && colcon test-result --verbose

COPY docker/entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "launch", "pcl_pipeline", "composed.launch.py", "ipc:=true"]
