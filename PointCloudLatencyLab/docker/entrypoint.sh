#!/bin/bash
# コンテナの入口: ROS 2 とこのワークスペースを読み込んでから、渡されたコマンドを実行する
set -e
source /opt/ros/jazzy/setup.bash
source /ws/install/setup.bash
exec "$@"
