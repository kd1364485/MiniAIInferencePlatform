# このプロジェクトで使う ROS 2 の環境。使い方: source env.sh
# RMW は Cyclone DDS に固定する。この PC(WSL2、networkingMode=mirrored)では、既定の Fast DDS だと
# 別プロセスのノード同士がつながらなかった(M0 の記録を参照)。
source /opt/ros/jazzy/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export RCUTILS_LOGGING_BUFFERED_STREAM=0
if [ -f install/setup.bash ]; then source install/setup.bash; fi
