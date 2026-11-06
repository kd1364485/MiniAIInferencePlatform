# このプロジェクトで使う ROS 2 の環境。使い方: source env.sh
# RMW は Cyclone DDS に固定する。この PC(WSL2、networkingMode=mirrored)では、既定の Fast DDS だと
# 別プロセスのノード同士がつながらなかった(M0 の記録を参照)。
source /opt/ros/jazzy/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export RCUTILS_LOGGING_BUFFERED_STREAM=0
# WSL の GPU ドライバ(libcuda)を先に読む。/lib/x86_64-linux-gnu にある Linux 用の libcuda が先に見つかると、
# 「CUDA の使える GPU がない」と言われる(MiniAI の CudaLearn で同じ問題があった)
if [ -d /usr/lib/wsl/lib ]; then export LD_LIBRARY_PATH=/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; fi
if [ -f install/setup.bash ]; then source install/setup.bash; fi
