#!/bin/bash
# M0: ROS 2 Jazzy を Ubuntu 24.04(WSL2)に入れる。公式の手順(deb パッケージ)に沿う。
# 使い方: bash tools/install_ros2_jazzy.sh   (途中で sudo のパスワードを聞かれる)
# 注意: apt upgrade(全体の更新)はしない。docker や nvidia のパッケージまで動くのを避けるため。
set -euo pipefail
sudo apt-get update
sudo apt-get install -y software-properties-common curl
sudo add-apt-repository -y universe

# ROS 2 の apt リポジトリ(ros2-apt-source の最新版)
V=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F '"tag_name"' | awk -F'"' '{print $4}')
CODENAME=$(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
DEB="$HOME/.cache/ros2-apt-source.deb"; mkdir -p "$HOME/.cache"
echo "ros2-apt-source $V ($CODENAME)"
curl -fL -o "$DEB" "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${V}/ros2-apt-source_${V}.${CODENAME}_all.deb"
sudo dpkg -i "$DEB"

sudo apt-get update
sudo apt-get install -y ros-jazzy-desktop ros-dev-tools
echo
echo "完了: source /opt/ros/jazzy/setup.bash で使える"
