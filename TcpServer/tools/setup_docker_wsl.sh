#!/bin/bash
# Phase 8 M1: WSL2 の Ubuntu に「Docker Engine」と「NVIDIA Container Toolkit」を入れる(公式の apt リポジトリ経由)。
#
# 使い方(中身を読んでから、自分の端末で):
#   sudo bash tools/setup_docker_wsl.sh
#   環境変数 ADD_USER_TO_DOCKER_GROUP=0 を付けると、docker グループへの追加を省く(後述の注意)
#
# やること:
#   1) 前提パッケージ                    2) Docker の公式 apt リポジトリを追加(署名鍵つき)
#   3) Docker Engine をインストール       4) NVIDIA の公式 apt リポジトリを追加し、Container Toolkit をインストール
#   5) Docker が「nvidia ランタイム」を使えるよう設定(/etc/docker/daemon.json)して再起動
#
# ★ 知っておくこと(セキュリティ):
#   - 外部の apt リポジトリ(download.docker.com、nvidia.github.io)と、その署名鍵を信頼する設定を追加する
#   - Docker は root 権限のデーモンとして動く。「docker グループ」に入ったユーザーは、事実上 root と同等の操作が
#     できる(ホストのファイルをコンテナにマウントして書き換えられる)。個人の開発 PC では一般的だが、
#     気になる場合は ADD_USER_TO_DOCKER_GROUP=0 にして、docker の前に毎回 sudo を付ける
#   - 前提: systemd が有効な WSL2 の Ubuntu(/etc/wsl.conf に systemd=true)、Windows 側に NVIDIA ドライバ
#
# NVIDIA Container Toolkit とは: コンテナを起動するとき、ホストの GPU ドライバのライブラリや
#   デバイスを自動でコンテナの中に注入する仕組み。コンテナのイメージに GPU ドライバを入れなくて済む
#   (ドライバはホストのものを使う)。WSL2 では、Windows 側のドライバが /usr/lib/wsl/lib に見せている
#   ライブラリが注入される。
set -euo pipefail

[ "$(id -u)" -eq 0 ] || { echo "sudo で実行してください:  sudo bash $0"; exit 1; }
. /etc/os-release
[ "${ID}" = "ubuntu" ] || { echo "Ubuntu 専用のスクリプトです(検出: ${ID})"; exit 1; }
[ "$(ps -p 1 -o comm=)" = "systemd" ] || { echo "systemd が動いていません(/etc/wsl.conf の [boot] systemd=true を確認)"; exit 1; }
export DEBIAN_FRONTEND=noninteractive
TARGET_USER=${SUDO_USER:-}

echo "== 1/5 前提パッケージ"
apt-get update
apt-get install -y ca-certificates curl gnupg

echo "== 2/5 Docker の公式 apt リポジトリを追加"
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu ${UBUNTU_CODENAME:-$VERSION_CODENAME} stable" \
  > /etc/apt/sources.list.d/docker.list

echo "== 3/5 Docker Engine をインストール"
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

echo "== 4/5 NVIDIA Container Toolkit の公式 apt リポジトリを追加してインストール"
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update
apt-get install -y nvidia-container-toolkit

echo "== 5/5 Docker が nvidia ランタイムを使うよう設定して再起動"
nvidia-ctk runtime configure --runtime=docker      # /etc/docker/daemon.json に nvidia ランタイムを登録する
systemctl enable --now docker
systemctl restart docker

if [ "${ADD_USER_TO_DOCKER_GROUP:-1}" = "1" ] && [ -n "$TARGET_USER" ]; then
  usermod -aG docker "$TARGET_USER"
  echo "ユーザー $TARGET_USER を docker グループに追加した(反映には WSL の再起動が必要)"
fi

echo
echo "完了。次の手順:"
echo "  1) Windows の PowerShell で  wsl --shutdown  を実行し、Ubuntu を開き直す(グループの反映)"
echo "  2) 動作確認:  docker run --rm --gpus all ubuntu:24.04 nvidia-smi"
