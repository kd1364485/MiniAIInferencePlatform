#!/bin/bash
# M2: 条件ごとに 30 秒ずつ動かし、各ノードの CSV と、OS の UDP の受信あふれ(RcvbufErrors)を記録する。
#   base : 5 ノードだけ(表示なし)
#   sink : + /sensor/points を受け取るだけのノード(受け取る相手が 1 つ増える)
#   rviz : + RViz2 で表示(受け取る相手が増える + 描画)
#   rings16 / rings64 : base で、点の数を半分 / 2 倍にする(1 フレームの大きさと遅延・取りこぼしの関係)
# 使い方(プロジェクトのフォルダで): bash tools/run_m2.sh [rounds] [seconds] [条件 ...]
set -u
ROUNDS=${1:-3}
SEC=${2:-30}
shift 2 2>/dev/null
CONDS=${*:-"base sink rviz rings16 rings64"}
set +u; source env.sh; set -u
OUT=${M2_OUT:-$PWD/results/m2}   # 環境変数 M2_OUT で保存先を変えられる
mkdir -p "$OUT"
rcvbuf() { awk '/^Udp:/{getline; print $6}' /proc/net/snmp; }

for r in $(seq 1 "$ROUNDS"); do
  for c in $CONDS; do
    d=$OUT/${c}_r$r
    rm -rf "$d"; mkdir -p "$d"
    extra="rviz:=false"
    case $c in
      sink) extra="rviz:=false sink:=true" ;;
      rviz) extra="rviz:=true" ;;
      rings16) extra="rviz:=false rings:=16" ;;
      rings64) extra="rviz:=false rings:=64" ;;
    esac
    e0=$(rcvbuf)
    timeout -s INT "$SEC" ros2 launch pcl_pipeline pipeline.launch.py $extra csv_dir:="$d" > "$d/launch.log" 2>&1
    e1=$(rcvbuf)
    echo "$((e1 - e0))" > "$d/udp_rcvbuf_errors.txt"
    echo "round=$r cond=$c udp_rcvbuf_errors=$((e1 - e0))"
    sleep 2
  done
done
