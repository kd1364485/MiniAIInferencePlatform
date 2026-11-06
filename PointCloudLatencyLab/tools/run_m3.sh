#!/bin/bash
# M3: 別プロセスと、1 プロセス(component container)を比べる。条件ごとに 30 秒。
#   procs      : 別プロセス(M1・M2 と同じ。DDS で受け渡す)
#   comp_dds   : 1 プロセス、intra-process なし(同じプロセスの中でも DDS を通る)
#   comp_ipc   : 1 プロセス、intra-process あり(unique_ptr をそのまま渡す)。1 スレッドで全ノードを回す
#   comp_ipc_mt: 同じ + 複数スレッドの executor(component_container_mt)
# それぞれ点の数 32 本(約 550 KB)と 64 本(約 1.1 MB)。CPU の使用率(PC 全体)と UDP の受信あふれも記録する。
# 使い方(プロジェクトのフォルダで): bash tools/run_m3.sh [rounds] [seconds]
set -u
ROUNDS=${1:-3}
SEC=${2:-30}
set +u; source env.sh; set -u
OUT=${M3_OUT:-$PWD/results/m3}
mkdir -p "$OUT"
rcvbuf() { awk '/^Udp:/{getline; print $6}' /proc/net/snmp; }
cpu() { awk '/^cpu /{print $2+$3+$4+$6+$7+$8, $5}' /proc/stat; }   # busy idle(単位: tick)

for r in $(seq 1 "$ROUNDS"); do
  for rings in 32 64; do
    for c in procs comp_dds comp_ipc comp_ipc_mt; do
      d=$OUT/${c}_${rings}_r$r
      rm -rf "$d"; mkdir -p "$d"
      case $c in
        procs) cmd="pipeline.launch.py rviz:=false" ;;
        comp_dds) cmd="composed.launch.py ipc:=false" ;;
        comp_ipc) cmd="composed.launch.py ipc:=true" ;;
        comp_ipc_mt) cmd="composed.launch.py ipc:=true container:=component_container_mt" ;;
      esac
      e0=$(rcvbuf); read b0 i0 < <(cpu)
      timeout -s INT "$SEC" ros2 launch pcl_pipeline $cmd rings:=$rings csv_dir:="$d" > "$d/launch.log" 2>&1
      e1=$(rcvbuf); read b1 i1 < <(cpu)
      echo "$((e1 - e0))" > "$d/udp_rcvbuf_errors.txt"
      busy=$(awk "BEGIN{printf \"%.1f\", 100*($b1-$b0)/(($b1-$b0)+($i1-$i0))}")
      echo "$busy" > "$d/cpu_busy_pct.txt"
      echo "round=$r rings=$rings cond=$c udp_rcvbuf_errors=$((e1 - e0)) cpu_busy=${busy}%"
      sleep 2
    done
  done
done
