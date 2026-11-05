#!/bin/bash
# M4: CPU に負荷をかけたとき、遅延の揺れ(p99・最大)がどう変わり、何が効くかを見る。条件ごとに 30 秒。
# 負荷 = 論理 CPU の数(20)だけの bash の空回し(stress-ng は入れていない)。
#   ipc_idle       : 1 プロセス + intra-process(M3 で一番よかった形)、負荷なし
#   ipc_load       : 同じ + 負荷(負荷も普通の優先度)
#   ipc_load_nice  : 同じ + 負荷を nice 19(優先度を一番下げる。root は要らない)
#   ipc_load_split : パイプラインを CPU 0-9、負荷を CPU 10-19 に分ける(taskset。SMT の兄弟は同じ側)
#   ipcmt_load     : 複数スレッドの executor + 負荷
#   procs_be       : 別プロセス、best_effort(M2 と同じ)、負荷なし
#   procs_rel      : 別プロセス、reliable、負荷なし
#   procs_rel_load : 別プロセス、reliable、負荷あり
# SCHED_FIFO(実時間の優先度)は、この WSL では root の設定変更が要るので試していない(rtprio の上限が 0)。
# 使い方(プロジェクトのフォルダで): bash tools/run_m4.sh [rounds] [seconds] [条件 ...]
set -u
ROUNDS=${1:-3}
SEC=${2:-30}
shift 2 2>/dev/null
CONDS=${*:-"ipc_idle ipc_load ipc_load_nice ipc_load_split ipcmt_load procs_be procs_rel procs_rel_load"}
set +u; source env.sh; set -u
OUT=${M4_OUT:-$PWD/results/m4}
mkdir -p "$OUT"
NCPU=$(nproc)
TCK=$(getconf CLK_TCK)
rcvbuf() { awk '/^Udp:/{getline; print $6}' /proc/net/snmp; }
# パイプラインのプロセス(部品の実行ファイルと container)の CPU 時間の合計 [tick]
pipe_ticks() {
  local s=0 p
  for p in $(pgrep -f "install/pcl_pipeline/lib|component_container"); do
    s=$((s + $(awk '{print $14 + $15}' /proc/$p/stat 2>/dev/null || echo 0)))
  done
  echo $s
}

HOGS=()
start_hogs() {  # $1=nice $2=cpu の範囲(空ならどこでも)
  local i pin=""
  [ -n "$2" ] && pin="taskset -c $2"
  for i in $(seq 1 "$NCPU"); do
    nice -n "$1" $pin bash -c 'while :; do :; done' &
    HOGS+=($!)
  done
}
stop_hogs() { [ ${#HOGS[@]} -gt 0 ] && kill "${HOGS[@]}" 2>/dev/null; wait "${HOGS[@]}" 2>/dev/null; HOGS=(); }
trap stop_hogs EXIT

for r in $(seq 1 "$ROUNDS"); do
  for c in $CONDS; do
    d=$OUT/${c}_r$r
    rm -rf "$d"; mkdir -p "$d"
    pin=""
    case $c in
      ipc_idle)       cmd="composed.launch.py ipc:=true" ;;
      ipc_load)       cmd="composed.launch.py ipc:=true"; start_hogs 0 "" ;;
      ipc_load_nice)  cmd="composed.launch.py ipc:=true"; start_hogs 19 "" ;;
      ipc_load_split) cmd="composed.launch.py ipc:=true"; start_hogs 0 "10-19"; pin="taskset -c 0-9" ;;
      ipcmt_load)     cmd="composed.launch.py ipc:=true container:=component_container_mt"; start_hogs 0 "" ;;
      procs_be)       cmd="pipeline.launch.py rviz:=false" ;;
      procs_rel)      cmd="pipeline.launch.py rviz:=false qos_reliable:=true" ;;
      procs_rel_load) cmd="pipeline.launch.py rviz:=false qos_reliable:=true"; start_hogs 0 "" ;;
    esac
    e0=$(rcvbuf)
    # CPU 時間は裏で測る(起動 3 秒後から、止める 3 秒前まで)。ros2 launch は前で動かす:
    # 非対話の bash で & を付けると SIGINT が無視され、止められなくなるため
    ( sleep 3; a=$(pipe_ticks); sleep $((SEC - 6)); b=$(pipe_ticks)
      awk "BEGIN{printf \"%.2f\", ($b-$a)/$TCK/($SEC-6)}" > "$d/pipeline_cpu_cores.txt" ) &
    sp=$!
    $pin timeout -s INT "$SEC" ros2 launch pcl_pipeline $cmd csv_dir:="$d" > "$d/launch.log" 2>&1
    wait $sp
    stop_hogs
    e1=$(rcvbuf)
    echo "$((e1 - e0))" > "$d/udp_rcvbuf_errors.txt"
    cpu=$(cat "$d/pipeline_cpu_cores.txt")   # パイプラインの CPU 時間(1 秒あたり、CPU 何個分か)
    echo "round=$r cond=$c udp_rcvbuf_errors=$((e1 - e0)) pipeline_cpu=${cpu}cores"
    sleep 2
  done
done
