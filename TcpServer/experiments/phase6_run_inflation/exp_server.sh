#!/bin/bash
# S1: ネットワークありの infer_server で、Run() の内訳・コンテキストスイッチ・softirq を測る
# 使い方: bash exp_server.sh <workers> <round>
B=$HOME/probe/build
MODEL=$HOME/seiya7183517/TcpServer/models/mnist.onnx
PERF=$(ls /usr/lib/linux-tools/*/perf | head -1)
W=$1
ROUND=$2
PW=${3:-0}
TAG=w${W}_pw${PW}_r${ROUND}
OUT=$HOME/probe/exp/$TAG
mkdir -p $HOME/probe/exp

# サーバーを起動(メトリクスのポートは無効。ポートは 18080)
INFER_METRICS_PORT=0 INFER_PORT=18080 $B/infer_server $MODEL $W 1 $PW 1 0 cpu > $OUT.srv.log 2>&1 &
SP=$!
for i in $(seq 1 50); do grep -q "infer_server: port" $OUT.srv.log 2>/dev/null && break; sleep 0.2; done
sleep 0.5

snap() {  # /proc/stat の1行目(全 CPU)と、NET_RX / NET_TX の総数
  awk '/^cpu /{print "STAT", $2,$3,$4,$5,$6,$7,$8,$9}' /proc/stat
  awk '/NET_RX:/{s=0; for(i=2;i<=NF;i++) s+=$i; print "NETRX", s} /NET_TX:/{s=0; for(i=2;i<=NF;i++) s+=$i; print "NETTX", s}' /proc/softirqs
}
snap > $OUT.snap1

# perf stat: サーバーのプロセスに、5.5 秒間つなぐ(ソフトウェアのイベントだけ)
$PERF stat -e task-clock,context-switches,cpu-migrations,page-faults -p $SP -o $OUT.perf -- sleep 6.5 &
PP=$!
sleep 0.3
INFER_PORT=18080 $B/infer_bench 100 5 dummy > $OUT.bench.log 2>&1
snap > $OUT.snap2
wait $PP 2>/dev/null

kill -INT $SP
wait $SP 2>/dev/null

# --- 集計 ---
echo "######## workers=$W per_worker_session=$PW round=$ROUND"
echo "-- bench"; grep -E "req/s|throughput|p50|p95|p99|completed|requests" $OUT.bench.log | head -6
echo "-- server"; sed -n '/requests=/,$p' $OUT.srv.log
echo "-- softirq / stat"
python3 - "$OUT.snap1" "$OUT.snap2" <<'PY'
import sys
def load(p):
    d = {}
    for line in open(p):
        k, *v = line.split()
        d[k] = [int(x) for x in v]
    return d
a, b = load(sys.argv[1]), load(sys.argv[2])
sa, sb = a["STAT"], b["STAT"]
delta = [y - x for x, y in zip(sa, sb)]
user, nice, system, idle, iowait, irq, softirq, steal = delta
busy = user + nice + system + irq + softirq
total = busy + idle + iowait + steal
print(f"  全CPU: 忙しい時間 {busy} jiffies / 全体 {total} jiffies(10ms 単位)")
print(f"  softirq は、忙しい時間の {100*softirq/busy:.1f}% / 全体の {100*softirq/total:.2f}%  (softirq={softirq}, irq={irq}, system={system}, user={user})")
print(f"  NET_RX {b['NETRX'][0]-a['NETRX'][0]} 回 / NET_TX {b['NETTX'][0]-a['NETTX'][0]} 回(約 6 秒間)")
PY
echo "-- perf stat (サーバーのプロセス全体、software event)"
grep -E "task-clock|context-switches|cpu-migrations|page-faults|elapsed" $OUT.perf
