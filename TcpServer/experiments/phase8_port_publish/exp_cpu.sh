#!/bin/bash
# 2026/11/05 追加調査の続き: 「--network host がブリッジより約 35% 遅い」理由を調べる
# 仮説: loopback では、送った瞬間に相手の受信処理まで同じ CPU で進む。サーバーのスレッドが 1 件ごとに起こされ、
#       bench と CPU を取り合い、Run() の中の CPU の部分(転送の準備、カーネルの起動、同期待ち)が遅れる
# 確かめ方: サーバー(コンテナ)と bench を、別々の CPU に分ける。取り合いが原因なら、host とブリッジの差は縮むはず
#   shared: CPU の指定なし(今までと同じ)
#   split : サーバー = CPU 0-9(物理コア 5 個)、bench = CPU 10-19(残りの 5 個)。SMT の兄弟は同じ側に入る
# 使い方(TcpServer/ で): bash experiments/phase8_port_publish/exp_cpu.sh <round>
set -u
R=${1:-1}
B=${INFER_BENCH:-$HOME/probe/build_bench/infer_bench}
DATA=$HOME/OnnxLearn/data/MNIST/raw
ARGS=${ARGS:-"/models/mnist.onnx 2 1 0 32 2 cuda"}   # 環境変数 ARGS で上書きできる
OUT=experiments/phase8_port_publish/logs; mkdir -p $OUT
netrx() { awk '/NET_RX/{s=0; for(i=2;i<=NF;i++) s+=$i; print s}' /proc/softirqs; }
ctx() { local p=$(docker inspect -f '{{.State.Pid}}' mi_exp); cat /proc/$p/task/*/status | awk '/^voluntary_ctxt_switches/{v+=$2} /^nonvoluntary_ctxt_switches/{n+=$2} END{print v, n}'; }

one() {  # $1=net(host|bridge) $2=cpu(shared|split)
  local net_arg="--network host" cpu_arg="" bench_pin=""
  [ $1 = bridge ] && net_arg="-p 8080:8080"
  [ $2 = split ] && cpu_arg="--cpuset-cpus=0-9" && bench_pin="taskset -c 10-19"
  docker run -d --rm --name mi_exp --gpus all $net_arg $cpu_arg -v "$PWD/models:/models:ro" infer-server:gpu $ARGS > /dev/null
  sleep 6
  local dst=127.0.0.1
  [ $1 = bridge ] && dst=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' mi_exp)
  INFER_HOST=$dst $bench_pin $B 100 2 $DATA > /dev/null 2>&1          # 空打ち
  local log=$OUT/srv_r${R}_$1_$2.log
  docker logs -f mi_exp > $log 2>&1 &
  local lp=$!
  local n0=$(netrx); read v0 nv0 < <(ctx)
  local res=$(INFER_HOST=$dst $bench_pin $B 100 5 $DATA 2>&1 | grep -o "req/s: [0-9.]*")
  local n1=$(netrx); read v1 nv1 < <(ctx)
  docker stop mi_exp > /dev/null; sleep 1; kill $lp 2>/dev/null
  local reqs=$(grep -o "requests=[0-9]*" $log | cut -d= -f2)
  local st=$(grep -o "avg ORT Run()=[0-9.]* us  batches=[0-9]*  avg_batch=[0-9.]*" $log)
  echo "round=$R net=$1 cpu=$2 | $res | $st | srv_ctx/req vol=$(awk "BEGIN{printf \"%.3f\", ($v1-$v0)/$reqs}") invol=$(awk "BEGIN{printf \"%.3f\", ($nv1-$nv0)/$reqs}") | NET_RX/req=$(awk "BEGIN{printf \"%.2f\", ($n1-$n0)/$reqs}")"
}
for net in host bridge; do for cpu in ${CPUS:-shared split}; do one $net $cpu; done; done
