#!/bin/bash
# Phase 8 の未解決「-p で公開すると約 33% 遅い」の原因を、経路を分けて確かめる
# 使い方(TcpServer/ で): bash experiments/phase8_port_publish/exp_port.sh <round>
# 条件(サーバーは同じ infer-server:gpu、mnist、cuda、I/O 2 + 推論 2、b32。bench は 100 接続 5 秒):
#   host   : --network host。bench → 127.0.0.1:8080(Phase 8 の基準)
#   p_loop : -p 8080:8080。bench → 127.0.0.1:8080(docker-proxy がユーザー空間で中継する経路)
#   p_cip  : 同じコンテナ。bench → コンテナの IP:8080(ブリッジを直接。proxy も NAT も通らない)
#   p_nat  : 同じコンテナ。bench → このマシンの 外向きのインターフェース(この PC では eth2)の IP:8080(iptables / nftables の DNAT。proxy は通らない)
# 各条件で、docker-proxy が使った CPU 時間(/proc/<pid>/stat の utime+stime)も測る
set -u
R=${1:-1}
B=${INFER_BENCH:-$HOME/probe/build_bench/infer_bench}   # Release ビルドの bench(cmake --build <dir> --target infer_bench)
DATA=$HOME/OnnxLearn/data/MNIST/raw
ARGS="/models/mnist.onnx 2 1 0 32 2 cuda"
HZ=$(getconf CLK_TCK)
proxy_ticks() { local t=0; for p in $(pgrep -x docker-proxy); do t=$((t + $(awk '{print $14+$15}' /proc/$p/stat))); done; echo $t; }
bench() {  # $1=条件名 $2=接続先
  # 測る前に 2 秒の空打ち(結果は捨てる)。直前の条件でサーバーや GPU が温まっているかどうかで、差が出ないようにする
  INFER_HOST=$2 $B 100 2 $DATA > /dev/null 2>&1
  local t0=$(proxy_ticks)
  local out=$(INFER_HOST=$2 $B 100 5 $DATA 2>&1 | tail -3 | tr '\n' ' ')
  local t1=$(proxy_ticks)
  echo "round=$R cond=$1 dst=$2 proxy_cpu_ms=$(( (t1 - t0) * 1000 / HZ )) | $out"
}
wait_up() { for i in $(seq 1 60); do (echo > /dev/tcp/$1/8080) 2>/dev/null && return 0; sleep 0.5; done; echo "server not up"; return 1; }

docker run -d --rm --name mi_exp --gpus all --network host -v "$PWD/models:/models:ro" infer-server:gpu $ARGS > /dev/null
wait_up 127.0.0.1; sleep 3
bench host 127.0.0.1
docker stop mi_exp > /dev/null

docker run -d --rm --name mi_exp --gpus all -p 8080:8080 -v "$PWD/models:/models:ro" infer-server:gpu $ARGS > /dev/null
CIP=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' mi_exp)
HIP=$(ip -4 route get 1.1.1.1 | awk '{for(i=1;i<NF;i++) if($i=="src") print $(i+1)}')   # 外向きのインターフェースの IP
wait_up 127.0.0.1; sleep 3
echo "# docker-proxy: $(pgrep -x docker-proxy | wc -l) 個  container_ip=$CIP  host_ip=$HIP"
bench p_loop 127.0.0.1
bench p_cip  $CIP
bench p_nat  $HIP
docker stop mi_exp > /dev/null
