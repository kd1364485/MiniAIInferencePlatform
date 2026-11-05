#!/bin/bash
# S4: strace -c で、システムコールの回数を、リクエストあたりで数える(サーバーを strace の下で起動する)
B=$HOME/probe/clean
MODEL=$HOME/seiya7183517/TcpServer/models/mnist.onnx
PW=${1:-0}
OUT=$HOME/probe/exp/strace_pw$PW
mkdir -p $HOME/probe/exp
INFER_METRICS_PORT=0 INFER_PORT=18081 strace -f -c -e trace=futex,epoll_wait,recvfrom,sendto,read,write,epoll_ctl,accept4 -o $OUT.txt \
    $B/infer_server $MODEL 8 1 $PW 1 0 cpu > $OUT.srv.log 2>&1 &
ST=$!
for i in $(seq 1 100); do grep -q "infer_server: port" $OUT.srv.log 2>/dev/null && break; sleep 0.2; done
sleep 0.5
INFER_PORT=18081 $B/infer_bench 100 5 dummy > $OUT.bench.log 2>&1
SP=$(pgrep -P $ST infer_server | head -1)
kill -INT $SP
wait $ST 2>/dev/null
echo "######## strace  workers=8 per_worker_session=$PW"
grep -E "completed|req/s" $OUT.bench.log | head -2
grep -E "requests=" $OUT.srv.log
cat $OUT.txt | head -16