#!/bin/bash
# S5: 主な比較を、5 回ずつ、交互に繰り返す(ばらつきに負けない結論かを確かめる)
B=$HOME/probe/build
MODEL=$HOME/seiya7183517/TcpServer/models/mnist.onnx
for r in 1 2 3 4 5; do
  for spec in "1 0 sleep 0" "1 0 net 0" "1 100 sleep 0" "1 100 spin 0" "8 0 sleep 0" "8 0 net 0" "8 0 sleep 12"; do
    set -- $spec
    echo "r=$r $($B/run_bench $MODEL $1 3 1 $2 $3 $4)"
  done
done