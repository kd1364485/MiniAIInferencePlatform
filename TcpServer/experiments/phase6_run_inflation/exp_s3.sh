#!/bin/bash
B=$HOME/probe/build
MODEL=$HOME/seiya7183517/TcpServer/models/mnist.onnx
echo "######## S3 round=$1"
# 引数: threads think_us mode burners
for spec in "1 0 sleep 0" "1 0 net 0" "8 0 sleep 0" "8 0 net 0" "8 0 sleep 4" "8 0 sleep 12" "8 0 net 12"; do
  set -- $spec
  $B/run_bench $MODEL $1 4 1 $2 $3 $4
done