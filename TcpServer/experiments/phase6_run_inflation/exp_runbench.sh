#!/bin/bash
# S2: ネットワークなしの run_bench で、「眠った後の Run()」が遅いかを調べる
# 使い方: bash exp_runbench.sh <round>
B=$HOME/probe/build
MODEL=$HOME/seiya7183517/TcpServer/models/mnist.onnx
echo "######## S2 round=$1"
# threads=8 は、サーバーの workers=8 に相当。threads=1 は、競合がない単独の場合
for spec in "1 0 sleep" "1 100 sleep" "1 100 spin" "8 0 sleep" "8 100 sleep" "8 100 spin" "8 300 sleep" "8 300 spin"; do
  set -- $spec
  $B/run_bench $MODEL $1 4 1 $2 $3
done
