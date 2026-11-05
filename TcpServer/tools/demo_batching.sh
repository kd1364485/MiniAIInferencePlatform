#!/bin/bash
# 録画用のデモ: 同じ推論サーバー(Docker、GPU)を、動的バッチ化なし(max_batch=1)と、あり(max_batch=32)で動かし、
# 同じ負荷(100 接続、5 秒)をかけて、req/s と遅延を比べる。
# 使い方(TcpServer/ で): bash tools/demo_batching.sh
# 必要なもの: infer-server:gpu のイメージ、models/mlp_dyn.onnx、Release の infer_bench(INFER_BENCH で場所を指定できる)
set -u
B=${INFER_BENCH:-$HOME/probe/build_bench/infer_bench}
MODEL=${MODEL:-/models/mlp_dyn.onnx}

run() {  # $1=max_batch
  echo
  echo "==================== max_batch=$1 ===================="
  # 引数: モデル  I/O スレッド 2  intra_op 1  共有 Session  max_batch  推論スレッド 1  GPU(cuda)
  docker run -d --rm --name demo_srv --gpus all --network host -v "$PWD/models:/models:ro" \
    infer-server:gpu $MODEL 2 1 0 $1 1 cuda > /dev/null
  until (echo > /dev/tcp/127.0.0.1/8080) 2>/dev/null; do sleep 0.3; done; sleep 2
  echo "\$ infer_bench 100 5 dummy     # 100 接続で 5 秒間、推論リクエストを送り続ける"
  $B 100 5 dummy 2>&1 | grep -E "req/s|latency"
  docker stop demo_srv > /dev/null
}

echo "推論サーバー(C++ / ONNX Runtime / CUDA、Docker)に、同じ負荷をかけて比べます"
run 1
run 32
echo
echo "→ 届いたリクエストをまとめて GPU に渡す(動的バッチ化)だけで、同じ GPU のまま req/s が伸び、遅延も縮む"
