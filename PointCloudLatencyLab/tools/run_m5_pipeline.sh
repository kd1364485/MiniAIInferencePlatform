#!/bin/bash
# M5: パイプラインの中で、voxel を CPU 版と CUDA 版で比べる(1 プロセス + intra-process、30 秒ずつ)。
# 点の数は 32 / 64 / 128 本。各回、nvidia-smi で GPU の状態も 200 ms ごとに記録する。
# 使い方(プロジェクトのフォルダで): bash tools/run_m5_pipeline.sh [rounds] [seconds]
set -u
ROUNDS=${1:-3}
SEC=${2:-30}
set +u; source env.sh; set -u
OUT=$PWD/results/m5; mkdir -p "$OUT"
for r in $(seq 1 "$ROUNDS"); do
  for rings in 32 64 128; do
    for dev in cpu cuda; do
      d=$OUT/${dev}_${rings}_r$r
      rm -rf "$d"; mkdir -p "$d"
      cuda=false; [ $dev = cuda ] && cuda=true
      nvidia-smi --query-gpu=timestamp,pstate,clocks.sm,utilization.gpu --format=csv,noheader,nounits \
        -lms 200 > "$d/gpu_clock.csv" &
      sp=$!
      timeout -s INT "$SEC" ros2 launch pcl_pipeline composed.launch.py ipc:=true use_cuda:=$cuda \
        rings:=$rings csv_dir:="$d" > "$d/launch.log" 2>&1
      kill $sp; wait $sp 2>/dev/null
      gpu=$(awk -F", " "NR>25{c[\$2]++; s[\$2]+=\$3} END{for(k in c) printf \"%s=%d%%(%.0fMHz) \", k, 100*c[k]/(NR-25), s[k]/c[k]}" "$d/gpu_clock.csv")
      echo "$gpu" > "$d/gpu_state.txt"
      echo "round=$r rings=$rings dev=$dev gpu: $gpu  voxel: $(grep -o "voxel on [A-Z]*" "$d/launch.log" | head -1)"
      sleep 2
    done
  done
done
