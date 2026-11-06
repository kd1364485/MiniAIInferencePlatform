#!/bin/bash
# M5: voxel の CPU 版と CUDA 版を、点の数を変えて比べる(voxel_bench を 3 回)。
# 同時に nvidia-smi で GPU の状態(P-state、SM のクロック)を 200 ms ごとに記録する(MiniAI で、
# 条件によって GPU が P8 に落ちて遅くなったことがあるため)。
# 使い方(プロジェクトのフォルダで): bash tools/run_m5_bench.sh [rounds]
set -u
ROUNDS=${1:-3}
set +u; source env.sh; set -u
OUT=$PWD/results/m5; mkdir -p "$OUT"
for r in $(seq 1 "$ROUNDS"); do
  nvidia-smi --query-gpu=timestamp,pstate,clocks.sm,clocks.mem,utilization.gpu,power.draw \
    --format=csv,noheader,nounits -lms 200 > "$OUT/gpu_clock_bench_r$r.csv" &
  sp=$!
  sleep 1
  ./install/pcl_pipeline/lib/pcl_pipeline/voxel_bench 0.2 50 > "$OUT/voxel_bench_r$r.txt"
  kill $sp; wait $sp 2>/dev/null
  echo "== round $r"; cat "$OUT/voxel_bench_r$r.txt"
  awk -F", " '{c[$2]++; s[$2]+=$3} END{for(k in c) printf "  GPU %s: %d samples, avg SM %.0f MHz\n", k, c[k], s[k]/c[k]}' "$OUT/gpu_clock_bench_r$r.csv"
  sleep 3
done
