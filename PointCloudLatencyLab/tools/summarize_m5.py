#!/usr/bin/env python3
"""M5: results/m5/<cpu|cuda>_<rings>_r<回> をまとめる(voxel の処理時間、端から端、GPU の状態)。
使い方: python3 tools/summarize_m5.py [results/m5]
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from analyze_latency import STAGES, load, pct  # noqa: E402

root = sys.argv[1] if len(sys.argv) > 1 else 'results/m5'
runs = [d for d in os.listdir(root) if re.match(r'(cpu|cuda)_\d+_r\d+$', d)]
runs.sort(key=lambda d: (int(d.split('_')[1]), d.split('_')[0], d))
print(f'{"run":14s} {"points":>7s} {"lost":>4s} | {"voxel p50":>9s} {"p99":>6s} | '
      f'{"e2e p50":>7s} {"p99":>6s} {"max":>6s} [ms] | gpu (P-state: time share, avg SM clock)')
for r in runs:
    d = os.path.join(root, r)
    data = {s: load(os.path.join(d, s + '.csv')) for s in STAGES}
    st = sorted(data['sensor_sim'])[20:-5]
    full = [s for s in st if all(s in data[x] for x in STAGES)]
    vox = data['voxel_downsample']
    v = [(int(vox[s]['done_ns']) - int(vox[s]['recv_ns'])) / 1e6 for s in full]
    e = [(int(data['cluster'][s]['done_ns']) - s) / 1e6 for s in full]
    pts = int(data['sensor_sim'][st[0]]['out_pts'])
    g = open(os.path.join(d, 'gpu_state.txt')).read().strip()
    print(f'{r:14s} {pts:7d} {len(st) - len(full):4d} | {pct(v, 50):9.3f} {pct(v, 99):6.3f} | '
          f'{pct(e, 50):7.2f} {pct(e, 99):6.2f} {max(e):6.2f} | {g}')
