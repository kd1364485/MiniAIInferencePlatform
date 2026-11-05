#!/usr/bin/env python3
"""M2: 各ノードの CSV(1 フレーム 1 行)を、センサーの時刻(stamp_ns)でつなぎ、段ごとの遅延を出す。

  受け渡し(hop) = この段の recv_ns - 前の段の done_ns   (シリアライズ、DDS、受け取り側が起きるまで)
  処理(proc)    = この段の done_ns - recv_ns
  端から端(e2e) = cluster の done_ns - stamp_ns

使い方: python3 tools/analyze_latency.py <run_dir> [<run_dir> ...] [--warmup 20] [--csv out.csv]
  run_dir には sensor_sim.csv 〜 cluster.csv が入っている(launch の csv_dir)。
  最初の --warmup フレームは捨てる(起動直後は DDS の探索などで遅い)。
"""
import argparse
import csv
import os
import sys

STAGES = ['sensor_sim', 'crop_box', 'voxel_downsample', 'ground_removal', 'cluster']


def load(path):
    with open(path, newline='') as f:
        return {int(r['stamp_ns']): r for r in csv.DictReader(f)}


def pct(values, p):
    if not values:
        return float('nan')
    s = sorted(values)
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]


def summarize(name, values):
    return {'metric': name, 'n': len(values), 'p50': pct(values, 50), 'p99': pct(values, 99),
            'max': max(values) if values else float('nan')}


def analyze(run_dir, warmup):
    data = {}
    for st in STAGES:
        path = os.path.join(run_dir, st + '.csv')
        if not os.path.exists(path):
            sys.exit(f'missing {path}')
        data[st] = load(path)

    stamps = sorted(data['sensor_sim'])[warmup:]
    rows = []
    for prev, cur in zip(STAGES, STAGES[1:]):
        hop = [(int(data[cur][s]['recv_ns']) - int(data[prev][s]['done_ns'])) / 1e6
               for s in stamps if s in data[cur] and s in data[prev]]
        rows.append(summarize(f'hop  {prev} -> {cur}', hop))
        proc = [(int(data[cur][s]['done_ns']) - int(data[cur][s]['recv_ns'])) / 1e6
                for s in stamps if s in data[cur]]
        rows.append(summarize(f'proc {cur}', proc))
    e2e = [(int(data['cluster'][s]['done_ns']) - s) / 1e6 for s in stamps if s in data['cluster']]
    rows.append(summarize('e2e  sensor -> cluster', e2e))
    hop_sum = [sum((int(data[c][s]['recv_ns']) - int(data[p][s]['done_ns'])) / 1e6
                   for p, c in zip(STAGES, STAGES[1:]))
               for s in stamps if all(s in data[st] for st in STAGES)]
    rows.append(summarize('sum of 4 hops', hop_sum))
    reached = sum(1 for s in stamps if s in data['cluster'])
    return rows, len(stamps), reached


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dirs', nargs='+')
    ap.add_argument('--warmup', type=int, default=20)
    ap.add_argument('--csv', help='結果を CSV にも書く')
    a = ap.parse_args()

    out = []
    for d in a.run_dirs:
        rows, sent, reached = analyze(d, a.warmup)
        print(f'== {d}   frames sent={sent} reached cluster={reached} (dropped {sent - reached})')
        print(f'   {"metric":36s} {"n":>5s} {"p50":>8s} {"p99":>8s} {"max":>8s}  [ms]')
        for r in rows:
            print(f'   {r["metric"]:36s} {r["n"]:5d} {r["p50"]:8.3f} {r["p99"]:8.3f} {r["max"]:8.3f}')
            out.append({'run': d, **r})
    if a.csv:
        with open(a.csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['run', 'metric', 'n', 'p50', 'p99', 'max'])
            w.writeheader()
            w.writerows(out)


if __name__ == '__main__':
    main()
