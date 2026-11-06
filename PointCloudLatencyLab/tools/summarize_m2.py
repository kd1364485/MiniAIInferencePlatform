#!/usr/bin/env python3
"""M2・M3: results/m2/<条件>_r<回> をまとめ、条件 × 回ごとに 1 行の表を出す。

  frames   : センサーが出したフレーム(最初の warmup フレームを除く)
  lost@hop : その受け渡しで届かなかったフレーム(前の段にはあり、次の段にない)
  rcvbuf   : その間に OS が UDP の受信バッファあふれで捨てたパケットの数(/proc/net/snmp)
使い方: python3 tools/summarize_m2.py [results/m2] [--warmup 20]
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from analyze_latency import STAGES, load, pct  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root', nargs='?', default='results/m2')
    ap.add_argument('--warmup', type=int, default=20)
    a = ap.parse_args()

    runs = sorted(d for d in os.listdir(a.root) if re.match(r'.+_r\d+$', d))
    order = ['base', 'sink', 'rviz', 'rings16', 'rings64', 'ipc_idle', 'ipc_load', 'ipc_load_nice', 'ipc_load_split', 'ipcmt_load', 'procs_be', 'procs_rel', 'procs_rel_load']
    runs.sort(key=lambda d: (order.index(d.rsplit('_r', 1)[0]) if d.rsplit('_r', 1)[0] in order
                             else 99, d))
    hdr = (f'{"run":12s} {"points":>7s} {"frames":>6s} {"lost@hop1..4":>14s} {"lost%":>6s} '
           f'{"rcvbuf / cpu":>16s} | {"hops p50":>8s} {"p99":>7s} | {"e2e p50":>8s} {"p99":>7s} {"max":>7s} [ms]')
    print(hdr)
    for r in runs:
        d = os.path.join(a.root, r)
        data = {st: load(os.path.join(d, st + '.csv')) for st in STAGES}
        stamps = sorted(data['sensor_sim'])[a.warmup:-5]  # 終わり際(止める順番の影響)も除く
        lost = []
        for prev, cur in zip(STAGES, STAGES[1:]):
            lost.append(sum(1 for s in stamps if s in data[prev] and s not in data[cur]))
        full = [s for s in stamps if all(s in data[st] for st in STAGES)]
        hops = [sum(int(data[c][s]['recv_ns']) - int(data[p][s]['done_ns'])
                    for p, c in zip(STAGES, STAGES[1:])) / 1e6 for s in full]
        e2e = [(int(data['cluster'][s]['done_ns']) - s) / 1e6 for s in full]
        pts = sorted(int(data['sensor_sim'][s]['out_pts']) for s in stamps)[len(stamps) // 2]
        try:
            rcv = open(os.path.join(d, 'udp_rcvbuf_errors.txt')).read().strip()
        except OSError:
            rcv = '?'
        try:
            rcv += " cpu" + open(os.path.join(d, "cpu_busy_pct.txt")).read().strip() + "%"
        except OSError:
            pass
        try:
            rcv += " " + open(os.path.join(d, "pipeline_cpu_cores.txt")).read().strip() + "c"
        except OSError:
            pass
        lost_pct = 100 * (len(stamps) - len(full)) / max(1, len(stamps))
        print(f'{r:12s} {pts:7d} {len(stamps):6d} {"/".join(map(str, lost)):>14s} {lost_pct:6.1f} '
              f'{rcv:>16s} | {pct(hops, 50):8.2f} {pct(hops, 99):7.2f} | '
              f'{pct(e2e, 50):8.2f} {pct(e2e, 99):7.2f} {max(e2e) if e2e else float("nan"):7.2f}')


if __name__ == '__main__':
    main()
