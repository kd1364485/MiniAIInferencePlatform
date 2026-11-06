# 閉じたループのクライアント N 本。各接続の「最初の失敗」だけを記録する(再接続しない)。
#   失敗までの時間(送信 → 失敗)で分類: 長い = 処理中に切られた / ほぼ 0 = 閉じた接続に送った
import socket, sys, threading, time
host, port, n, dur = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
req = bytes(784); res = [None]*n; lats = [[] for _ in range(n)]
def run(i):
    try: s = socket.create_connection((host, port), timeout=3.0); s.settimeout(3.0)
    except Exception as e: res[i] = ("connect", 0.0, type(e).__name__); return
    t_end = time.time() + dur
    while time.time() < t_end:
        t0 = time.perf_counter()
        try:
            s.sendall(req); got = 0
            while got < 44:
                b = s.recv(44 - got)
                if not b: raise ConnectionError("closed")
                got += len(b)
            lats[i].append(time.perf_counter() - t0)
        except Exception as e:
            res[i] = ("fail", time.perf_counter() - t0, type(e).__name__); return
    res[i] = ("alive", 0.0, "")
ths = [threading.Thread(target=run, args=(i,)) for i in range(n)]
[t.start() for t in ths]; [t.join() for t in ths]
allv = sorted(x for l in lats for x in l)
med = allv[len(allv)//2]*1000 if allv else 0
f = [r for r in res if r[0] == "fail"]
long_ = [r for r in f if r[1] > 0.002]; short = [r for r in f if r[1] <= 0.002]
print(f"conns={n} failed={len(f)} alive={sum(1 for r in res if r[0]=='alive')} median_lat={med:.1f}ms "
      f"| 処理中に切られた(>2ms)={len(long_)} 閉じた接続に送った(<=2ms)={len(short)} "
      f"| fail_lat_ms={[round(r[1]*1000,1) for r in f]}")
