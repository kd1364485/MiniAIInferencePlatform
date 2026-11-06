# k8sfail.py の再送対応版。retry=1 のとき、失敗したリクエストを「新しい接続で 1 回だけ」やり直す。
#   retried = 1 回目が失敗した回数(再送した回数)。errors = 再送しても失敗した回数(ユーザーから見た失敗)
import socket, sys, threading, time
host, port, nclients, dur, retry = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
t0 = time.time(); errs = []; retried = [0]*nclients; ok = [0]*nclients; lock = threading.Lock()
req = bytes(784)
def once(s):
    s.sendall(req); got = 0
    while got < 44:
        b = s.recv(44 - got)
        if not b: raise ConnectionError("closed")
        got += len(b)
def conn():
    s = socket.create_connection((host, port), timeout=1.0); s.settimeout(1.0); return s
def run(i):
    s = None
    while time.time() - t0 < dur:
        for attempt in range(2 if retry else 1):
            try:
                if s is None: s = conn()
                once(s); ok[i] += 1; break
            except Exception as e:
                try: s and s.close()
                except Exception: pass
                s = None
                if attempt == 0 and retry: retried[i] += 1; continue
                with lock: errs.append((time.time(), i, type(e).__name__))
                time.sleep(0.05)
        time.sleep(0.005)
ths = [threading.Thread(target=run, args=(i,)) for i in range(nclients)]
[t.start() for t in ths]; [t.join() for t in ths]
print(f"ok={sum(ok)} retried={sum(retried)} errors={len(errs)}")
for t, i, n in errs[:20]: print(f"  {time.strftime('%H:%M:%S', time.localtime(t))}.{int((t%1)*1000):03d} client={i} {n}")
