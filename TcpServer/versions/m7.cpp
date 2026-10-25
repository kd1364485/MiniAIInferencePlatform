// M7: ハイブリッド = epoll(accept) + epoll(ワーカー各1個ずつ)
// メインスレッドがlisten_fdだけをepollで監視してacceptし、
// 接続fdをワーカーへラウンドロビンで振り分ける。
// 各ワーカーは「自分専用のepoll」を持ち、eventfdで起こされて新fdを登録する。

#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/socket.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <unistd.h>

void die(const char* msg) {
    perror(msg);
    std::exit(1);
}

static void set_nonblocking(int fd) {
    int flags = fcntl(fd, F_GETFL, 0);
    fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

// ワーカー1人分の状態（mutexを含むのでコピー不可 → unique_ptrで管理する）
struct Worker {
    int epoll_fd;             // このワーカー専用のepollインスタンス
    int event_fd;             // acceptor→worker への「起きろ」通知用fd
    std::mutex mtx;
    std::queue<int> pending;  // acceptorが預けた、まだepoll未登録のfd
};

// ── ワーカーの動作 ────────────────────────────────────────────
// epoll_wait が起きる契機は2種類:
//   (a) event_fd が読める   → acceptorが新接続を預けた合図
//   (b) クライアントfdが読める → データ到着(エコー) or 切断
void worker_loop(Worker& w) {
    epoll_event events[64];
    while (true) {
        int n = epoll_wait(w.epoll_fd, events, 64, -1);
        if (n <= 0) continue;

        for (int i = 0; i < n; ++i) {
            int fd = events[i].data.fd;

            // (a) 新しい接続の割り当て通知
            if (fd == w.event_fd) {
                uint64_t v;
                read(w.event_fd, &v, sizeof(v));  // カウンタを消費
                // pendingキューに溜まったfdを全部自分のepollへ登録
                while (true) {
                    int cfd = -1;
                    {
                        std::lock_guard<std::mutex> lk(w.mtx);
                        if (w.pending.empty()) break;
                        cfd = w.pending.front();
                        w.pending.pop();
                    }
                    set_nonblocking(cfd);
                    epoll_event ev{};
                    ev.events = EPOLLIN;
                    ev.data.fd = cfd;
                    epoll_ctl(w.epoll_fd, EPOLL_CTL_ADD, cfd, &ev);
                }
                continue;
            }

            // (b) クライアントのエコー処理
            char buf[1024];
            ssize_t r = recv(fd, buf, sizeof(buf), 0);
            if (r <= 0) {          // 0=切断 / <0=エラー(EAGAIN含む)
                epoll_ctl(w.epoll_fd, EPOLL_CTL_DEL, fd, nullptr);
                close(fd);
            } else {
                send(fd, buf, r, 0);
            }
        }
    }
}

// ── メインスレッド = accept専任 ───────────────────────────────
int main() {
    int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(8080);
    addr.sin_addr.s_addr = INADDR_ANY;
    if (bind(listen_fd, (sockaddr*)&addr, sizeof(addr)) < 0) die("bind");
    if (listen(listen_fd, 128) < 0) die("listen");
    set_nonblocking(listen_fd);

    // ワーカー作成: 各ワーカーに専用epollとeventfdを1つずつ
    constexpr int kWorkers = 8;
    std::vector<std::unique_ptr<Worker>> workers;  // mutexがあるのでptr経由
    std::vector<std::thread> threads;
    for (int i = 0; i < kWorkers; ++i) {
        auto w = std::make_unique<Worker>();
        w->epoll_fd = epoll_create1(0);
        w->event_fd = eventfd(0, EFD_NONBLOCK);

        // 自分のevent_fdを自分のepollへ登録（通知待ち構え）
        epoll_event ev{};
        ev.events = EPOLLIN;
        ev.data.fd = w->event_fd;
        epoll_ctl(w->epoll_fd, EPOLL_CTL_ADD, w->event_fd, &ev);

        workers.push_back(std::move(w));
        threads.emplace_back(worker_loop, std::ref(*workers[i]));
    }

    // メインのepoll: listen_fdだけを監視
    int ep = epoll_create1(0);
    epoll_event ev{};
    ev.events = EPOLLIN;
    ev.data.fd = listen_fd;
    epoll_ctl(ep, EPOLL_CTL_ADD, listen_fd, &ev);

    int next = 0;   // ラウンドロビン用の順番
    while (true) {
        epoll_event e;
        int n = epoll_wait(ep, &e, 1, -1);  // listen_fdのイベントを1個ずつ
        if (n <= 0) continue;

        // listen_fdが読める = 新着接続あり。EAGAINまで全部accept
        while (true) {
            int cfd = accept(listen_fd, nullptr, nullptr);
            if (cfd < 0) break;  // EAGAIN = これ以上新着なし

            Worker& w = *workers[next];          // ラウンドロビンで選ぶ
            next = (next + 1) % kWorkers;
            {
                std::lock_guard<std::mutex> lk(w.mtx);
                w.pending.push(cfd);              // fdをキューに預ける
            }
            uint64_t one = 1;
            write(w.event_fd, &one, sizeof(one)); // ワーカーを起こす
        }
    }
}
