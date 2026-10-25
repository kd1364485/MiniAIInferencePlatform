// Phase 2 M4: epoll echo server
// スレッドを捨てて「カーネルにfdを監視させる」方式。
// 1スレッドがイベントループで全接続を捌く。

#include <cerrno>         // errno, EAGAIN
#include <cstring>
#include <iostream>
#include <fcntl.h>        // fcntl, O_NONBLOCK
#include <sys/epoll.h>    // epoll_create1, epoll_ctl, epoll_wait
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>

// fdをノンブロッキング化する。
// 以後このfdへの recv/send/accept は「今すぐできなければ
// -1を返して errno=EAGAIN」になる。待たない。
void set_nonblocking(int fd) {
    int flags = fcntl(fd, F_GETFL, 0);
    fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

int main() {
    int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd < 0) {
        perror("socket");
        return 1;
    }

    int opt = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(8080);

    if (bind(listen_fd, (sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind");
        return 1;
    }
    if (listen(listen_fd, 5) < 0) {
        perror("listen");
        return 1;
    }

    // ===== M4: epollの準備 =====

    set_nonblocking(listen_fd);

    // epoll_fd = 「fdの監視リスト」を持つ特別なfd(昨日の質問の答え)。
    // これ自体もfdなのでfdテーブルに乗る。
    int epfd = epoll_create1(0);
    if (epfd < 0) {
        perror("epoll_create1");
        return 1;
    }

    // listen_fdを監視対象に登録。EPOLLIN=「読めるデータ/新接続が来たら教えて」
    epoll_event ev{};
    ev.events = EPOLLIN;
    ev.data.fd = listen_fd;             // 通知に同梱される識別子
    epoll_ctl(epfd, EPOLL_CTL_ADD, listen_fd, &ev);

    std::cout << "listening on port 8080 (epoll, single thread)\n";

    // ===== イベントループ =====
    // epoll_waitは「登録したfdのどれかに用事ができるまで寝る」。
    // 起きたら「用事があったfdのリスト」が返るので全部処理する。
    epoll_event events[64];
    while (true) {
        int n = epoll_wait(epfd, events, 64, -1); // -1=無期限に待つ
        if (n < 0) {
            perror("epoll_wait");
            break;
        }

        for (int i = 0; i < n; ++i) {
            int fd = events[i].data.fd;

            if (fd == listen_fd) {
                // --- 玄関に新しい接続が届いた ---
                // 複数溜まっている可能性があるので、EAGAINになるまで全部accept
                while (true) {
                    int client = accept(listen_fd, nullptr, nullptr);
                    if (client < 0) {
                        if (errno == EAGAIN || errno == EWOULDBLOCK) break;
                        perror("accept");
                        break;
                    }
                    set_nonblocking(client);

                    // 新しいクライアントfdも監視リストに登録
                    epoll_event cev{};
                    cev.events = EPOLLIN;
                    cev.data.fd = client;
                    epoll_ctl(epfd, EPOLL_CTL_ADD, client, &cev);
                    std::cout << "client connected: fd=" << client << "\n";
                }
            } else {
                // --- 既存の接続にデータが届いた ---
                char buf[4096];
                ssize_t r = recv(fd, buf, sizeof(buf), 0);
                if (r > 0) {
                    send(fd, buf, r, 0); // echo
                } else if (r == 0) {
                    // 相手が切断
                    std::cout << "client disconnected: fd=" << fd << "\n";
                    close(fd); // closeすればepollの監視リストからも自動で外れる
                } else {
                    // r<0: EAGAINなら「今はデータ無し」なので無視、それ以外はエラー
                    if (errno != EAGAIN && errno != EWOULDBLOCK) {
                        perror("recv");
                        close(fd);
                    }
                }
            }
        }
    }
}
