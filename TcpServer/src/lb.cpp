// Phase 9 M1: 自作のロードバランサ(L4 の TCP プロキシ)
//
// 使い方:  lb <listen_port> <backend>[,<backend>...] [threads=2] [metrics_port=9101]
//   backend = host:port:health_port     例:  worker:8080:9100   または  127.0.0.1:18081:19101
//   host は IP でも DNS 名でもよい。DNS 名は 2 秒ごとに引き直し、複数の IP に引ければ全部をバックエンドにする
//   (docker compose の --scale や、Kubernetes の Service のように、数が増減する相手に追従するため)
//
// 何をするか:
//   クライアントの接続を受けて、健康なバックエンドの中から「接続数が最少」のものを選び、そこへ中継する。
//   中身(784B / 44B の独自プロトコル)は見ない = L4。接続単位の振り分けなので、同じ接続の間は同じバックエンド。
//
// 設計:
//   ・proxy スレッド x N: 各スレッドが自分専用の epoll と、SO_REUSEPORT の listen ソケットを持つ。
//     同じポートに複数の listen ソケットを作ると、カーネルが accept を振り分けてくれる(M7 の eventfd の受け渡しが要らない)
//   ・ヘルスチェックのスレッド x 1: 1 秒ごとに各バックエンドの /healthz を見る。2 回連続の失敗で外し、2 回連続の成功で戻す
//   ・メトリクスのスレッド x 1: metrics.h の HTTP サーバーを再利用(/metrics, /healthz)
//
// 1 つの接続を 2 つの fd(クライアント側・バックエンド側)で見る。片方を閉じて解放した直後に、同じ epoll_wait の結果の中に
// もう片方の通知が残っていると、解放済みのメモリを触ってしまう(Phase 6 の「erase 後の参照」と同じ種類の事故)。
// → 閉じたら dead の印を付けて fd だけ片付け、メモリの解放は「その回の処理が全部終わってから」まとめて行う。

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <ctime>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>
#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <unistd.h>
#include "metrics.h"

// ---------------- バックエンドの状態 ----------------
struct Backend {
    std::string name;            // "ip:port"(ログとメトリクスの表示用)
    int spec_id = 0;             // どの backend 指定(host:port:health)から来たか
    sockaddr_in addr{};          // データ用の宛先
    uint16_t health_port = 0;
    std::atomic<bool> healthy{false};          // 新しく見つけたものは、ヘルスチェックに通るまで使わない
    std::atomic<int64_t> active{0};            // いま中継中の接続数(最少接続の判断に使う)
    std::atomic<uint64_t> total{0};            // 中継を始めた接続の累計
    std::atomic<uint64_t> connect_fail{0};     // データ用の接続に失敗した回数
    int ok_streak = 0, fail_streak = 0;        // ヘルスチェックのスレッドだけが触る
};
using BackendList = std::vector<std::shared_ptr<Backend>>;

// 現在のバックエンドの一覧。更新はヘルスチェックのスレッド、読むのは proxy スレッド。
// 一覧を丸ごと差し替える(shared_ptr の atomic)ので、読む側はロック不要で、古い一覧を持ったままでも安全。
std::atomic<std::shared_ptr<const BackendList>> g_list;

std::atomic<bool> g_stop{false};
std::atomic<uint64_t> g_accepted{0}, g_no_backend{0};
std::atomic<uint64_t> g_bytes_c2b{0}, g_bytes_b2c{0};

struct Spec { std::string host; uint16_t port, health_port; };

// 時刻(時:分:秒.ミリ秒)つきで標準エラーに出す。障害の前後関係を、後から追えるようにするため
static void logf(const std::string& s) {
    auto now = std::chrono::system_clock::now();
    std::time_t tt = std::chrono::system_clock::to_time_t(now);
    int ms = (int)(std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000);
    char ts[32];
    std::strftime(ts, sizeof(ts), "%H:%M:%S", std::localtime(&tt));
    char line[48];
    std::snprintf(line, sizeof(line), "[%s.%03d] ", ts, ms);
    std::cerr << line << s << std::endl;
}

// 健康でないと判断して外す。どの経路(ヘルスチェック/データ用の接続の失敗)で外したかを、ログに残す。
// exchange は「直前が true だったときだけ」ログを出すため(外れている間に何度失敗しても、1 回しか出さない)
static void mark_down(Backend& b, const char* reason) {
    if (b.healthy.exchange(false)) logf("backend 切り離し: " + b.name + "(" + reason + ")");
}

// ---------------- ヘルスチェック ----------------
// バックエンドの health_port に TCP で接続し、"GET /healthz" に 200 が返るかを見る。500ms で打ち切る。
static bool check_health(const Backend& b) {
    int fd = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK, 0);
    if (fd < 0) return false;
    sockaddr_in a = b.addr;
    a.sin_port = htons(b.health_port);
    int r = connect(fd, (sockaddr*)&a, sizeof(a));
    bool ok = false;
    if (r == 0 || errno == EINPROGRESS) {
        pollfd p{fd, POLLOUT, 0};
        int err = 0; socklen_t len = sizeof(err);
        if (poll(&p, 1, 500) > 0 && getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &len) == 0 && err == 0) {
            const char req[] = "GET /healthz HTTP/1.0\r\n\r\n";
            if (send(fd, req, sizeof(req) - 1, MSG_NOSIGNAL) > 0) {
                p.events = POLLIN;
                char buf[64] = {};
                if (poll(&p, 1, 500) > 0 && recv(fd, buf, sizeof(buf) - 1, 0) > 0)
                    ok = std::strstr(buf, " 200 ") != nullptr;
            }
        }
    }
    close(fd);
    return ok;
}

static std::vector<in_addr> resolve_ipv4(const std::string& host, bool& ok) {
    std::vector<in_addr> out;
    addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    int rc = getaddrinfo(host.c_str(), nullptr, &hints, &res);
    ok = (rc == 0);
    if (ok) {
        for (addrinfo* p = res; p; p = p->ai_next) out.push_back(((sockaddr_in*)p->ai_addr)->sin_addr);
        freeaddrinfo(res);
    }
    return out;
}

// DNS を引き直して、バックエンドの一覧を更新する。既にある IP は、同じ Backend を使い回す(状態・接続数を保つ)。
// 引き直しに失敗した backend 指定は、今の一覧をそのまま残す(DNS の一時的な失敗で、全部を外さない)。
static void refresh_backends(const std::vector<Spec>& specs) {
    auto cur = g_list.load();
    std::map<std::string, std::shared_ptr<Backend>> by_key;
    for (auto& b : *cur) by_key[b->name + "#" + std::to_string(b->spec_id)] = b;

    BackendList next;
    for (size_t i = 0; i < specs.size(); ++i) {
        bool ok = false;
        auto ips = resolve_ipv4(specs[i].host, ok);
        if (!ok) {   // 失敗: この指定から来ている既存のバックエンドを残す
            for (auto& b : *cur) if (b->spec_id == (int)i) next.push_back(b);
            continue;
        }
        for (auto& ip : ips) {
            char s[INET_ADDRSTRLEN];
            inet_ntop(AF_INET, &ip, s, sizeof(s));
            std::string name = std::string(s) + ":" + std::to_string(specs[i].port);
            auto it = by_key.find(name + "#" + std::to_string(i));
            if (it != by_key.end()) { next.push_back(it->second); continue; }
            auto b = std::make_shared<Backend>();
            b->name = name; b->spec_id = (int)i;
            b->addr.sin_family = AF_INET; b->addr.sin_addr = ip; b->addr.sin_port = htons(specs[i].port);
            b->health_port = specs[i].health_port;
            next.push_back(b);
            logf("backend 追加: " + name + "(ヘルスチェックに通るまで使わない)");
        }
    }
    for (auto& b : *cur) {   // 一覧から消えたもの(新しい接続は来なくなる。中継中の接続は続く)
        bool still = std::any_of(next.begin(), next.end(), [&](auto& n) { return n.get() == b.get(); });
        if (!still) logf("backend 削除: " + b->name);
    }
    g_list.store(std::make_shared<const BackendList>(std::move(next)));
}

static void health_loop(const std::vector<Spec>& specs) {
    int tick = 0;
    while (!g_stop) {
        if (tick++ % 2 == 0) refresh_backends(specs);   // 2 秒ごと
        auto list = g_list.load();
        for (auto& b : *list) {
            bool ok = check_health(*b);
            if (ok) { b->ok_streak++; b->fail_streak = 0; }
            else    { b->fail_streak++; b->ok_streak = 0; }
            bool h = b->healthy.load();
            if (!h && b->ok_streak >= 2)   { b->healthy = true;  logf("backend 復帰: " + b->name); }
            if (h && b->fail_streak >= 2)  mark_down(*b, "ヘルスチェックが2回連続で失敗");
        }
        for (int i = 0; i < 10 && !g_stop; ++i) usleep(100 * 1000);   // 1 秒(100ms ごとに停止の合図を見る)
    }
}

// ---------------- 中継(proxy) ----------------
struct Conn;
struct Endpoint { Conn* conn; bool is_client; };

struct Conn {
    int cfd = -1, bfd = -1;
    std::shared_ptr<Backend> be;            // 中継先(接続中は参照を持つ。一覧から消えても解放されない)
    bool established = false;               // バックエンドへの接続が済んだか
    bool dead = false;                      // 閉じた(fd は片付け済み。メモリは回の終わりに解放する)
    int attempts = 0;
    std::vector<Backend*> tried;            // 接続に失敗したバックエンド(同じ接続で二度選ばない)
    std::string c2b, b2c;                   // 相手が受け取れず、書ききれなかった分
    Endpoint ec{this, true}, eb{this, false};
};

struct ProxyThread {
    int ep = -1, lfd = -1;
    std::unordered_set<Conn*> conns;
    std::vector<Conn*> graveyard;           // 閉じた Conn。この回の epoll_wait の処理が終わってから delete する
    uint32_t rr = 0;                        // 同数のときの偏りを避けるための回転
};

static void set_nb(int fd) { fcntl(fd, F_SETFL, fcntl(fd, F_GETFL, 0) | O_NONBLOCK); }

static void close_conn(ProxyThread& t, Conn* c) {
    if (c->dead) return;
    c->dead = true;
    if (c->cfd >= 0) { epoll_ctl(t.ep, EPOLL_CTL_DEL, c->cfd, nullptr); close(c->cfd); c->cfd = -1; }
    if (c->bfd >= 0) { epoll_ctl(t.ep, EPOLL_CTL_DEL, c->bfd, nullptr); close(c->bfd); c->bfd = -1; }
    if (c->be) { --c->be->active; c->be.reset(); }
    t.conns.erase(c);
    t.graveyard.push_back(c);               // ここでは delete しない
}

// 健康なバックエンドから、接続数が最少のものを選ぶ(tried に入っているものは除く)。同数なら順番に回す
static std::shared_ptr<Backend> pick_backend(ProxyThread& t, const std::vector<Backend*>& tried) {
    auto list = g_list.load();
    std::shared_ptr<Backend> best;
    size_t n = list->size();
    for (size_t k = 0; k < n; ++k) {
        auto& b = (*list)[(t.rr + k) % n];
        if (!b->healthy) continue;
        if (std::find(tried.begin(), tried.end(), b.get()) != tried.end()) continue;
        if (!best || b->active < best->active) best = b;
    }
    ++t.rr;
    return best;
}

// バックエンドへ非同期に接続を始める。完了(または失敗)は epoll の EPOLLOUT で知る。
// 同期の connect にしないのは、バックエンドが落ちていると SYN の再送で長く止まり、他の全接続を巻き込むから。
static void start_connect(ProxyThread& t, Conn* c) {
    while (c->attempts < 3) {
        auto be = pick_backend(t, c->tried);
        if (!be) break;
        ++c->attempts;
        int fd = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK, 0);
        int r = fd < 0 ? -1 : connect(fd, (sockaddr*)&be->addr, sizeof(be->addr));
        if (fd >= 0 && (r == 0 || errno == EINPROGRESS)) {
            int one = 1; setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
            c->bfd = fd; c->be = be; ++be->active;
            epoll_event ev{}; ev.events = EPOLLOUT; ev.data.ptr = &c->eb;
            epoll_ctl(t.ep, EPOLL_CTL_ADD, fd, &ev);
            return;
        }
        if (fd >= 0) close(fd);
        ++be->connect_fail; mark_down(*be, "データ用の接続が即座に失敗"); c->tried.push_back(be.get());
    }
    ++g_no_backend;                         // 使えるバックエンドが無い → クライアントの接続を閉じる
    close_conn(t, c);
}

static void arm_out(ProxyThread& t, Conn* c, bool to_backend, bool on) {
    int fd = to_backend ? c->bfd : c->cfd;
    epoll_event ev{}; ev.events = EPOLLIN | (on ? (uint32_t)EPOLLOUT : 0u);
    ev.data.ptr = to_backend ? &c->eb : &c->ec;
    epoll_ctl(t.ep, EPOLL_CTL_MOD, fd, &ev);
}

// src から読めるだけ読んで dst へ書く。書ききれなかった分は pending に残し、EPOLLOUT で続きを書く。
// false を返したら、その接続は閉じる(相手が閉じた/エラー/溜まりすぎ)。
static bool pump(ProxyThread& t, Conn* c, bool from_client) {
    int src = from_client ? c->cfd : c->bfd, dst = from_client ? c->bfd : c->cfd;
    std::string& pend = from_client ? c->c2b : c->b2c;
    char buf[16384];
    while (true) {
        ssize_t r = recv(src, buf, sizeof(buf), 0);
        if (r == 0) return false;                                           // 相手が close
        if (r < 0) { if (errno == EAGAIN || errno == EWOULDBLOCK) return true; if (errno == EINTR) continue; return false; }
        (from_client ? g_bytes_c2b : g_bytes_b2c) += (uint64_t)r;
        size_t off = 0;
        if (pend.empty()) {                                                 // 順序を守る: 溜まっている分があれば、その後ろに足す
            while (off < (size_t)r) {
                ssize_t w = send(dst, buf + off, r - off, MSG_NOSIGNAL);
                if (w > 0) off += (size_t)w;
                else if (w < 0 && errno == EINTR) continue;
                else if (w < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) break;
                else return false;
            }
        }
        if (off < (size_t)r) {
            pend.append(buf + off, r - off);
            if (pend.size() > (1u << 20)) return false;                      // 相手が読まないまま 1MB 溜まった → 諦める
            arm_out(t, c, from_client, true);                                // dst が書けるようになったら起こしてもらう
        }
        if ((size_t)r < sizeof(buf)) return true;                           // 読み切った
    }
}

static bool flush_pending(ProxyThread& t, Conn* c, bool to_backend) {
    int dst = to_backend ? c->bfd : c->cfd;
    std::string& pend = to_backend ? c->c2b : c->b2c;
    while (!pend.empty()) {
        ssize_t w = send(dst, pend.data(), pend.size(), MSG_NOSIGNAL);
        if (w > 0) pend.erase(0, (size_t)w);
        else if (w < 0 && errno == EINTR) continue;
        else if (w < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) return true;   // まだ書けない。EPOLLOUT を待つ
        else return false;
    }
    arm_out(t, c, to_backend, false);                                       // 全部書けた → EPOLLOUT の監視をやめる
    return true;
}

static void proxy_loop(int listen_port) {
    ProxyThread t;
    t.lfd = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    setsockopt(t.lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    setsockopt(t.lfd, SOL_SOCKET, SO_REUSEPORT, &one, sizeof(one));         // 同じポートに複数の listen ソケット → カーネルが振り分ける
    sockaddr_in a{}; a.sin_family = AF_INET; a.sin_port = htons(listen_port); a.sin_addr.s_addr = INADDR_ANY;
    if (bind(t.lfd, (sockaddr*)&a, sizeof(a)) < 0 || listen(t.lfd, 1024) < 0) { perror("lb: bind/listen"); std::exit(1); }
    set_nb(t.lfd);
    t.ep = epoll_create1(0);
    epoll_event lev{}; lev.events = EPOLLIN; lev.data.ptr = nullptr;       // nullptr の印 = listen ソケット
    epoll_ctl(t.ep, EPOLL_CTL_ADD, t.lfd, &lev);

    epoll_event events[128];
    while (!g_stop) {
        int n = epoll_wait(t.ep, events, 128, 200);                         // 200ms ごとに起きて、切り離されたバックエンドの接続を掃除する
        for (int i = 0; i < n; ++i) {
            if (events[i].data.ptr == nullptr) {                            // 新しい接続
                while (true) {
                    int cfd = accept4(t.lfd, nullptr, nullptr, SOCK_NONBLOCK);
                    if (cfd < 0) break;
                    ++g_accepted;
                    setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
                    Conn* c = new Conn; c->cfd = cfd; t.conns.insert(c);
                    start_connect(t, c);
                }
                continue;
            }
            auto* e = (Endpoint*)events[i].data.ptr;
            Conn* c = e->conn;
            if (c->dead) continue;                                          // この回のうちに閉じた接続の、残りの通知は無視する
            uint32_t ev = events[i].events;

            if (!c->established) {                                          // バックエンドへの接続の完了待ち
                if (e->is_client) continue;
                int err = 0; socklen_t len = sizeof(err);
                getsockopt(c->bfd, SOL_SOCKET, SO_ERROR, &err, &len);
                if (err != 0 || (ev & (EPOLLERR | EPOLLHUP))) {             // 接続失敗 → 別のバックエンドで、やり直す
                    ++c->be->connect_fail; mark_down(*c->be, "データ用の接続に失敗"); c->tried.push_back(c->be.get());
                    epoll_ctl(t.ep, EPOLL_CTL_DEL, c->bfd, nullptr); close(c->bfd); c->bfd = -1;
                    --c->be->active; c->be.reset();
                    start_connect(t, c);
                    continue;
                }
                c->established = true; ++c->be->total;
                epoll_event bev{}; bev.events = EPOLLIN; bev.data.ptr = &c->eb;
                epoll_ctl(t.ep, EPOLL_CTL_MOD, c->bfd, &bev);
                epoll_event cev{}; cev.events = EPOLLIN; cev.data.ptr = &c->ec;
                epoll_ctl(t.ep, EPOLL_CTL_ADD, c->cfd, &cev);                // クライアントの読み取りは、中継先が決まってから始める
                continue;
            }

            bool ok = true;
            if (ev & (EPOLLERR | EPOLLHUP)) ok = false;
            if (ok && (ev & EPOLLIN)) ok = pump(t, c, e->is_client);
            if (ok && !c->dead && (ev & EPOLLOUT)) ok = flush_pending(t, c, !e->is_client);
            if (!ok) close_conn(t, c);
        }

        // 切り離された(ヘルスチェックに落ちた)バックエンドへの接続は、こちらから閉じる。
        // 相手が落ちて TCP の切断も来ない(固まった)場合でも、クライアントが再接続して、別のバックエンドへ行けるように。
        for (auto it = t.conns.begin(); it != t.conns.end();) {
            Conn* c = *it; ++it;
            if (c->established && c->be && !c->be->healthy) close_conn(t, c);
        }
        for (Conn* c : t.graveyard) delete c;                               // ここで初めて解放する
        t.graveyard.clear();
    }
}

// ---------------- メトリクス ----------------
static std::string render_metrics() {
    std::ostringstream o;
    using metrics::header;
    auto list = g_list.load();
    int64_t healthy = 0;
    for (auto& b : *list) healthy += b->healthy ? 1 : 0;
    header(o, "lb_backends_healthy", "gauge", "Backends currently passing health checks.");
    o << "lb_backends_healthy " << healthy << '\n';
    header(o, "lb_backend_up", "gauge", "1 if the backend passes health checks.");
    for (auto& b : *list) o << "lb_backend_up{backend=\"" << b->name << "\"} " << (b->healthy ? 1 : 0) << '\n';
    header(o, "lb_backend_connections", "gauge", "Connections currently proxied to the backend.");
    for (auto& b : *list) o << "lb_backend_connections{backend=\"" << b->name << "\"} " << b->active.load() << '\n';
    header(o, "lb_backend_connections_total", "counter", "Connections proxied to the backend since start.");
    for (auto& b : *list) o << "lb_backend_connections_total{backend=\"" << b->name << "\"} " << b->total.load() << '\n';
    header(o, "lb_backend_connect_failures_total", "counter", "Failed attempts to connect to the backend.");
    for (auto& b : *list) o << "lb_backend_connect_failures_total{backend=\"" << b->name << "\"} " << b->connect_fail.load() << '\n';
    header(o, "lb_client_connections_accepted_total", "counter", "Client connections accepted.");
    o << "lb_client_connections_accepted_total " << g_accepted.load() << '\n';
    header(o, "lb_no_backend_total", "counter", "Client connections closed because no healthy backend was available.");
    o << "lb_no_backend_total " << g_no_backend.load() << '\n';
    header(o, "lb_bytes_total", "counter", "Bytes relayed.");
    o << "lb_bytes_total{direction=\"client_to_backend\"} " << g_bytes_c2b.load() << '\n'
      << "lb_bytes_total{direction=\"backend_to_client\"} " << g_bytes_b2c.load() << '\n';
    return o.str();
}

static std::vector<Spec> parse_specs(const std::string& arg) {
    std::vector<Spec> out;
    std::stringstream ss(arg);
    std::string item;
    while (std::getline(ss, item, ',')) {
        std::vector<std::string> p;
        std::stringstream is(item);
        std::string f;
        while (std::getline(is, f, ':')) p.push_back(f);
        if (p.size() != 3) { std::cerr << "backend の形式は host:port:health_port です: " << item << "\n"; std::exit(1); }
        out.push_back({p[0], (uint16_t)std::atoi(p[1].c_str()), (uint16_t)std::atoi(p[2].c_str())});
    }
    return out;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: " << argv[0] << " <listen_port> <host:port:health_port>[,...] [threads=2] [metrics_port=9101]\n";
        return 1;
    }
    int listen_port = std::atoi(argv[1]);
    auto specs = parse_specs(argv[2]);
    int nthreads = argc > 3 ? std::max(1, std::atoi(argv[3])) : 2;
    int metrics_port = argc > 4 ? std::atoi(argv[4]) : 9101;

    signal(SIGPIPE, SIG_IGN);
    signal(SIGINT,  [](int) { g_stop = true; });
    signal(SIGTERM, [](int) { g_stop = true; });

    g_list.store(std::make_shared<const BackendList>());
    refresh_backends(specs);                                                 // 最初の一覧(まだ誰も健康ではない)

    std::vector<std::thread> threads;
    threads.emplace_back(health_loop, std::cref(specs));
    for (int i = 0; i < nthreads; ++i) threads.emplace_back(proxy_loop, listen_port);
    if (metrics_port > 0)
        threads.emplace_back(metrics::serve, metrics_port, [] { return render_metrics(); }, std::cref(g_stop));

    std::cout << "lb: port " << listen_port << ", threads=" << nthreads << ", backends=" << argv[2]
              << ", metrics=" << (metrics_port > 0 ? ":" + std::to_string(metrics_port) : std::string("off")) << std::endl;
    while (!g_stop) usleep(200 * 1000);
    _exit(0);        // スレッドは epoll_wait などで待っているので join せず終了する(infer_server と同じ)
}
