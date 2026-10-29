// Phase 8 M3: メトリクスの出力口。Prometheus が読む「テキスト形式」(exposition format 0.0.4)の最小実装。
//
// 部品は3つ:
//   Histogram<N> : 値を「上限(le)ごとの区間」に数える。p99 などを後から計算できる(平均だけでは分からない)
//   header()/escape_label() : "# HELP" / "# TYPE" 行と、ラベルの値のエスケープ
//   serve()      : 別スレッドで動く小さな HTTP サーバー。GET /metrics と GET /healthz だけに答える
//
// 設計の要点:
//   - 推論の経路には触れない。数える側は atomic の加算だけ(ロックしない)。読む側(serve)は別スレッド。
//   - 外部ライブラリは使わない(HTTP は GET の1行目だけ読めば足りる)。
//   - /metrics は認証なし。運用情報(接続数・エラー数・モデル名)が見えるので、外部に公開するポートでは注意。
#pragma once

#include <poll.h>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

namespace metrics {

// "# HELP name help" と "# TYPE name type"。type は counter / gauge / histogram
inline void header(std::ostringstream& o, const char* name, const char* type, const char* help) {
    o << "# HELP " << name << ' ' << help << "\n# TYPE " << name << ' ' << type << '\n';
}

// ラベルの値に " や \ や改行が入ると形式が壊れる(モデル名などは外から来る)ので、エスケープする
inline std::string escape_label(const std::string& s) {
    std::string r;
    for (char c : s) {
        if (c == '\\') r += "\\\\";
        else if (c == '"') r += "\\\"";
        else if (c == '\n') r += "\\n";
        else r += c;
    }
    return r;
}

// ヒストグラム。bounds は小さい順の「上限」。観測値 v は「v <= 上限」を満たす最初の区間に数える。
// counts_ の最後(添字 N)は「どの上限よりも大きい」区間(+Inf)。
// 区間ごとに数えておき、出力のときに累積(「この上限以下は全部で何件」)に直す。
// 数えるのは atomic の加算だけなので、複数のスレッドが同時に observe してもロック不要。
template <size_t N>
class Histogram {
public:
    explicit Histogram(const std::array<double, N>& bounds) : bounds_(bounds) {}

    void observe(double v) {
        size_t i = 0;
        while (i < N && v > bounds_[i]) ++i;
        counts_[i].fetch_add(1, std::memory_order_relaxed);
        sum_.fetch_add(v, std::memory_order_relaxed);        // C++20 の atomic<double>::fetch_add
    }

    void render(std::ostringstream& o, const char* name, const char* help) const {
        header(o, name, "histogram", help);
        uint64_t cumulative = 0;
        for (size_t i = 0; i < N; ++i) {
            cumulative += counts_[i].load(std::memory_order_relaxed);
            o << name << "_bucket{le=\"" << bounds_[i] << "\"} " << cumulative << '\n';
        }
        cumulative += counts_[N].load(std::memory_order_relaxed);
        o << name << "_bucket{le=\"+Inf\"} " << cumulative << '\n';
        o << name << "_sum " << sum_.load(std::memory_order_relaxed) << '\n';
        o << name << "_count " << cumulative << '\n';
    }

private:
    std::array<double, N> bounds_;
    std::array<std::atomic<uint64_t>, N + 1> counts_{};      // {} で全部 0 から始まる
    std::atomic<double> sum_{0.0};
};

// HTTP の応答を最後まで送る(send は一部だけ送って戻ることがあるので、残りを繰り返す)
inline void send_all(int fd, const std::string& data) {
    size_t off = 0;
    while (off < data.size()) {
        ssize_t n = send(fd, data.data() + off, data.size() - off, MSG_NOSIGNAL);
        if (n <= 0) return;                                  // 相手が切れた / 送信タイムアウト → 諦める
        off += (size_t)n;
    }
}

inline void reply(int fd, int code, const char* status, const char* content_type, const std::string& body) {
    std::ostringstream o;
    o << "HTTP/1.1 " << code << ' ' << status << "\r\n"
      << "Content-Type: " << content_type << "\r\n"
      << "Content-Length: " << body.size() << "\r\n"
      << "Connection: close\r\n\r\n"
      << body;
    send_all(fd, o.str());
}

// 別スレッドで動く HTTP サーバー(1接続ずつ順に処理する。Prometheus の取得は数秒に1回で足りる)。
//   render_body : GET /metrics のたびに呼ばれ、メトリクスの本文を作る
//   stop        : true になったら止まる(accept を 500ms ごとに起きて確認する)
//   healthy     : GET /healthz の答え。false を返すと 503(終了準備中 = 振り分けから外してほしい)。
//                 省略(空)なら、常に 200。/metrics は、終了準備中も答え続ける(最後の様子を取れるように)
// bind に失敗してもサーバー全体は止めない(推論が本体で、メトリクスはおまけ)。
inline void serve(int port, std::function<std::string()> render_body, const std::atomic<bool>& stop,
                  std::function<bool()> healthy = nullptr) {
    int lfd = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons(port);
    a.sin_addr.s_addr = INADDR_ANY;                          // コンテナの外(Prometheus)から届くよう全インターフェース
    if (bind(lfd, (sockaddr*)&a, sizeof(a)) < 0 || listen(lfd, 16) < 0) {
        perror("metrics: bind/listen");
        close(lfd);
        return;
    }
    while (!stop) {
        pollfd p{lfd, POLLIN, 0};
        if (poll(&p, 1, 500) <= 0) continue;
        int cfd = accept(lfd, nullptr, nullptr);
        if (cfd < 0) continue;

        // 遅い・悪意のあるクライアントで、メトリクスの取得が止まらないようにする:
        // 1回の recv/send に 2 秒のタイムアウトを付け、さらに全体で 3 秒を超えたら打ち切る
        timeval tv{2, 0};
        setsockopt(cfd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        setsockopt(cfd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);

        // リクエストの先頭(ヘッダの終わり "\r\n\r\n" まで)を読む。1行目の「GET /パス」しか使わない
        char buf[2048];
        size_t used = 0;
        while (used < sizeof(buf) - 1 && std::chrono::steady_clock::now() < deadline) {
            ssize_t n = recv(cfd, buf + used, sizeof(buf) - 1 - used, 0);
            if (n <= 0) break;
            used += (size_t)n;
            buf[used] = '\0';
            if (std::strstr(buf, "\r\n\r\n")) break;
        }
        std::string req(buf, used);
        std::string path;
        bool is_get = req.rfind("GET ", 0) == 0;
        if (is_get) {
            size_t sp = req.find(' ', 4);
            if (sp != std::string::npos) path = req.substr(4, sp - 4);
            path = path.substr(0, path.find('?'));           // ?以降(クエリ)は使わない
        }

        if (!is_get)                 reply(cfd, 405, "Method Not Allowed", "text/plain", "only GET is supported\n");
        else if (path == "/metrics") reply(cfd, 200, "OK", "text/plain; version=0.0.4; charset=utf-8", render_body());
        else if (path == "/healthz") {
            if (!healthy || healthy()) reply(cfd, 200, "OK", "text/plain", "ok\n");
            else                       reply(cfd, 503, "Service Unavailable", "text/plain", "draining\n");
        }
        else                         reply(cfd, 404, "Not Found", "text/plain", "not found\n");
        close(cfd);
    }
    close(lfd);
}

}  // namespace metrics
