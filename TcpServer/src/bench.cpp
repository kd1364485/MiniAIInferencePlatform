// bench: TCP echo ベンチマーククライアント (M5: レイテンシ分布対応)
// 使い方: ./bench <接続数> <秒数>   例: ./bench 100 5
// N本の接続を張り、各接続で send->recv をデッドラインまで繰り返す。
// 各往復の時間(マイクロ秒)を記録し、最後に p50/p95/p99 を計算する。

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

std::atomic<long> g_completed{0};   // 全スレッド合計の往復回数
std::atomic<int>  g_served{0};      // 1往復以上できた接続の数

// 各スレッドが自分専用のvectorにレイテンシ(μs)を記録する。
// 共有しないのでロック不要（収集フェーズのオーバーヘッドを消す）
void bench_client(int idx,
                  std::vector<std::vector<long long>>& samples,
                  std::chrono::steady_clock::time_point deadline) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(8080);
    inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);

    if (connect(fd, (sockaddr*)&addr, sizeof(addr)) < 0) {
        close(fd);
        return; // 接続失敗(backlog溢れ等)は0回扱い
    }

    const char msg[] = "ping";
    char buf[256];
    bool served = false;
    auto& my_samples = samples[idx];

    while (std::chrono::steady_clock::now() < deadline) {
        auto t0 = std::chrono::steady_clock::now();          // 送る直前
        if (send(fd, msg, sizeof(msg), 0) <= 0) break;
        ssize_t r = recv(fd, buf, sizeof(buf), 0);           // 返るまで待つ
        auto t1 = std::chrono::steady_clock::now();          // 返った直後
        if (r <= 0) break;

        my_samples.push_back(
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
        ++g_completed;
        if (!served) { served = true; ++g_served; }
    }
    close(fd);
}

// sortedな配列の p パーセンタイルを取る
// p95 = 「95%のリクエストがこの時間より速かった」値
long long percentile(const std::vector<long long>& sorted, double p) {
    if (sorted.empty()) return 0;
    size_t i = (size_t)(sorted.size() * p / 100.0);
    if (i >= sorted.size()) i = sorted.size() - 1;
    return sorted[i];
}

int main(int argc, char** argv) {
    int conns = (argc > 1) ? std::atoi(argv[1]) : 100;
    int secs  = (argc > 2) ? std::atoi(argv[2]) : 5;

    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(secs);

    std::vector<std::vector<long long>> samples(conns); // スレッドごとのサンプル入れ

    std::vector<std::thread> ts;
    ts.reserve(conns);
    for (int i = 0; i < conns; ++i) {
        ts.emplace_back(bench_client, i, std::ref(samples), deadline);
    }
    for (auto& t : ts) t.join();

    // 全スレッドのサンプルを1つにマージしてソート
    std::vector<long long> all;
    for (auto& v : samples) {
        all.insert(all.end(), v.begin(), v.end());
    }
    std::sort(all.begin(), all.end());

    long total = g_completed.load();
    std::cout << "connections: " << conns
              << "  duration: " << secs << "s"
              << "  completed: " << total
              << "  served_conns: " << g_served.load() << "/" << conns
              << "  req/s: " << (double)total / secs << "\n";
    std::cout << "latency(us): p50=" << percentile(all, 50)
              << "  p95=" << percentile(all, 95)
              << "  p99=" << percentile(all, 99)
              << "  max=" << (all.empty() ? 0 : all.back()) << "\n";
}
