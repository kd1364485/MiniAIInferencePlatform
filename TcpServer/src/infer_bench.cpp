// infer_bench: 推論サーバー用ベンチ + 正解率チェック
// 使い方: ./infer_bench <接続数> <秒数> [mnist_dir]
// 各接続が MNIST テスト画像を順に送り、返ってきた予測をラベルと照合する。
// 「速い」だけでなく「正しい答えが返っている」ことを同時に確認する(壊れた高速化の検出)。
//
// 構造は bench.cpp(Phase 2)と同じ: 1接続 = 1スレッド、同期的に send → recv を繰り返す。
// 違いは (1) 送るのが固定の "ping" でなく MNIST 画像 (2) 返事の中身(予測)を検査する点。

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

// サーバー(infer_server.cpp)と必ず一致させる。ずれると通信が壊れる
constexpr size_t kReqBytes  = 784;   // 1リクエスト = 画像1枚
constexpr size_t kRespBytes = 44;    // 1レスポンス = int32 pred + float32 x 10

// 全スレッド合計のカウンタ。++ だけなので atomic で足りる
std::atomic<long> g_completed{0};   // 往復が完了した回数
std::atomic<long> g_correct{0};     // そのうち予測がラベルと一致した回数
std::atomic<int>  g_served{0};      // 1往復以上できた接続の数(捌かれず飢えた接続の検出用)

// MNISTのIDXファイルのヘッダはビッグエンディアン(上位バイトが先)。
// x86はリトルエンディアンなので、1バイトずつ読んで自分で組み立てる(OnnxLearn/infer.cpp と同じ)
static uint32_t be32(std::ifstream& f) {
    uint8_t b[4];
    f.read((char*)b, 4);
    return (b[0] << 24) | (b[1] << 16) | (b[2] << 8) | b[3];
}

// TCPはストリームなので n バイト揃うまで繰り返し recv する
// (サーバーの Conn.filled と同じ考え方。サーバーは非同期だから状態に持ち、こちらは同期なのでループで済む)
static bool recv_all(int fd, uint8_t* p, size_t n) {
    size_t got = 0;
    while (got < n) {
        ssize_t r = recv(fd, p + got, n - got, 0);   // 前回の続きから、残りの分だけ
        if (r <= 0) return false;                    // 0=サーバーが閉じた / <0=エラー
        got += r;
    }
    return true;
}

// 1スレッド = 1接続。deadline まで「画像を送る → 結果を受ける → 照合 → 時間記録」を繰り返す
//   imgs/lbls : 全テスト画像とラベル(読み取り専用で全スレッド共有。書かないのでロック不要)
//   samples   : samples[idx] が自分専用の記録箱(他スレッドと共有しない → ロック不要)
void bench_client(int idx, const std::vector<uint8_t>& imgs, const std::vector<uint8_t>& lbls,
                  std::vector<std::vector<long long>>& samples,
                  std::chrono::steady_clock::time_point deadline) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(8080);
    inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);   // 文字列のIP → バイナリ
    if (connect(fd, (sockaddr*)&addr, sizeof(addr)) < 0) { close(fd); return; }
    // サーバー側と同じ。小さい送信をカーネルに溜めさせない(計測にNagleの待ちを混ぜない)
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

    size_t n_img = lbls.size();
    size_t k = (size_t)idx * 97 % n_img;      // 接続ごとに開始位置をずらす
                                              // (全接続が同じ画像を同時に送るのを避ける)
    bool served = false;
    auto& mine = samples[idx];
    uint8_t resp[kRespBytes];

    while (std::chrono::steady_clock::now() < deadline) {
        const uint8_t* img = imgs.data() + k * kReqBytes;   // k枚目の先頭アドレス
        auto t0 = std::chrono::steady_clock::now();         // 送る直前
        if (send(fd, img, kReqBytes, MSG_NOSIGNAL) != (ssize_t)kReqBytes) break;
        if (!recv_all(fd, resp, kRespBytes)) break;         // 44B揃うまで待つ
        auto t1 = std::chrono::steady_clock::now();         // 返事が揃った直後
        // → t1-t0 が「クライアントから見た往復時間」。サーバーの推論時間だけではなく、
        //   ネットワークスタック+スレッドの起床+推論 すべてを含む

        // 返事の先頭4Bが予測。ラベルと一致するか数える(高速化で結果が壊れていないか確認)
        int32_t pred;
        std::memcpy(&pred, resp, sizeof(pred));
        if (pred == lbls[k]) ++g_correct;

        mine.push_back(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
        ++g_completed;
        if (!served) { served = true; ++g_served; }
        k = (k + 1) % n_img;   // 次の画像へ(最後まで行ったら先頭へ戻る)
    }
    close(fd);
}

// ソート済みの配列から p パーセンタイルを取る
// p95 = 「95%のリクエストがこの時間より速かった」値(Phase 2 M5 と同じ)
static long long percentile(const std::vector<long long>& s, double p) {
    if (s.empty()) return 0;
    size_t i = (size_t)(s.size() * p / 100.0);
    return s[std::min(i, s.size() - 1)];   // p=100 などで範囲外にならないよう頭打ち
}

int main(int argc, char** argv) {
    int conns = (argc > 1) ? std::atoi(argv[1]) : 16;   // 同時接続数
    int secs  = (argc > 2) ? std::atoi(argv[2]) : 5;    // 計測時間
    // 既定パスは「TcpServer/ で実行する」前提。build/ から実行するなら第3引数で渡す
    std::string dir = (argc > 3) ? argv[3] : "../OnnxLearn/data/MNIST/raw";

    // ===== MNIST テストデータを全部メモリに読む(計測中にファイルI/Oを混ぜないため) =====
    // dir == "dummy": MNIST なしで乱数画像を使う(重いモデルのレイテンシ測定用。正解率は無意味)
    const bool dummy = (dir == "dummy");
    std::vector<uint8_t> imgs, lbls;
    if (dummy) {
        const uint32_t n = 10000;
        imgs.resize((size_t)n * kReqBytes);
        lbls.assign(n, 255);   // 255 はどの予測(0〜9)とも一致しない
        uint32_t x = 12345;    // 固定シードの簡易乱数(測定ごとに入力を変えない)
        for (auto& b : imgs) { x = x * 1664525u + 1013904223u; b = x >> 24; }
    } else {
        std::ifstream fi(dir + "/t10k-images-idx3-ubyte", std::ios::binary);
        std::ifstream fl(dir + "/t10k-labels-idx1-ubyte", std::ios::binary);
        if (!fi || !fl) { std::cerr << "cannot open MNIST in " << dir << "\n"; return 1; }
        // 画像ファイルのヘッダ: magic, 枚数, 行数, 列数 の順(各4B)。枚数だけ使い、残りは読み飛ばす
        be32(fi); uint32_t n = be32(fi); be32(fi); be32(fi);
        // ラベルファイルのヘッダ: magic, 枚数(各4B)。どちらも読み飛ばす
        be32(fl); be32(fl);
        imgs.resize((size_t)n * kReqBytes);
        lbls.resize(n);
        fi.read((char*)imgs.data(), imgs.size());
        fl.read((char*)lbls.data(), lbls.size());
    }

    // ===== 接続数ぶんのスレッドを起動し、全員が同じ deadline まで走る =====
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(secs);
    std::vector<std::vector<long long>> samples(conns);   // スレッドごとの記録箱
    std::vector<std::thread> ts;
    for (int i = 0; i < conns; ++i)
        // std::cref/ref: スレッド引数は既定でコピーされるので、巨大な imgs をコピーしないよう参照で渡す
        ts.emplace_back(bench_client, i, std::cref(imgs), std::cref(lbls),
                        std::ref(samples), deadline);
    for (auto& t : ts) t.join();   // 全員が終わるまで待つ

    // ===== 集計: 全スレッドの記録を1本にまとめてソート → パーセンタイルを取る =====
    // 計測中は各スレッドが自分専用の箱に書くだけで、集計はここで1回だけ(計測を邪魔しない)
    std::vector<long long> all;
    for (auto& v : samples) all.insert(all.end(), v.begin(), v.end());
    std::sort(all.begin(), all.end());

    long total = g_completed.load();
    // served_conns が conns 未満なら、捌かれずに飢えた接続がある(Phase 2 M3 で見た現象)
    std::cout << "connections: " << conns << "  duration: " << secs << "s"
              << "  completed: " << total
              << "  served_conns: " << g_served.load() << "/" << conns
              << "  req/s: " << (double)total / secs << "\n";
    std::cout << "latency(us): p50=" << percentile(all, 50)
              << "  p95=" << percentile(all, 95)
              << "  p99=" << percentile(all, 99)
              << "  max=" << (all.empty() ? 0 : all.back()) << "\n";
    // 正解率が極端に低い(例: 10%前後)なら、速くなったのではなく壊れている
    if (dummy) std::cout << "accuracy: n/a (dummy input)\n";
    else std::cout << "accuracy: " << g_correct.load() << "/" << total << " = "
                   << (total ? 100.0 * g_correct.load() / total : 0.0) << "%\n";
}
