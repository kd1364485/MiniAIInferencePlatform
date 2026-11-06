// run_bench: ネットワークなしで ORT の Run() だけを N スレッドで回し、Run() の平均時間を測る
//
// 目的: infer_server で worker を増やすと Run() が 25us → 100us超に伸びた。その原因が
//   (A) サーバー側(ネットワーク、epoll、スレッド起床など) なのか
//   (B) ORT/ハードウェア側(同時に Run() を走らせること自体) なのか を切り分ける。
//   ネットワークを外しても伸びるなら (B)。
//
// 使い方: ./run_bench <model.onnx> <threads> <seconds> <shared=1|0> [think_us=0] [mode=sleep|spin|net] [burners=0]
//   threads : 同時に Run() を呼ぶスレッド数(infer_server の workers に相当)
//   shared  : 1=全スレッドで Session を1個共有 / 0=スレッドごとに専用 Session
//   think_us: Run() と Run() の間に入れる「待ち」の長さ(us)。0(既定)なら休みなく連続で回す。
//             サーバーのワーカーは「リクエストを待って眠る → 起きて Run()」を繰り返す。
//             その「眠った後は遅いのか」を、ネットワークなしで確かめるための実験用の引数
//   mode    : sleep=眠る(スレッドが CPU を手放す) / spin=CPU を回したまま待つ(手放さない)
//             同じ長さの待ちで、sleep だけ Run() が伸びるなら、「眠って起きること」自体が原因
//             net=Run() の間に、同じスレッドが、ループバック TCP の送受信(784B の送信→受信、44B の送信→受信)をする。
//             サーバーのワーカーが、Run() の前後にやっている recv / send と同じ種類の処理。
//             これで Run() が伸びるなら、「ネットワーク処理でキャッシュが汚れること」が原因の濃厚な候補
//   burners : 計算だけを回し続ける「邪魔なスレッド」の数。機械全体を混雑させたとき(SMT・省電力コア・
//             周波数など)に、Run() が伸びるかを見る。サーバーの実験では、bench の 100 スレッドなどが他の CPU を使っている
//
// 比べ方: 同じ N を「1プロセスN スレッド」と「N プロセス x 1スレッド」で測る。
//   プロセスを分けると、メモリ空間も ORT のグローバル状態も別になる。それでも伸びるなら
//   プロセス内の共有物ではなく、CPU(ハードウェア)側の要因が濃厚。

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <time.h>
#include <vector>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
#include <onnxruntime_cxx_api.h>

using Clock = std::chrono::steady_clock;

// スレッドごとの結果。false sharing(隣の変数と同じキャッシュラインを奪い合う現象)を避けるため
// 64バイト境界に揃えて、スレッド間で別のキャッシュラインに載せる
struct alignas(64) Result {
    uint64_t runs = 0;
    uint64_t ns = 0;       // Run() の壁時計の時間の合計
    uint64_t cpu_ns = 0;   // 同じ区間で、そのスレッドが CPU を使った時間の合計
};

// そのスレッドが CPU を使った時間(ns)。壁時計より小さければ、CPU を取られて(待たされて)いた
static uint64_t thread_cpu_ns() {
    timespec ts;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

// このスレッド専用の、ループバック TCP の接続の組(a と b)を作る。サーバーの recv / send と同じ種類の処理を、
// 他のスレッド・プロセスを使わずに、1 スレッドの中で再現するため。失敗したら false
struct LoopbackPair {
    int a = -1, b = -1;
    bool open() {
        int ls = socket(AF_INET, SOCK_STREAM, 0);
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        addr.sin_port = 0;
        socklen_t len = sizeof(addr);
        if (ls < 0 || bind(ls, (sockaddr*)&addr, sizeof(addr)) != 0 || listen(ls, 1) != 0 ||
            getsockname(ls, (sockaddr*)&addr, &len) != 0) { if (ls >= 0) close(ls); return false; }
        a = socket(AF_INET, SOCK_STREAM, 0);
        if (connect(a, (sockaddr*)&addr, sizeof(addr)) != 0) { close(ls); return false; }
        b = accept(ls, nullptr, nullptr);
        close(ls);
        int one = 1;
        setsockopt(a, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        setsockopt(b, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        return b >= 0;
    }
    // 784B を送って受け取り、44B を送って受け取る(1 リクエスト分のやり取りの、4 回の syscall)
    void roundtrip() {
        char req[784] = {}, resp[44] = {}, buf[784];
        if (send(a, req, sizeof(req), MSG_NOSIGNAL) < 0) return;
        for (size_t got = 0; got < sizeof(req);) { ssize_t n = recv(b, buf, sizeof(buf), 0); if (n <= 0) return; got += n; }
        if (send(b, resp, sizeof(resp), MSG_NOSIGNAL) < 0) return;
        for (size_t got = 0; got < sizeof(resp);) { ssize_t n = recv(a, buf, sizeof(buf), 0); if (n <= 0) return; got += n; }
    }
    void close_all() { if (a >= 0) close(a); if (b >= 0) close(b); }
};

static void worker(Ort::Session& session, std::atomic<bool>& go, int seconds, Result& out,
                   int think_us, bool spin, bool net) {
    LoopbackPair pair;
    if (net && !pair.open()) { std::cerr << "loopback pair failed\n"; net = false; }
    const char* in_names[]  = {"input"};
    const char* out_names[] = {"output"};
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    float input[784];
    for (int j = 0; j < 784; ++j) input[j] = (j % 256) / 255.0f * 2.0f - 1.0f;   // 固定の入力
    int64_t shape[4] = {1, 1, 28, 28};

    auto run_once = [&]() {
        Ort::Value in = Ort::Value::CreateTensor<float>(mem, input, 784, shape, 4);
        auto outs = session.Run(Ort::RunOptions{nullptr}, in_names, &in, 1, out_names, 1);
    };

    for (int i = 0; i < 500; ++i) run_once();       // ウォームアップ(初回の遅延初期化を除く)

    while (!go.load()) {}                           // 全スレッドの準備ができるまで待つ(開始を揃える)

    auto end = Clock::now() + std::chrono::seconds(seconds);
    while (Clock::now() < end) {
        const uint64_t c0 = thread_cpu_ns();
        auto t0 = Clock::now();
        run_once();
        auto t1 = Clock::now();
        const uint64_t c1 = thread_cpu_ns();
        out.ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        out.cpu_ns += c1 - c0;
        ++out.runs;

        // Run() の間の処理(測定の対象外)
        if (net) {
            pair.roundtrip();          // ネットワークの syscall(本物のループバック TCP)
        } else if (think_us > 0) {     // sleep は CPU を手放し、spin は手放さない
            if (spin) {
                auto until = Clock::now() + std::chrono::microseconds(think_us);
                while (Clock::now() < until) {}
            } else {
                std::this_thread::sleep_for(std::chrono::microseconds(think_us));
            }
        }
    }
    pair.close_all();
}

int main(int argc, char** argv) {
    if (argc < 5) { std::cerr << "usage: " << argv[0] << " <model.onnx> <threads> <seconds> <shared=1|0>\n"; return 1; }
    const char* model = argv[1];
    int n = std::atoi(argv[2]);
    int seconds = std::atoi(argv[3]);
    int shared = std::atoi(argv[4]);
    int think_us = (argc > 5) ? std::atoi(argv[5]) : 0;
    const std::string mode = (argc > 6) ? argv[6] : "sleep";
    bool spin = (mode == "spin");
    bool net = (mode == "net");
    int burners = (argc > 7) ? std::atoi(argv[7]) : 0;

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "run_bench");
    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(1);                     // infer_server と同じ設定
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // shared=1 なら Session は1個、0 ならスレッドごとに作る
    std::vector<std::unique_ptr<Ort::Session>> sessions;
    int n_sessions = shared ? 1 : n;
    for (int i = 0; i < n_sessions; ++i)
        sessions.push_back(std::make_unique<Ort::Session>(env, model, so));

    std::vector<Result> results(n);
    std::atomic<bool> go{false};
    std::vector<std::thread> ts;
    for (int i = 0; i < n; ++i)
        ts.emplace_back(worker, std::ref(*sessions[shared ? 0 : i]), std::ref(go), seconds, std::ref(results[i]),
                        think_us, spin, net);

    // 邪魔なスレッド: 計算だけを回し続けて、機械全体を混雑させる(メモリ・ネットワークは使わない)
    std::atomic<bool> burn_stop{false};
    std::vector<std::thread> burn;
    for (int i = 0; i < burners; ++i)
        burn.emplace_back([&burn_stop] {
            volatile double x = 1.0;
            while (!burn_stop.load(std::memory_order_relaxed)) {
                for (int k = 0; k < 1000; ++k) x = x * 1.0000001 + 0.5;
            }
        });

    std::this_thread::sleep_for(std::chrono::milliseconds(1500));   // ウォームアップ完了を待つ
    go = true;
    for (auto& t : ts) t.join();
    burn_stop = true;
    for (auto& t : burn) t.join();

    uint64_t runs = 0, ns = 0, cpu_ns = 0;
    for (auto& r : results) { runs += r.runs; ns += r.ns; cpu_ns += r.cpu_ns; }
    std::cout << "threads=" << n << " shared=" << shared
              << "  think=" << think_us << "us(" << (net ? "net" : (think_us > 0 ? (spin ? "spin" : "sleep") : "none")) << ")"
              << "  burners=" << burners
              << "  avg Run()=" << (runs ? ns / runs / 1000.0 : 0.0) << " us"
              << "  cpu=" << (runs ? cpu_ns / runs / 1000.0 : 0.0) << " us"
              << "  runs/s=" << runs / seconds << std::endl;
}
