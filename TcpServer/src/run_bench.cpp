// run_bench: ネットワークなしで ORT の Run() だけを N スレッドで回し、Run() の平均時間を測る
//
// 目的: infer_server で worker を増やすと Run() が 25us → 100us超に伸びた。その原因が
//   (A) サーバー側(ネットワーク、epoll、スレッド起床など) なのか
//   (B) ORT/ハードウェア側(同時に Run() を走らせること自体) なのか を切り分ける。
//   ネットワークを外しても伸びるなら (B)。
//
// 使い方: ./run_bench <model.onnx> <threads> <seconds> <shared=1|0>
//   threads : 同時に Run() を呼ぶスレッド数(infer_server の workers に相当)
//   shared  : 1=全スレッドで Session を1個共有 / 0=スレッドごとに専用 Session
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
#include <thread>
#include <vector>
#include <onnxruntime_cxx_api.h>

using Clock = std::chrono::steady_clock;

// スレッドごとの結果。false sharing(隣の変数と同じキャッシュラインを奪い合う現象)を避けるため
// 64バイト境界に揃えて、スレッド間で別のキャッシュラインに載せる
struct alignas(64) Result {
    uint64_t runs = 0;
    uint64_t ns = 0;
};

static void worker(Ort::Session& session, std::atomic<bool>& go, int seconds, Result& out) {
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
        auto t0 = Clock::now();
        run_once();
        auto t1 = Clock::now();
        out.ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        ++out.runs;
    }
}

int main(int argc, char** argv) {
    if (argc < 5) { std::cerr << "usage: " << argv[0] << " <model.onnx> <threads> <seconds> <shared=1|0>\n"; return 1; }
    const char* model = argv[1];
    int n = std::atoi(argv[2]);
    int seconds = std::atoi(argv[3]);
    int shared = std::atoi(argv[4]);

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
        ts.emplace_back(worker, std::ref(*sessions[shared ? 0 : i]), std::ref(go), seconds, std::ref(results[i]));

    std::this_thread::sleep_for(std::chrono::milliseconds(1500));   // ウォームアップ完了を待つ
    go = true;
    for (auto& t : ts) t.join();

    uint64_t runs = 0, ns = 0;
    for (auto& r : results) { runs += r.runs; ns += r.ns; }
    std::cout << "threads=" << n << " shared=" << shared
              << "  avg Run()=" << (runs ? ns / runs / 1000.0 : 0.0) << " us"
              << "  runs/s=" << runs / seconds << std::endl;
}
