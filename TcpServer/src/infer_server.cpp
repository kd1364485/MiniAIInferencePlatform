// Phase 6 M1: 推論サーバー = M7(acceptor + epollワーカー) の echo を ONNX Runtime 推論に置換
//
// プロトコル(固定長バイナリ):
//   request : uint8 x 784   (MNIST 28x28 の生ピクセル)
//   response: int32 pred + float32 x 10 (logits)  = 44 bytes
//
// 使い方: ./infer_server <model.onnx> [workers=8] [intra_op_threads=1]
//
// 設計メモ:
//  - Ort::Session は Run() がスレッドセーフ → 全ワーカーで1個を共有(モデルの重みを1コピーに保つ)
//  - intra_op_threads は Session が持つ内部スレッドプールの大きさ。プールは Session 単位で、
//    同時に走る Run() 同士で共有される。実測のスレッド数は
//    「main + workers + (intra_op-1) + 1(正体未確認)」で、workers x intra_op の掛け算にはならない
//    (workers=4,intra=8 → 13本 / workers=8,intra=4 → 13本 / workers=8,intra=1 → 10本)
//    → 既定は intra_op=1: Run() を呼んだワーカー自身が計算し、プールへの受け渡しが無い。
//      33us の軽いモデルでは受け渡しが割に合わない、と推測している(intra_op を振った測定は未実施)
//  - TCPはバイトストリーム: 784B が1回の recv で揃う保証はない
//    → 接続ごとのバッファに溜め、784B揃ったら1リクエスト処理(フレーミング)
//
// 全体の流れ(M7と同じ骨格。変わったのは worker_loop の「受信→処理」部分だけ):
//
//   client ─784B→ [main: acceptor] ─fd(pending+eventfd)→ [worker x N: 各自epoll]
//                                                           │ recv(溜める) → ORT Run() → send(44B)
//                                                           ▼
//                                                         client

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>
#include <vector>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>        // TCP_NODELAY
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/socket.h>
#include <unistd.h>
#include <onnxruntime_cxx_api.h>   // ONNX Runtime の C++ API(Ort::Session など)

// ===== プロトコルのサイズ定義 =====
// 固定長にしておくと「どこまでが1リクエストか」を長さだけで判断できる(区切り文字が要らない)。
// クライアント(infer_bench.cpp)側にも同じ値が書いてある。ずれると通信が壊れるので注意。
constexpr size_t kReqBytes  = 784;                                   // 28x28 の1バイト画素
constexpr size_t kRespBytes = sizeof(int32_t) + 10 * sizeof(float);  // 予測(4B) + logits(40B) = 44B

// システムコール失敗時にメッセージを出して終了する。perror は errno の内容を文字にして表示する
static void die(const char* msg) { perror(msg); std::exit(1); }

// fd をノンブロッキングにする(M4と同じ)。
// 以後 recv/accept は「今できなければ -1 + errno=EAGAIN」で即座に返り、スレッドを眠らせない。
// epoll は「読めるよ」と教えてくれるだけで、実際の読み取りでブロックしないためにこれが要る。
static void set_nonblocking(int fd) {
    int flags = fcntl(fd, F_GETFL, 0);
    fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

// サーバー全体の統計。ワーカー間共有なので atomic(1命令の加算だけ守れば足りる)
// mutex を使わない理由: 「+= する」という1操作だけの保護なので、CPUの不可分命令(atomic)で足り、
// ロック待ちでワーカーが止まるコストを避けられる(Phase 2 の Q2 と同じ使い分け)。
std::atomic<uint64_t> g_requests{0};
std::atomic<uint64_t> g_infer_ns{0};   // ORT Run() にかかった時間の合計
std::atomic<bool>     g_stop{false};   // SIGINT/SIGTERM で true になる「止めて」フラグ

// ワーカー1人分の持ち物(M7と同じ)。mutex を含むのでコピー不可 → unique_ptr で管理する
struct Worker {
    int epoll_fd;             // このワーカー専用の epoll インスタンス(他スレッドと共有しない)
    int event_fd;             // acceptor が「新しい接続を渡したよ」と起こすための通知用 fd
    std::mutex mtx;           // pending を守る。触るのは acceptor と このワーカーの2者だけ
    std::queue<int> pending;  // acceptor が預けた、まだ epoll に登録していない接続 fd
};

// 接続ごとの受信バッファ。ワーカー専用のmapに入れるのでロック不要
//
// 【なぜ必要か】TCP はバイトの流れ(ストリーム)で、「send 1回 = recv 1回」は保証されない。
//   784B が 300B + 484B に分かれて届くことがある。echo は来た分をそのまま返せば良かったが、
//   推論は784B揃わないと入力にできない。だから「ここまで溜めた」を接続ごとに覚えておく。
//   この状態は epoll_wait をまたいで残る(M7の worker_loop は状態を持たなかった)。
struct Conn {
    uint8_t buf[kReqBytes];   // 溜め込み用の箱。リクエスト1個分ぴったり
    size_t  filled = 0;       // buf の何バイト目まで埋まっているか(0 なら空)
};

// 1リクエスト分の推論。前処理(uint8→正規化float) → ORT Run → argmax
//   session : 共有モデル(const ではないが Run() 自体はスレッドセーフ)
//   px      : 受信した 784B の生ピクセル
//   resp    : 書き込み先。kRespBytes(44B)分の領域を呼び出し側が用意する
static void run_inference(Ort::Session& session, const uint8_t* px, uint8_t* resp) {
    // 入出力テンソルの名前。export 時に付けた名前と一致している必要がある(export_*.py 参照)
    static const char* in_names[]  = {"input"};
    static const char* out_names[] = {"output"};

    // 「この入力バッファは CPU メモリ上にある」と ORT に伝えるラベル。
    // CreateTensor は確保せず既存配列を指すだけなので、ここで arena から確保が起きるわけではない。
    // thread_local = スレッドごとに1個だけ作って使い回す(毎回作らない)。
    thread_local Ort::MemoryInfo mem =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // --- 前処理: 学習時と同じ正規化を、サーバー側でやる ---
    // 0..255 → 0..1 → -1..1。ここを学習時とずらすと、推論は動くが精度が落ちる(エラーは出ない)。
    float input[784];
    for (int j = 0; j < 784; ++j)
        input[j] = px[j] / 255.0f * 2.0f - 1.0f;   // 学習時の Normalize と同じ

    // テンソルの形 (batch, channel, H, W)。今は1枚ずつなので batch=1
    int64_t shape[4] = {1, 1, 28, 28};
    // input 配列を「コピーせず」包む。→ Run() が終わるまで input はスタック上で生きている必要がある
    Ort::Value in = Ort::Value::CreateTensor<float>(mem, input, 784, shape, 4);

    // --- 推論本体。Run() だけを steady_clock で挟んで測る ---
    // steady_clock = 時計合わせで巻き戻らない単調時計(M5で学んだ区間計測の定石)
    auto t0 = std::chrono::steady_clock::now();
    auto outs = session.Run(Ort::RunOptions{nullptr}, in_names, &in, 1, out_names, 1);
    auto t1 = std::chrono::steady_clock::now();
    g_infer_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    ++g_requests;

    // --- 後処理: logits(10個の生スコア) → argmax で予測数字 ---
    // outs[0] の中身は ORT が確保したメモリ。outs が生きている間だけ有効なので、ここで resp にコピーする
    const float* logits = outs[0].GetTensorData<float>();
    int32_t pred = std::max_element(logits, logits + 10) - logits;   // ポインタ引き算=添字
    std::memcpy(resp, &pred, sizeof(pred));                          // 先頭4B: 予測
    std::memcpy(resp + sizeof(pred), logits, 10 * sizeof(float));    // 続く40B: logits
}

// 接続を片付ける。順番が大事:
//   1) epoll の監視リストから外す  2) 自分の管理表(conns)から消す  3) fd を閉じる
// ※ conns.erase すると、その Conn への参照(Conn& c)は無効になる。
//   呼んだ後に c を触ってはいけない(だから呼び出し側で closed フラグを経由している)。
static void close_conn(Worker& w, std::unordered_map<int, Conn>& conns, int fd) {
    epoll_ctl(w.epoll_fd, EPOLL_CTL_DEL, fd, nullptr);
    conns.erase(fd);
    close(fd);
}

// ── ワーカーの動作 ────────────────────────────────────────────
// epoll_wait が起きる契機は2種類(M7と同じ):
//   (a) event_fd が読める      → acceptor が新接続を預けた合図
//   (b) クライアント fd が読める → データ到着 or 切断
// M7との違い: (b) で「溜める→784B揃ったら推論→返信」をする。そのために conns(状態表)を持つ。
static void worker_loop(Worker& w, Ort::Session& session) {
    // このワーカーが担当する接続の状態表(キー = fd)。
    // ローカル変数なので他スレッドは触れない → ロック不要(sharding の利点)
    std::unordered_map<int, Conn> conns;
    epoll_event events[64];
    while (!g_stop) {
        // 登録した fd のどれかに用事ができるまで寝る(CPUを使わない)。-1 = 無期限
        int n = epoll_wait(w.epoll_fd, events, 64, -1);
        if (n <= 0) continue;   // シグナル割り込み(EINTR)などは単に回り直す

        for (int i = 0; i < n; ++i) {
            // 登録時に data.fd へ預けた「目印」がそのまま返ってくる(Phase 2 Q5)
            int fd = events[i].data.fd;

            // ---- (a) acceptor からの「新しい fd あるよ」 ----
            if (fd == w.event_fd) {
                uint64_t v;
                // eventfd のカウンタを読んで 0 に戻す(読まないと「起きろ」が鳴りっぱなしになる)。
                // `if (...) {}` は戻り値を無視する警告を黙らせるための書き方
                if (read(w.event_fd, &v, sizeof(v)) < 0) {}
                // 通知は1回でも pending に複数溜まっていることがあるので、空になるまで取り出す
                while (true) {
                    int cfd;
                    {   // mutex で守るのは「pending の出し入れ」だけ。外に出たら手放す(ロックを短く)
                        std::lock_guard<std::mutex> lk(w.mtx);
                        if (w.pending.empty()) break;
                        cfd = w.pending.front();
                        w.pending.pop();
                    }
                    set_nonblocking(cfd);
                    // TCP_NODELAY: 小さい応答(44B)をカーネルが「もう少し溜めてから送る」(Nagle)のを止める。
                    // 往復型の通信ではこの待ちがそのままレイテンシになる。
                    int one = 1;
                    setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
                    // 自分の epoll に登録: 「この fd が読めるようになったら起こして」
                    epoll_event ev{};
                    ev.events = EPOLLIN;
                    ev.data.fd = cfd;
                    epoll_ctl(w.epoll_fd, EPOLL_CTL_ADD, cfd, &ev);
                    // 状態表に空の Conn を作る(Conn{} は buf も 0 初期化、filled=0)
                    conns.emplace(cfd, Conn{});
                }
                continue;   // (a) はここで終わり。(b) の処理には進まない
            }

            // ---- (b) クライアントからのデータ or 切断 ----
            // 自分の状態表にない fd は無視(念のための防御)
            auto it = conns.find(fd);
            if (it == conns.end()) continue;
            Conn& c = it->second;   // この接続の「書きかけ」への参照

            // 受信: 784B揃うまで溜める。EAGAINまで読み切る。
            //  ・読み切る理由: 1回のepoll通知の間に複数リクエストが溜まっていることがあり、
            //    まとめて処理すると epoll_wait の呼び出し回数(システムコール)を減らせる。
            //    ※ここは epoll の既定(レベルトリガ)なので、読み残しても次の epoll_wait が
            //      すぐ再通知する。データを失う訳ではなく、効率の話。エッジトリガ(EPOLLET)に
            //      変えるなら「EAGAINまで読み切る」は必須になる。
            //  ・EAGAINで止まれる理由: set_nonblocking 済みだから。ブロッキングfdだと、空の
            //    recvで眠り、この worker が担当する全接続が止まってしまう。
            bool closed = false;    // ループ内で close_conn すると c が無効になるので、フラグで外に出す
            while (true) {
                // 書き込み先 = buf の「埋まっている所の次」。読む量 = 「残り」だけ。
                //  → 前回の続きから書き、784B を超えて次のリクエストの先頭を食べない。
                ssize_t r = recv(fd, c.buf + c.filled, kReqBytes - c.filled, 0);
                if (r > 0) {
                    c.filled += r;                     // 読めた分だけ「埋まり」を進める
                    if (c.filled == kReqBytes) {       // ちょうど1リクエスト分が揃った
                        uint8_t resp[kRespBytes];
                        run_inference(session, c.buf, resp);
                        // 44Bは送信バッファに必ず入る前提(同期クライアントなので溜まらない)
                        // MSG_NOSIGNAL: 相手が切れていても SIGPIPE で自分が死なないようにする
                        if (send(fd, resp, kRespBytes, MSG_NOSIGNAL) != (ssize_t)kRespBytes) {
                            closed = true; break;      // 全部送れなかった = 異常とみなして閉じる
                        }
                        c.filled = 0;   // 次のリクエストのために空に戻す(ここで0に戻すので、
                                        // 長さ0の recv=「切断」と誤認する事態が起きない)
                    }
                } else if (r == 0) { closed = true; break; }                // 相手がclose
                else if (errno == EAGAIN || errno == EWOULDBLOCK) break;   // 今は空
                else if (errno == EINTR) continue;                         // シグナルで中断 → やり直し
                else { closed = true; break; }                             // それ以外はエラー
            }
            if (closed) close_conn(w, conns, fd);   // ループの外で片付ける(c はもう使わない)
        }
    }
}

int main(int argc, char** argv) {
    // ===== 引数 =====
    if (argc < 2) { std::cerr << "usage: " << argv[0] << " <model.onnx> [workers] [intra_op_threads] [per_worker_session=0|1]\n"; return 1; }
    const char* model_path = argv[1];
    int kWorkers = (argc > 2) ? std::atoi(argv[2]) : 8;   // ワーカースレッド数(並列に捌く本数)
    int intra    = (argc > 3) ? std::atoi(argv[3]) : 1;   // ORT 1回の Run() 内部で使うスレッド数
    // 実験用: 1 なら worker ごとに専用の Session を作る(重みは worker 数ぶんのコピーになる)。
    // 0(既定)なら全 worker で1個を共有。共有による競合が Run() を遅くしているかを切り分けるためのもの
    int per_worker = (argc > 4) ? std::atoi(argv[4]) : 0;

    // ===== シグナル =====
    // 相手が切れたソケットに send すると既定では SIGPIPE でプロセスごと死ぬ → 無視にする
    signal(SIGPIPE, SIG_IGN);
    // Ctrl+C(SIGINT) / kill(SIGTERM) では g_stop を立てるだけ。
    // ハンドラ内でできることは極めて限られるため、「フラグを立てる→メインループが気づいて後片付け」にする
    signal(SIGINT,  [](int) { g_stop = true; });
    signal(SIGTERM, [](int) { g_stop = true; });

    // ===== ONNX Runtime: セッションは1個を全ワーカーで共有 =====
    // Env = ONNX Runtime のグローバルな実行環境(ログ設定など)。Session より長生きさせる必要がある
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "infer_server");
    Ort::SessionOptions so;
    // Session の内部演算スレッドプールの大きさ(全 Run() で共有)。1 ならプールを作らず、
    // Run() を呼んだワーカースレッドがその場で計算する
    so.SetIntraOpNumThreads(intra);
    // グラフ最適化(演算の融合など)を最大にする。モデル読み込み時に1回だけ行う
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    // モデルファイルを読み込み、重みをメモリに載せる。ここは起動時の1回だけ(リクエストごとではない)
    Ort::Session session(env, model_path, so);

    // ウォームアップ: 初回Runは遅延初期化(メモリ確保等)を含むので測定に混ぜない
    // (Phase 4 M5 の「初回157ms=JITコンパイル混入」と同じ種類の落とし穴)
    {
        uint8_t zero[kReqBytes] = {}, resp[kRespBytes];
        for (int i = 0; i < 20; ++i) run_inference(session, zero, resp);
        g_requests = 0; g_infer_ns = 0;   // ウォームアップ分の統計は捨てる
    }

    // per_worker=1 のときだけ使う、worker 専用の Session 群(共有時は空のまま)
    std::vector<std::unique_ptr<Ort::Session>> own_sessions;
    if (per_worker) {
        uint8_t zero[kReqBytes] = {}, resp[kRespBytes];
        for (int i = 0; i < kWorkers; ++i) {
            own_sessions.push_back(std::make_unique<Ort::Session>(env, model_path, so));
            for (int k = 0; k < 20; ++k) run_inference(*own_sessions[i], zero, resp);
        }
        g_requests = 0; g_infer_ns = 0;
    }

    // ===== listen ソケット(M1〜M7と同じ定型) =====
    int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    // SO_REUSEADDR: 再起動直後に "Address already in use" で bind 失敗するのを防ぐ(M1で体験したやつ)
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(8080);         // htons = ホスト→ネットワークのバイト順変換
    addr.sin_addr.s_addr = INADDR_ANY;   // 全ネットワークインターフェースで待ち受け
    if (bind(listen_fd, (sockaddr*)&addr, sizeof(addr)) < 0) die("bind");
    // 第2引数 = 「accept される前の接続待ち行列」の長さ。100接続が一斉に来ても溢れないよう大きめに
    // (M7の128から拡大。実際の上限は OS 設定 somaxconn で頭打ちになる)
    if (listen(listen_fd, 1024) < 0) die("listen");
    set_nonblocking(listen_fd);          // accept も EAGAIN で即戻るようにする

    // ===== ワーカー起動 =====
    std::vector<std::unique_ptr<Worker>> workers;   // Worker は mutex を含みコピー不可 → ポインタで持つ
    std::vector<std::thread> threads;
    for (int i = 0; i < kWorkers; ++i) {
        auto w = std::make_unique<Worker>();
        w->epoll_fd = epoll_create1(0);             // このワーカー専用の epoll
        w->event_fd = eventfd(0, EFD_NONBLOCK);     // 「起きろ」用の通知 fd
        // 自分の event_fd を自分の epoll に登録(acceptor の合図で起きられるように)
        epoll_event ev{};
        ev.events = EPOLLIN;
        ev.data.fd = w->event_fd;
        epoll_ctl(w->epoll_fd, EPOLL_CTL_ADD, w->event_fd, &ev);
        workers.push_back(std::move(w));
        // std::ref: スレッドへの引数は既定でコピーされる。Worker も Session もコピー不可/したくないので
        // 「参照で渡せ」の目印(M3の FdQueue と同じ)。session は全ワーカーが同じ1個を指す。
        Ort::Session& s = per_worker ? *own_sessions[i] : session;   // 共有 or 専用
        threads.emplace_back(worker_loop, std::ref(*workers[i]), std::ref(s));
    }

    // ===== メイン = acceptor(listen_fd だけを epoll で監視) =====
    int ep = epoll_create1(0);
    epoll_event ev{};
    ev.events = EPOLLIN;
    ev.data.fd = listen_fd;
    epoll_ctl(ep, EPOLL_CTL_ADD, listen_fd, &ev);

    // endl = 改行 + flush。起動完了をすぐ表示させる
    std::cout << "infer_server: port 8080, model=" << model_path
              << ", workers=" << kWorkers << ", intra_op_threads=" << intra
              << ", session=" << (per_worker ? "per-worker" : "shared") << std::endl;

    int next = 0;   // ラウンドロビン用: 次に渡すワーカーの番号
    while (!g_stop) {
        epoll_event e;
        // M7は -1(無期限)だったが、止められなくなるので500msごとに起きて g_stop を確認する
        int n = epoll_wait(ep, &e, 1, 500);
        if (n <= 0) continue;
        // listen_fd が読める = 新着接続あり。EAGAIN(cfd<0)になるまで全部 accept
        while (true) {
            int cfd = accept(listen_fd, nullptr, nullptr);
            if (cfd < 0) break;
            Worker& w = *workers[next];            // ラウンドロビンで担当ワーカーを選ぶ
            next = (next + 1) % kWorkers;
            {
                std::lock_guard<std::mutex> lk(w.mtx);
                w.pending.push(cfd);               // fd を「預ける」(中身の受け渡し)
            }
            uint64_t one = 1;
            // eventfd に書く = 寝ているワーカーを起こす(「起こす手段」と「中身」は別物。Phase 2 Q3)
            if (write(w.event_fd, &one, sizeof(one)) < 0) {}
        }
    }

    // 停止時に統計を出す: ORT Run() の平均時間 = 「サーバー内の推論コスト」
    // これと クライアントから見た往復時間(p50等)を比べると、推論が全体の何割かが分かる
    uint64_t n = g_requests.load();
    std::cout << "requests=" << n << "  avg ORT Run()="
              << (n ? g_infer_ns.load() / n / 1000.0 : 0.0) << " us" << std::endl;
    // ワーカーは epoll_wait(-1) で寝ているので join できない。_exit で強制終了する。
    // ※ _exit は iostream のバッファを flush しない。上の endl はそのために必要(前回のバグの原因)
    _exit(0);
}
