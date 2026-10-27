// Phase 6 M1: 推論サーバー = M7(acceptor + epollワーカー) の echo を ONNX Runtime 推論に置換
// Phase 6 M5: Request Queue 版を追加(infer_threads >= 1 のとき。0 なら従来どおりインライン推論)
//
// プロトコル(固定長バイナリ):
//   request : uint8 x 784   (MNIST 28x28 の生ピクセル)
//   response: int32 pred + float32 x 10 (logits)  = 44 bytes
//
// 使い方: ./infer_server <model.onnx> [workers=8] [intra_op_threads=1] [per_worker_session=0]
//                        [max_batch=1] [infer_threads=0]
//   workers       : I/O worker の本数(epoll で受信・返信する)。infer_threads=0 ならこの worker が推論もする
//   infer_threads : 0 = インライン(I/O worker が自分で Run() する。従来動作)
//                   1以上 = 推論専用 worker をこの本数だけ作り、共有 Request Queue 経由で推論する
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
// 全体の流れ(インライン版。M7と同じ骨格):
//
//   client ─784B→ [main: acceptor] ─fd(pending+eventfd)→ [worker x N: 各自epoll]
//                                                           │ recv(溜める) → ORT Run() → send(44B)
//                                                           ▼
//                                                         client
//
// 全体の流れ(キュー版 infer_threads >= 1):
//
//   client ─→ [main: acceptor] ─fd→ [I/O worker x N_io : epoll]
//                                      │ ① 784B 揃ったら Request を push
//                                      ▼
//                         [ 共有 Request Queue (mutex + condition_variable) ]
//                                      │ ② 最大 max_batch 件まとめて pop
//                                      ▼
//                         [ 推論 worker x infer_threads ] ── Run() ── 共有 Session
//                                      │ ③ 持ち主の I/O worker の done に push + eventfd
//                                      ▼
//                         [ I/O worker ] ④ 接続が生きているか確認(gen) → send 44B
//
//  ★キュー版の最重要ルール: fd と Conn に触るのは「持ち主の I/O worker だけ」。
//    推論 worker は send も close もしない。→ 推論中に接続が閉じて fd 番号が別の接続に
//    再利用されても、他人の接続へ誤送信しない(返信に付けた gen で照合して、不一致なら捨てる)。

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
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

using Clock = std::chrono::steady_clock;

// サーバー全体の統計。ワーカー間共有なので atomic(1命令の加算だけ守れば足りる)
// mutex を使わない理由: 「+= する」という1操作だけの保護なので、CPUの不可分命令(atomic)で足り、
// ロック待ちでワーカーが止まるコストを避けられる(Phase 2 の Q2 と同じ使い分け)。
std::atomic<uint64_t> g_requests{0};
std::atomic<uint64_t> g_infer_ns{0};   // ORT Run() にかかった時間の合計
std::atomic<uint64_t> g_batches{0};    // Run() を呼んだ回数(バッチ数)
std::atomic<bool>     g_stop{false};   // SIGINT/SIGTERM で true になる「止めて」フラグ

// ---- キュー版の統計 ----
std::atomic<uint64_t> g_qwait_ns{0};      // キューで待った時間の合計(push → pop)
std::atomic<uint64_t> g_qwait_max_ns{0};  // キュー待ち時間の最大
std::atomic<uint64_t> g_qdepth_max{0};    // キューに溜まった件数の最大
std::atomic<uint64_t> g_stale{0};         // 返信を捨てた回数(接続が閉じた/別の接続になっていた)
std::atomic<uint64_t> g_violations{0};    // 「1接続につき未完了1件まで」違反で接続を閉じた回数

// 最大値を更新する(他スレッドと競合するので compare_exchange でやり直す)
static void atomic_max(std::atomic<uint64_t>& a, uint64_t v) {
    uint64_t cur = a.load();
    while (cur < v && !a.compare_exchange_weak(cur, v)) {}
}

// 推論 worker → 持ち主の I/O worker への返信
//   fd と gen の組で「どの接続宛てか」を表す。fd 番号だけだと、閉じた後に再利用された
//   別の接続と区別できないので、接続ごとの通し番号 gen を一緒に持たせる。
struct Response {
    int      fd;
    uint64_t gen;
    bool     ok;                 // false = 推論に失敗(接続を閉じてほしい)
    uint8_t  resp[kRespBytes];
};

// I/O worker → 推論 worker へのリクエスト。784B は「コピー」して持つ。
//   → I/O worker 側の Conn のバッファはすぐ空にできる。推論中に Conn が消えても安全。
struct Request {
    int      io_id;              // 持ち主の I/O worker の番号(返信の宛先)
    int      fd;
    uint64_t gen;
    uint8_t  px[kReqBytes];
    Clock::time_point t_enq;     // キューに入れた時刻(キュー待ち時間の測定用)
};

// ワーカー1人分の持ち物(M7と同じ)。mutex を含むのでコピー不可 → unique_ptr で管理する
struct Worker {
    int epoll_fd;             // このワーカー専用の epoll インスタンス(他スレッドと共有しない)
    int event_fd;             // 「何か届いたよ」と起こすための通知用 fd(新接続 / 推論の返信 の両方で使う)
    std::mutex mtx;           // pending と done を守る。触るのは「入れる側」と このワーカーの2者だけ
    std::queue<int> pending;  // acceptor が預けた、まだ epoll 未登録の接続 fd
    std::queue<Response> done;   // 推論 worker が預けた返信(キュー版のみ使う)
};

// 共有 Request Queue(キュー版のみ)。M3 の producer-consumer と同じ形:
//   mutex でキューを守り、condition_variable で「空なら寝る / 入ったら起きる」。
std::mutex              g_q_mtx;
std::condition_variable g_q_cv;
std::deque<Request>     g_q;

// 接続ごとの受信バッファ。ワーカー専用のmapに入れるのでロック不要
//
// 【なぜ必要か】TCP はバイトの流れ(ストリーム)で、「send 1回 = recv 1回」は保証されない。
//   784B が 300B + 484B に分かれて届くことがある。echo は来た分をそのまま返せば良かったが、
//   推論は784B揃わないと入力にできない。だから「ここまで溜めた」を接続ごとに覚えておく。
//   この状態は epoll_wait をまたいで残る(M7の worker_loop は状態を持たなかった)。
struct Conn {
    uint8_t  buf[kReqBytes];   // 溜め込み用の箱。リクエスト1個分ぴったり
    size_t   filled = 0;       // buf の何バイト目まで埋まっているか(0 なら空)
    uint64_t gen = 0;          // 接続の通し番号(キュー版): fd 番号が再利用されても別の接続と見分ける
    bool     in_flight = false;   // キュー版: 推論待ち(返信がまだ戻っていない)かどうか
};

// logits(10個)から返信 44B を作る: 先頭4B = argmax の予測数字、続く40B = logits
static void make_resp(const float* logits, uint8_t* resp) {
    int32_t pred = std::max_element(logits, logits + 10) - logits;   // ポインタ引き算=添字
    std::memcpy(resp, &pred, sizeof(pred));
    std::memcpy(resp + sizeof(pred), logits, 10 * sizeof(float));
}

// 推論の中心部分(インライン版・キュー版の両方がこれを使う → 比較が公平になる)
//   px[0..b) : 各 784B の生ピクセルへのポインタ(b 件)
//   out      : 結果。(b,10) の logits が行ごとに連続して並ぶ(i 番目の入力 ↔ out[i*10 .. i*10+9])
//   モデルのバッチ軸が可変である必要がある(b>=2 のとき)。Run() が失敗すると Ort::Exception を投げる。
static void forward_batch(Ort::Session& session, const uint8_t* const* px, size_t b,
                          std::vector<float>& out) {
    // 入出力テンソルの名前。export 時に付けた名前と一致している必要がある
    static const char* in_names[]  = {"input"};
    static const char* out_names[] = {"output"};

    // 「この入力バッファは CPU メモリ上にある」と ORT に伝えるラベル。
    // CreateTensor は確保せず既存配列を指すだけ。thread_local = スレッドごとに1個を使い回す。
    thread_local Ort::MemoryInfo mem =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    thread_local std::vector<float> input;   // 毎回 new しないよう使い回す(スレッド専用)
    input.resize(b * kReqBytes);

    // --- 前処理: b 枚を縦に並べて1つのテンソルにする(batch 軸 = 先頭) ---
    // 学習時と同じ正規化: 0..255 → 0..1 → -1..1。ずらすと推論は動くが精度が落ちる(エラーは出ない)。
    for (size_t i = 0; i < b; ++i) {
        float* dst = input.data() + i * kReqBytes;
        for (size_t j = 0; j < kReqBytes; ++j) dst[j] = px[i][j] / 255.0f * 2.0f - 1.0f;
    }
    // テンソルの形 (batch, channel, H, W)。input を「コピーせず」包む → Run() の間 input は生きている
    int64_t shape[4] = {(int64_t)b, 1, 28, 28};
    Ort::Value in = Ort::Value::CreateTensor<float>(mem, input.data(), input.size(), shape, 4);

    // --- 推論本体。Run() だけを steady_clock で挟んで測る ---
    auto t0 = Clock::now();
    auto outs = session.Run(Ort::RunOptions{nullptr}, in_names, &in, 1, out_names, 1);
    auto t1 = Clock::now();
    g_infer_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    g_requests += b;
    ++g_batches;

    // outs の中身は ORT が確保したメモリで、outs が生きている間だけ有効 → out にコピーして返す
    const float* all = outs[0].GetTensorData<float>();
    out.assign(all, all + b * 10);
}

// 1リクエスト分の推論(起動時のウォームアップ用)
static void run_inference(Ort::Session& session, const uint8_t* px, uint8_t* resp) {
    std::vector<float> logits;
    forward_batch(session, &px, 1, logits);
    make_resp(logits.data(), resp);
}

// 接続を片付ける。順番が大事:
//   1) epoll の監視リストから外す  2) 自分の管理表(conns)から消す  3) fd を閉じる
// ※ conns.erase すると、その Conn への参照(Conn& c)は無効になる。
//   呼んだ後に c を触ってはいけない(だから呼び出し側で closed フラグを経由している)。
// ※ close(fd) が最後なのは、閉じた後の fd 番号は他の接続に再利用されうるから。
static void close_conn(Worker& w, std::unordered_map<int, Conn>& conns, int fd) {
    epoll_ctl(w.epoll_fd, EPOLL_CTL_DEL, fd, nullptr);
    conns.erase(fd);
    close(fd);
}

// ===== インライン版: 受信が揃った接続 fd(ready[0..b))を、バッチ b 件として1回の Run() で処理して返信する =====
//   送信に失敗した fd(閉じるべき接続)は fails に積む。ここで close しないのは、呼び出し側が
//   ready を回している最中に conns を壊さないため。
static void run_batch(Ort::Session& session, std::unordered_map<int, Conn>& conns,
                      const int* ready, size_t b, std::vector<int>& fails) {
    thread_local std::vector<const uint8_t*> px;
    thread_local std::vector<float> logits;
    px.clear();
    for (size_t i = 0; i < b; ++i) px.push_back(conns.at(ready[i]).buf);

    try {
        forward_batch(session, px.data(), b, logits);
    } catch (const Ort::Exception& e) {
        // 例外を握りつぶさず報告し、このバッチの接続は閉じる(放置すると worker ごと terminate する)
        std::cerr << "Run() failed: " << e.what() << std::endl;
        for (size_t i = 0; i < b; ++i) fails.push_back(ready[i]);
        return;
    }
    // 出力は (b,10) が連続して並ぶ。i 番目の接続には i 行目を返す
    for (size_t i = 0; i < b; ++i) {
        uint8_t resp[kRespBytes];
        make_resp(logits.data() + i * 10, resp);
        int fd = ready[i];
        if (send(fd, resp, kRespBytes, MSG_NOSIGNAL) != (ssize_t)kRespBytes) fails.push_back(fd);
        conns.at(fd).filled = 0;   // 次のリクエストのために空に戻す
    }
}

// ── I/O ワーカーの動作 ────────────────────────────────────────────
// epoll_wait が起きる契機は2種類(M7と同じ):
//   (a) event_fd が読める      → 新接続が預けられた / (キュー版) 推論の返信が届いた
//   (b) クライアント fd が読める → データ到着 or 切断
// use_queue=false: 溜めて、揃った分をこの worker が自分で Run() して返信(従来)
// use_queue=true : 溜めて、揃った分を Request として共有キューに積む。返信は (a) で受け取って send
static void worker_loop(Worker& w, Ort::Session& session, size_t max_batch,
                        int io_id, bool use_queue) {
    // このワーカーが担当する接続の状態表(キー = fd)。
    // ローカル変数なので他スレッドは触れない → ロック不要(sharding の利点)
    std::unordered_map<int, Conn> conns;
    uint64_t next_gen = 1;            // 接続の通し番号(この worker 内で一意なら十分)
    epoll_event events[64];
    std::vector<int> ready;           // インライン版: この回で「784B揃った」接続(バッチの材料)
    std::vector<int> fails;
    std::vector<Request> outbox;      // キュー版: この回で共有キューへ積む Request
    std::vector<Response> got;        // キュー版: 推論 worker から受け取った返信
    while (!g_stop) {
        ready.clear();
        outbox.clear();
        // 登録した fd のどれかに用事ができるまで寝る(CPUを使わない)。-1 = 無期限
        int n = epoll_wait(w.epoll_fd, events, 64, -1);
        if (n <= 0) continue;   // シグナル割り込み(EINTR)などは単に回り直す

        for (int i = 0; i < n; ++i) {
            // 登録時に data.fd へ預けた「目印」がそのまま返ってくる(Phase 2 Q5)
            int fd = events[i].data.fd;

            // ---- (a) event_fd: 新しい fd / 推論の返信 ----
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
                    // 状態表に新しい Conn を作る。gen = この接続だけの通し番号
                    Conn nc;
                    nc.gen = next_gen++;
                    conns.emplace(cfd, nc);
                }

                // キュー版: 推論 worker から届いた返信を取り出して、持ち主としてここで send する
                if (use_queue) {
                    got.clear();
                    {
                        std::lock_guard<std::mutex> lk(w.mtx);
                        while (!w.done.empty()) { got.push_back(w.done.front()); w.done.pop(); }
                    }
                    for (const Response& r : got) {
                        auto cit = conns.find(r.fd);
                        // fd が表に無い(閉じた) or gen が違う(同じ fd 番号の別の接続) → 返信を捨てる
                        if (cit == conns.end() || cit->second.gen != r.gen) { ++g_stale; continue; }
                        cit->second.in_flight = false;
                        if (!r.ok || send(r.fd, r.resp, kRespBytes, MSG_NOSIGNAL) != (ssize_t)kRespBytes)
                            close_conn(w, conns, r.fd);   // この後 cit は使わない
                    }
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
            // 1接続につき 1リクエスト分(784B)揃った時点で読むのをやめる(推論は後でまとめて行う)。
            while (c.filled < kReqBytes) {
                // 書き込み先 = buf の「埋まっている所の次」。読む量 = 「残り」だけ。
                //  → 前回の続きから書き、784B を超えて次のリクエストの先頭を食べない。
                ssize_t r = recv(fd, c.buf + c.filled, kReqBytes - c.filled, 0);
                if (r > 0) c.filled += r;                                   // 読めた分だけ「埋まり」を進める
                else if (r == 0) { closed = true; break; }                  // 相手がclose
                else if (errno == EAGAIN || errno == EWOULDBLOCK) break;   // 今は空
                else if (errno == EINTR) continue;                         // シグナルで中断 → やり直し
                else { closed = true; break; }                             // それ以外はエラー
            }
            if (closed) {
                close_conn(w, conns, fd);   // ループの外で片付ける(c はもう使わない)
            } else if (c.filled == kReqBytes) {
                if (!use_queue) {
                    ready.push_back(fd);    // インライン版: 推論待ちの列に並べる
                } else if (c.in_flight) {
                    // 返信が戻る前に2件目が揃った = 1接続につき未完了1件までの約束違反(パイプライン)。
                    // 返信の順序保証が複雑になるので、v1 では接続を閉じる。
                    ++g_violations;
                    close_conn(w, conns, fd);
                } else {
                    // キュー版: 784B をコピーして Request にし、後でまとめて共有キューへ積む。
                    // コピーしたので Conn のバッファはすぐ空に戻せる(次のリクエストの受信に使える)。
                    outbox.emplace_back();
                    Request& rq = outbox.back();
                    rq.io_id = io_id;
                    rq.fd = fd;
                    rq.gen = c.gen;
                    std::memcpy(rq.px, c.buf, kReqBytes);
                    c.in_flight = true;
                    c.filled = 0;
                }
            }
        }

        if (use_queue) {
            // ---- キュー版: この回に溜まった Request をまとめて共有キューへ積む ----
            // ロックは1回だけ取る(件数ぶん取ると競合が増える)。起こすのはロックを手放した後。
            if (!outbox.empty()) {
                auto now = Clock::now();
                for (Request& r : outbox) r.t_enq = now;
                size_t depth;
                {
                    std::lock_guard<std::mutex> lk(g_q_mtx);
                    for (Request& r : outbox) g_q.push_back(std::move(r));
                    depth = g_q.size();
                }
                atomic_max(g_qdepth_max, depth);
                if (outbox.size() == 1) g_q_cv.notify_one(); else g_q_cv.notify_all();
            }
        } else {
            // ---- インライン版: この回に溜まった分を max_batch 件ずつまとめて推論 → 返信 ----
            // max_batch=1 なら 1件ずつ(M3までと同じ動き)。ready の順に先頭から切り出す
            fails.clear();
            for (size_t off = 0; off < ready.size(); off += max_batch) {
                size_t b = std::min(max_batch, ready.size() - off);
                run_batch(session, conns, ready.data() + off, b, fails);
            }
            for (int fd : fails) close_conn(w, conns, fd);   // 失敗した接続は最後にまとめて閉じる
        }
    }
}

// ── 推論ワーカーの動作(キュー版のみ) ─────────────────────────────
// 共有キューから最大 max_batch 件を取り出し → 1回の Run() → 持ち主ごとに返信を預けて起こす。
// この thread は fd にも Conn にも触らない(触るのは持ち主の I/O worker だけ)。
static void infer_loop(Ort::Session& session, size_t max_batch,
                       std::vector<std::unique_ptr<Worker>>& workers) {
    std::vector<Request> batch;
    std::vector<const uint8_t*> px;
    std::vector<float> logits;
    std::vector<char> touched(workers.size());   // この回に返信を預けた I/O worker(起こす対象)
    while (!g_stop) {
        batch.clear();
        {
            std::unique_lock<std::mutex> lk(g_q_mtx);
            // 空なら寝る。述語つき wait は「spurious wakeup」と「起こされる前に入っていた」の両方に強い(M3)
            g_q_cv.wait(lk, [] { return !g_q.empty() || g_stop; });
            if (g_q.empty()) continue;
            // 溜まっている分だけ取る(待ち時間は足さない=日和見バッチ)。取り出すのはロックの中、Run() は外。
            size_t take = std::min(max_batch, g_q.size());
            for (size_t i = 0; i < take; ++i) {
                batch.push_back(std::move(g_q.front()));
                g_q.pop_front();
            }
        }

        // キューで待った時間(push → pop)を記録
        auto t_deq = Clock::now();
        for (const Request& r : batch) {
            uint64_t wait = std::chrono::duration_cast<std::chrono::nanoseconds>(t_deq - r.t_enq).count();
            g_qwait_ns += wait;
            atomic_max(g_qwait_max_ns, wait);
        }

        px.clear();
        for (const Request& r : batch) px.push_back(r.px);
        bool ok = true;
        try {
            forward_batch(session, px.data(), batch.size(), logits);
        } catch (const Ort::Exception& e) {
            std::cerr << "Run() failed: " << e.what() << std::endl;
            ok = false;   // 返信は ok=false にして、持ち主に接続を閉じてもらう
        }

        // i 番目の入力 ↔ logits の i 行目 ↔ batch[i] の持ち主。添字でそろえる。
        std::fill(touched.begin(), touched.end(), 0);
        for (size_t i = 0; i < batch.size(); ++i) {
            Response resp;
            resp.fd = batch[i].fd;
            resp.gen = batch[i].gen;
            resp.ok = ok;
            if (ok) make_resp(logits.data() + i * 10, resp.resp);
            Worker& w = *workers[batch[i].io_id];
            {
                std::lock_guard<std::mutex> lk(w.mtx);
                w.done.push(resp);
            }
            touched[batch[i].io_id] = 1;
        }
        // 返信を預けた I/O worker を起こす(1回のバッチにつき、worker ごとに1回だけ書く)
        for (size_t k = 0; k < touched.size(); ++k) {
            if (!touched[k]) continue;
            uint64_t one = 1;
            if (write(workers[k]->event_fd, &one, sizeof(one)) < 0) {}
        }
    }
}

int main(int argc, char** argv) {
    // ===== 引数 =====
    if (argc < 2) {
        std::cerr << "usage: " << argv[0]
                  << " <model.onnx> [workers] [intra_op_threads] [per_worker_session=0|1] [max_batch=1] [infer_threads=0]\n";
        return 1;
    }
    const char* model_path = argv[1];
    int kWorkers = (argc > 2) ? std::atoi(argv[2]) : 8;   // I/O ワーカースレッド数(インライン版では推論も担当)
    int intra    = (argc > 3) ? std::atoi(argv[3]) : 1;   // ORT 1回の Run() 内部で使うスレッド数
    // 実験用: 1 なら worker ごとに専用の Session を作る(重みは worker 数ぶんのコピーになる)。
    // 0(既定)なら全 worker で1個を共有。共有による競合が Run() を遅くしているかを切り分けるためのもの
    int per_worker = (argc > 4) ? std::atoi(argv[4]) : 0;
    // M4: 1回の Run() にまとめる最大件数。1(既定)ならバッチ化なし。
    // 2以上にはバッチ軸が可変のモデル(mlp_dyn.onnx など)が必要(固定 batch=1 のモデルだと Run() が失敗する)
    size_t max_batch = (argc > 5) ? (size_t)std::max(1, std::atoi(argv[5])) : 1;
    // M5: 推論専用 worker の本数。0(既定)ならインライン。1以上なら共有 Request Queue 経由で推論する
    int infer_threads = (argc > 6) ? std::max(0, std::atoi(argv[6])) : 0;
    bool use_queue = infer_threads > 0;
    if (use_queue && per_worker) {
        std::cerr << "note: キュー版では Session は共有のみ(per_worker_session は無視)\n";
        per_worker = 0;
    }

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
        g_requests = 0; g_infer_ns = 0; g_batches = 0;   // ウォームアップ分の統計は捨てる
    }

    // per_worker=1 のときだけ使う、worker 専用の Session 群(共有時は空のまま)
    std::vector<std::unique_ptr<Ort::Session>> own_sessions;
    if (per_worker) {
        uint8_t zero[kReqBytes] = {}, resp[kRespBytes];
        for (int i = 0; i < kWorkers; ++i) {
            own_sessions.push_back(std::make_unique<Ort::Session>(env, model_path, so));
            for (int k = 0; k < 20; ++k) run_inference(*own_sessions[i], zero, resp);
        }
        g_requests = 0; g_infer_ns = 0; g_batches = 0;
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

    // ===== I/O ワーカー起動 =====
    std::vector<std::unique_ptr<Worker>> workers;   // Worker は mutex を含みコピー不可 → ポインタで持つ
    std::vector<std::thread> threads;
    // ★ 推論 worker は workers[io_id] を添字で引くので、先に全 Worker を作り終えてからスレッドを起動する
    //    (vector が再確保されて参照が壊れるのを防ぐ)
    for (int i = 0; i < kWorkers; ++i) {
        auto w = std::make_unique<Worker>();
        w->epoll_fd = epoll_create1(0);             // このワーカー専用の epoll
        w->event_fd = eventfd(0, EFD_NONBLOCK);     // 「起きろ」用の通知 fd
        // 自分の event_fd を自分の epoll に登録(acceptor / 推論 worker の合図で起きられるように)
        epoll_event ev{};
        ev.events = EPOLLIN;
        ev.data.fd = w->event_fd;
        epoll_ctl(w->epoll_fd, EPOLL_CTL_ADD, w->event_fd, &ev);
        workers.push_back(std::move(w));
    }
    for (int i = 0; i < kWorkers; ++i) {
        // std::ref: スレッドへの引数は既定でコピーされる。Worker も Session もコピー不可/したくないので
        // 「参照で渡せ」の目印(M3の FdQueue と同じ)。session は全ワーカーが同じ1個を指す。
        Ort::Session& s = per_worker ? *own_sessions[i] : session;   // 共有 or 専用
        threads.emplace_back(worker_loop, std::ref(*workers[i]), std::ref(s), max_batch, i, use_queue);
    }
    // ===== 推論 worker 起動(キュー版のみ) =====
    for (int i = 0; i < infer_threads; ++i)
        threads.emplace_back(infer_loop, std::ref(session), max_batch, std::ref(workers));

    // ===== メイン = acceptor(listen_fd だけを epoll で監視) =====
    int ep = epoll_create1(0);
    epoll_event ev{};
    ev.events = EPOLLIN;
    ev.data.fd = listen_fd;
    epoll_ctl(ep, EPOLL_CTL_ADD, listen_fd, &ev);

    // endl = 改行 + flush。起動完了をすぐ表示させる
    std::cout << "infer_server: port 8080, model=" << model_path
              << ", workers=" << kWorkers << ", intra_op_threads=" << intra
              << ", session=" << (per_worker ? "per-worker" : "shared")
              << ", max_batch=" << max_batch
              << ", infer_threads=" << infer_threads << (use_queue ? " (queue)" : " (inline)") << std::endl;

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
    // Run() は「バッチ1回」ごとの時間。リクエスト1件あたりに直すと Run()/avg_batch
    uint64_t n = g_requests.load(), nb = g_batches.load();
    std::cout << "requests=" << n << "  avg ORT Run()="
              << (nb ? g_infer_ns.load() / nb / 1000.0 : 0.0) << " us"
              << "  batches=" << nb << "  avg_batch=" << (nb ? (double)n / nb : 0.0)
              << "  Run()/request=" << (n ? g_infer_ns.load() / n / 1000.0 : 0.0) << " us" << std::endl;
    if (use_queue) {
        // キュー待ち時間 = push → pop。p50 の内訳(I/O・キュー待ち・Run())を分けて語るための数字
        std::cout << "queue: avg_wait=" << (n ? g_qwait_ns.load() / n / 1000.0 : 0.0) << " us"
                  << "  max_wait=" << g_qwait_max_ns.load() / 1000.0 << " us"
                  << "  max_depth=" << g_qdepth_max.load()
                  << "  stale_dropped=" << g_stale.load()
                  << "  violations=" << g_violations.load() << std::endl;
    }
    // ワーカーは epoll_wait(-1) / cv.wait で寝ているので join できない。_exit で強制終了する。
    // ※ _exit は iostream のバッファを flush しない。上の endl はそのために必要(前回のバグの原因)
    _exit(0);
}
