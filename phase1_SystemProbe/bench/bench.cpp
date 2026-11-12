// bench.cpp — SystemProbe の各収集処理の所要時間を測る
//
// 知りたいこと:
//   1. 各収集処理は 1 回あたり何 µs かかるか(平均 / p50 / p95 / p99)
//   2. その時間のうち、「ファイルを開いて読む」コストと「文字列を解釈する」コストの割合
//      → 同じ内容を、実ファイルから読む場合と、メモリ上の文字列から読む場合で比べる
//   3. スレッド数を数える 2 つの方式(/proc/<pid>/task を数える vs status を読む)はどちらが速いか
//
// 使い方: ./bench [反復回数(既定 2000)]
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

#include <sys/resource.h>
#include <sys/syscall.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>

#include "collect.h"

using Clock = std::chrono::steady_clock;

struct Stats {
    double mean_us, p50_us, p95_us, p99_us, max_us;
};

static Stats summarize(std::vector<double>& us) {
    std::sort(us.begin(), us.end());
    auto pct = [&](double p) { return us[static_cast<size_t>(p * (us.size() - 1))]; };
    double sum = 0;
    for (double v : us) sum += v;
    return {sum / us.size(), pct(0.50), pct(0.95), pct(0.99), us.back()};
}

// fn を iters 回呼び、1 回ごとの時間(µs)を測る。最初に数回ウォームアップする。
static Stats measure(int iters, const std::function<void()>& fn) {
    for (int i = 0; i < 20; ++i) fn();
    std::vector<double> us;
    us.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        auto t0 = Clock::now();
        fn();
        auto t1 = Clock::now();
        us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    return summarize(us);
}

static void print_row(const char* name, const Stats& s) {
    std::printf("  %-34s mean %9.1f  p50 %9.1f  p95 %9.1f  p99 %9.1f  max %10.1f  (µs)\n",
                name, s.mean_us, s.p50_us, s.p95_us, s.p99_us, s.max_us);
}

static std::string slurp(const char* path) {
    std::ifstream f(path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// 防止: 結果を使わないとコンパイラが処理ごと消すことがあるので、値を貯めて最後に出力する
static volatile long g_sink = 0;

int main(int argc, char** argv) {
    int iters = argc > 1 ? std::atoi(argv[1]) : 2000;
    if (iters < 100) iters = 100;
    using namespace probe;

    std::printf("SystemProbe benchmark  (iterations per case: %d)\n\n", iters);

    // ---- 1. 実ファイルから読む(open + read + close + 解釈) ----
    std::printf("[1] /proc を実際に開いて読む(open/read/close + parse)\n");
    print_row("meminfo_kb(MemAvailable)", measure(iters, [] { g_sink += meminfo_kb("MemAvailable"); }));
    print_row("cpu_model_name", measure(iters, [] { g_sink += cpu_model_name().size(); }));
    print_row("logical_cpus (cpuinfo)", measure(iters, [] { g_sink += logical_cpus(); }));
    print_row("logical_cpus_online (sysconf)", measure(iters, [] { g_sink += logical_cpus_online(); }));
    print_row("read_cpu_times", measure(iters, [] { g_sink += read_cpu_times().user; }));
    print_row("read_netdev", measure(iters, [] { g_sink += read_netdev().size(); }));
    print_row("read_diskstats", measure(iters, [] { g_sink += read_diskstats().size(); }));

    // ---- 2. メモリ上の同じ内容から解釈だけする(ファイルを開く分を除く) ----
    std::printf("\n[2] 同じ内容をメモリ上の文字列から解釈するだけ(parse のみ)\n");
    const std::string meminfo = slurp("/proc/meminfo");
    const std::string cpuinfo = slurp("/proc/cpuinfo");
    const std::string stat = slurp("/proc/stat");
    const std::string netdev = slurp("/proc/net/dev");
    const std::string diskstats = slurp("/proc/diskstats");
    print_row("parse_meminfo_kb(MemAvailable)", measure(iters, [&] {
                  std::istringstream in(meminfo);
                  g_sink += parse_meminfo_kb(in, "MemAvailable");
              }));
    print_row("parse_cpu_model", measure(iters, [&] {
                  std::istringstream in(cpuinfo);
                  g_sink += parse_cpu_model(in).size();
              }));
    print_row("parse_logical_cpus", measure(iters, [&] {
                  std::istringstream in(cpuinfo);
                  g_sink += parse_logical_cpus(in);
              }));
    print_row("parse_cpu_times", measure(iters, [&] {
                  std::istringstream in(stat);
                  g_sink += parse_cpu_times(in).user;
              }));
    print_row("parse_netdev", measure(iters, [&] {
                  std::istringstream in(netdev);
                  g_sink += parse_netdev(in).size();
              }));
    print_row("parse_diskstats", measure(iters, [&] {
                  std::istringstream in(diskstats);
                  g_sink += parse_diskstats(in).size();
              }));

    // ---- 3. スレッド数の 2 方式を、全プロセスに対して比べる ----
    std::printf("\n[3] 全プロセスのスレッド数を数える 2 方式の比較\n");
    std::vector<std::string> pids;
    for (const auto& e : fs::directory_iterator("/proc")) {
        std::string n = e.path().filename().string();
        if (is_pid_name(n)) pids.push_back(n);
    }
    std::printf("  対象プロセス数: %zu\n", pids.size());
    const int scans = std::max(20, iters / 40);
    print_row("by_task  (task/ を数える)", measure(scans, [&] {
                  long s = 0;
                  for (const auto& p : pids) s += threads_by_task(p);
                  g_sink += s;
              }));
    print_row("by_status(Threads: 行を読む)", measure(scans, [&] {
                  long s = 0;
                  for (const auto& p : pids) s += threads_by_status(p);
                  g_sink += s;
              }));
    std::printf("  (1 回 = 全プロセスを 1 周。反復 %d 回。上の µs は 1 周あたり)\n", scans);

    // ---- 4. 全体: SystemProbe が 1 回の表示で /proc を読む総量 ----
    std::printf("\n[4] 1 周の収集(メモリ・CPU・全プロセス・ネットワーク・ディスク。500ms の待ちは除く)\n");
    print_row("collect_all", measure(scans, [&] {
                  g_sink += meminfo_kb("MemTotal") + meminfo_kb("MemAvailable") + meminfo_kb("MemFree");
                  g_sink += cpu_model_name().size() + logical_cpus();
                  for (const auto& e : fs::directory_iterator("/proc")) {
                      std::string n = e.path().filename().string();
                      if (!is_pid_name(n)) continue;
                      g_sink += process_state(n);
                      g_sink += threads_by_task(n) + threads_by_status(n);
                      g_sink += process_name(n).size();
                  }
                  g_sink += read_cpu_times().user + read_netdev().size() + read_diskstats().size();
              }));

    // ---- 5. 時間の内訳: ユーザー空間 vs カーネル(getrusage) ----
    // 「ファイル読みが重い」の裏付け。システムコールの中(カーネル)で使った時間は sys に入る。
    // strace / perf が使えない環境でも測れる。1 回あたりの µs に直して表示する。
    std::printf("\n[5] 1 回あたりの CPU 時間の内訳(getrusage の user と sys)\n");
    auto split = [&](const char* name, int n, const std::function<void()>& fn) {
        for (int i = 0; i < 20; ++i) fn();
        rusage a{}, b{};
        getrusage(RUSAGE_SELF, &a);
        for (int i = 0; i < n; ++i) fn();
        getrusage(RUSAGE_SELF, &b);
        auto us = [](const timeval& x, const timeval& y) {
            return (y.tv_sec - x.tv_sec) * 1e6 + (y.tv_usec - x.tv_usec);
        };
        double user = us(a.ru_utime, b.ru_utime) / n;
        double sys = us(a.ru_stime, b.ru_stime) / n;
        double total = user + sys;
        std::printf("  %-34s user %8.1f  sys %8.1f  (sys %4.0f%%)  (µs/回)\n", name, user, sys,
                    total > 0 ? 100.0 * sys / total : 0.0);
    };
    split("logical_cpus (cpuinfo)", iters, [] { g_sink += logical_cpus(); });
    split("logical_cpus_online (sysconf)", iters, [] { g_sink += logical_cpus_online(); });
    split("parse_logical_cpus (メモリ上)", iters, [&] {
        std::istringstream in(cpuinfo);
        g_sink += parse_logical_cpus(in);
    });
    // 注意: getrusage は CPU 時間を一定間隔のサンプリングで数えるため、1 回が短い処理
    // (全プロセス 1 周など)を少ない反復で測ると、user と sys の割合が実行ごとに大きくぶれる
    // (試したところ sys が 42%〜100% で不安定だった)。ここには、割合が安定した項目だけを載せている。

    // ---- 6. C++ のストリームを外して、素のシステムコールだけで比べる ----
    // [3] では task/ を数える方式のほうが速かった。strace で数えると、システムコールの回数は
    // task が 1 プロセスあたり約 7 回、status が約 3 回で、回数では逆になる。
    // では差はどこから来るのか。ifstream / directory_iterator(ユーザー空間の処理)を外し、
    // open + read(または getdents64) + close だけで測り直して、カーネル側の差か C++ 側の差かを切り分ける。
    std::printf("\n[6] 素のシステムコールだけで、全プロセスを 1 周(C++ のストリームなし)\n");
    auto raw_status = [](const std::string& pid) -> int {
        std::string path = "/proc/" + pid + "/status";
        int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) return 0;
        char buf[8192];
        ssize_t n = ::read(fd, buf, sizeof(buf) - 1);
        ::close(fd);
        if (n <= 0) return 0;
        buf[n] = '\0';
        const char* p = std::strstr(buf, "Threads:");
        return p ? std::atoi(p + 8) : 0;
    };
    auto raw_task = [](const std::string& pid) -> int {
        std::string path = "/proc/" + pid + "/task";
        int fd = ::open(path.c_str(), O_RDONLY | O_DIRECTORY);
        if (fd < 0) return 0;
        char buf[8192];
        int count = 0;
        for (;;) {
            long n = ::syscall(SYS_getdents64, fd, buf, sizeof(buf));
            if (n <= 0) break;
            for (long off = 0; off < n;) {
                // struct linux_dirent64 { u64 d_ino; s64 d_off; u16 d_reclen; u8 d_type; char d_name[]; }
                unsigned short reclen;
                std::memcpy(&reclen, buf + off + 16, sizeof(reclen));
                const char* name = buf + off + 19;
                if (std::strcmp(name, ".") != 0 && std::strcmp(name, "..") != 0) ++count;
                off += reclen;
            }
        }
        ::close(fd);
        return count;
    };
    // 正しさの確認: 素のシステムコール版と C++ 版で、合計が一致するか
    long sum_raw_status = 0, sum_raw_task = 0, sum_cpp_status = 0, sum_cpp_task = 0;
    for (const auto& p : pids) {
        sum_raw_status += raw_status(p);
        sum_raw_task += raw_task(p);
        sum_cpp_status += threads_by_status(p);
        sum_cpp_task += threads_by_task(p);
    }
    std::printf("  合計スレッド数(確認): raw status=%ld  raw task=%ld  C++ status=%ld  C++ task=%ld\n",
                sum_raw_status, sum_raw_task, sum_cpp_status, sum_cpp_task);
    print_row("raw task   (open+getdents64+close)", measure(scans, [&] {
                  long s = 0;
                  for (const auto& p : pids) s += raw_task(p);
                  g_sink += s;
              }));
    print_row("raw status (open+read+close)", measure(scans, [&] {
                  long s = 0;
                  for (const auto& p : pids) s += raw_status(p);
                  g_sink += s;
              }));
    return 0;
}