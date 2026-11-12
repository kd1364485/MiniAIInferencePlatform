// syscalls.cpp — strace / perf で観察するための、1 つの処理だけを繰り返すプログラム
//
// bench は「何 µs かかるか」を測る。これは「どんなシステムコールが何回出るか」を見るためのもの。
// 1 つのモードだけを N 回繰り返すので、strace -c の合計を N で割れば 1 回あたりの回数が出る。
//
// 使い方: ./syscalls <mode> [N]
//   cpuinfo   logical_cpus()        /proc/cpuinfo を開いて数える
//   sysconf   logical_cpus_online() sysconf で得る
//   task      全プロセスのスレッド数を /proc/<pid>/task/ を数えて得る
//   status    全プロセスのスレッド数を /proc/<pid>/status の Threads: 行から得る
//   meminfo   meminfo_kb("MemAvailable")
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "collect.h"

static volatile long g_sink = 0;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <cpuinfo|sysconf|task|status|meminfo> [N]\n", argv[0]);
        return 2;
    }
    const std::string mode = argv[1];
    const int n = argc > 2 ? std::atoi(argv[2]) : 100;
    using namespace probe;

    // プロセス一覧は最初に 1 回だけ作る(その分のシステムコールは計測に混ざる。N を大きくして薄める)
    std::vector<std::string> pids;
    if (mode == "task" || mode == "status") {
        for (const auto& e : fs::directory_iterator("/proc")) {
            std::string name = e.path().filename().string();
            if (is_pid_name(name)) pids.push_back(name);
        }
        std::fprintf(stderr, "processes: %zu, sweeps: %d\n", pids.size(), n);
    }

    for (int i = 0; i < n; ++i) {
        if (mode == "cpuinfo") {
            g_sink += logical_cpus();
        } else if (mode == "sysconf") {
            g_sink += logical_cpus_online();
        } else if (mode == "meminfo") {
            g_sink += meminfo_kb("MemAvailable");
        } else if (mode == "task") {
            for (const auto& p : pids) g_sink += threads_by_task(p);
        } else if (mode == "status") {
            for (const auto& p : pids) g_sink += threads_by_status(p);
        } else {
            std::fprintf(stderr, "unknown mode: %s\n", mode.c_str());
            return 2;
        }
    }
    return 0;
}
