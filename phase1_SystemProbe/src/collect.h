// collect.h — /proc を実際に開いて、parse.h で解釈する層
//
// 構成: collect.h(ファイルを開く) → parse.h(文字列を解釈する)
// ファイルを開くたびに open / read / close のシステムコールが発生する。
// そのコストと解釈のコストを分けて測れるように、層を分けてある(bench/bench.cpp)。
#pragma once

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include <unistd.h>

#include "parse.h"

namespace probe {

namespace fs = std::filesystem;

// /proc/<pid>/task/ の中身を数える方式でスレッド数を得る。
// 予想: ディレクトリ一覧は遅いだろう。 → 測ったら逆で、status を読む方式より速かった。
// 全プロセス 1 周で task 137〜150µs 対 status 225〜246µs(bench の [3])。
// システムコールの回数は task のほうが多い(約 7 回 対 約 3 回、strace -c)のに速い。
// ストリームを外した素のシステムコールでも status は約 2.4 倍遅い(bench の [6])ので、
// 差の主因は、status を読むたびにカーネルが文章を組み立てるコスト(カーネル側)。
// カーネルの中のどこが重いかは、perf でカーネル側が見えないため未確認。
inline int threads_by_task(const std::string& pid) {
    int count = 0;
    try {
        for (const auto& e : fs::directory_iterator("/proc/" + pid + "/task")) {
            (void)e;
            ++count;
        }
    } catch (const std::exception& e) {
        // プロセスが途中で終了すると読めなくなる(正常な競合)
        std::cerr << "Error reading /proc/" << pid << "/task: " << e.what() << "\n";
    }
    return count;
}

// /proc/<pid>/status の "Threads:" 行を読む方式
inline int threads_by_status(const std::string& pid) {
    std::ifstream file("/proc/" + pid + "/status");
    if (!file) return 0;
    return parse_threads(file);
}

inline char process_state(const std::string& pid) {
    std::ifstream file("/proc/" + pid + "/status");
    return parse_state(file);
}

// 読めなければ空文字が返る
inline std::string process_name(const std::string& pid) {
    std::ifstream file("/proc/" + pid + "/comm");
    std::string name;
    std::getline(file, name);
    return name;
}

inline long meminfo_kb(const std::string& key) {
    std::ifstream file("/proc/meminfo");
    return parse_meminfo_kb(file, key);
}

inline std::string cpu_model_name() {
    std::ifstream file("/proc/cpuinfo");
    return parse_cpu_model(file);
}

inline int logical_cpus() {
    std::ifstream file("/proc/cpuinfo");
    return parse_logical_cpus(file);
}

// 同じ値(オンラインの論理 CPU 数)を、cpuinfo を読まずに得る方法。
// bench の [1] で logical_cpus が約 160µs かかり、そのうち解釈は約 8µs だった。
// 時間のほとんどは、カーネルが全 CPU 分の cpuinfo を組み立てるコスト。
// sysconf は glibc が /sys/devices/system/cpu/online を読んで答えるので、それを避けられる。
inline int logical_cpus_online() {
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? static_cast<int>(n) : 0;
}

inline CpuTimes read_cpu_times() {
    std::ifstream file("/proc/stat");
    return parse_cpu_times(file);
}

inline std::vector<NetDev> read_netdev() {
    std::ifstream file("/proc/net/dev");
    return parse_netdev(file);
}

inline std::vector<DiskStat> read_diskstats() {
    std::ifstream file("/proc/diskstats");
    return parse_diskstats(file);
}

}  // namespace probe
