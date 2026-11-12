// parse.h — /proc のテキストを読み解く部分だけを切り出したもの
//
// 「ファイルを開く」処理(collect.h)と「文字列を解釈する」処理(ここ)を分けた。
// 解釈する側は std::istream を受け取るだけなので、/proc の中身を模した文字列で
// テストできる(test/parse_test.cpp)。実機の /proc に依存しないので結果が毎回同じになる。
#pragma once

#include <algorithm>
#include <cctype>
#include <istream>
#include <sstream>
#include <string>
#include <vector>

namespace probe {

// ===== Process =====

// 数字だけの名前か判定(/proc 直下からプロセスのディレクトリを選ぶのに使う)
// 空文字は false。std::all_of は空の範囲で true を返すので、明示的に除く。
inline bool is_pid_name(const std::string& name) {
    return !name.empty() &&
           std::all_of(name.begin(), name.end(),
                       [](unsigned char c) { return std::isdigit(c) != 0; });
}

// /proc/<pid>/status の "Threads:" 行の値。見つからなければ 0。
inline int parse_threads(std::istream& in) {
    std::string line;
    while (std::getline(in, line)) {
        if (line.starts_with("Threads:")) {
            size_t i = line.find_first_not_of(" \t", 8);  // "Threads:" の後
            if (i != std::string::npos) {
                return std::stoi(line.substr(i));
            }
        }
    }
    return 0;
}

// /proc/<pid>/status の "State:" 行から状態コード(S, R, D ...)を1文字取る。なければ '?'。
inline char parse_state(std::istream& in) {
    std::string line;
    while (std::getline(in, line)) {
        if (line.starts_with("State:")) {
            size_t i = line.find_first_not_of(" \t", 6);  // "State:" の後
            return (i == std::string::npos) ? '?' : line[i];
        }
    }
    return '?';
}

// ===== Memory (/proc/meminfo) =====

// 各行は "Key:    value kB" 形式。指定したキーの値(kB)を返す。見つからなければ -1。
// HugePages 系など、単位のない行もある。
inline long parse_meminfo_kb(std::istream& in, const std::string& key) {
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream iss(line);
        std::string k;
        long value = 0;
        std::string unit;
        iss >> k >> value >> unit;
        if (k == key + ":") {
            return value;
        }
    }
    return -1;
}

// ===== CPU (/proc/cpuinfo, /proc/stat) =====

// "model name : Intel(R) ..." の右側。なければ "unknown"。
inline std::string parse_cpu_model(std::istream& in) {
    std::string line;
    while (std::getline(in, line)) {
        if (line.starts_with("model name")) {
            size_t colon = line.find(':');
            if (colon == std::string::npos) continue;
            size_t i = line.find_first_not_of(" \t", colon + 1);
            return (i == std::string::npos) ? "" : line.substr(i);
        }
    }
    return "unknown";
}

// "processor : N" 行の数 = 論理 CPU 数
inline int parse_logical_cpus(std::istream& in) {
    std::string line;
    int count = 0;
    while (std::getline(in, line)) {
        if (line.starts_with("processor")) ++count;
    }
    return count;
}

// /proc/stat の1行目 "cpu  user nice system idle iowait irq softirq steal" は、
// 起動からの累積時間(jiffies)。瞬間値ではなくカウンタ。
struct CpuTimes {
    long user = 0, nice = 0, system = 0, idle = 0;
    long iowait = 0, irq = 0, softirq = 0, steal = 0;
};

inline CpuTimes parse_cpu_times(std::istream& in) {
    std::string line;
    std::getline(in, line);  // 1行目だけ使う
    std::istringstream iss(line);
    std::string cpu;
    CpuTimes t;
    iss >> cpu >> t.user >> t.nice >> t.system >> t.idle >> t.iowait >> t.irq >>
        t.softirq >> t.steal;
    return t;
}

inline long total_time(const CpuTimes& t) {
    return t.user + t.nice + t.system + t.idle + t.iowait + t.irq + t.softirq + t.steal;
}

// iowait は「CPU が暇で、ディスク待ちだった時間」なので idle に含める
inline long idle_time(const CpuTimes& t) { return t.idle + t.iowait; }

// カウンタは累積なので、1回の読み取りでは使用率が分からない。2回読んで差分を取る。
inline double cpu_usage_percent(const CpuTimes& a, const CpuTimes& b) {
    long total_d = total_time(b) - total_time(a);
    long idle_d = idle_time(b) - idle_time(a);
    if (total_d == 0) return 0.0;  // 変化なし
    return 100.0 * (total_d - idle_d) / total_d;
}

// ===== Network (/proc/net/dev) =====
// 先頭2行はヘッダ。3行目以降: "   eth0: rx_bytes rx_packets ... tx_bytes tx_packets ..."
// ':' の左がインターフェイス名(先頭に空白あり)、右が数値 16 個(前半 8 個が受信、後半 8 個が送信)。

struct NetDev {
    std::string name;
    long rx_bytes = 0;
    long tx_bytes = 0;
};

// 1行を解釈して out に入れる。失敗したら false。
inline bool parse_netdev_line(const std::string& line, NetDev& out) {
    size_t colon = line.find(':');
    if (colon == std::string::npos) return false;
    std::string name = line.substr(0, colon);
    size_t first = name.find_first_not_of(" \t");
    if (first == std::string::npos) return false;
    out.name = name.substr(first);
    // 使うのは 1 個目(rx_bytes)と 9 個目(tx_bytes)だけ。読み飛ばすために全部読む。
    std::istringstream iss(line.substr(colon + 1));
    long rx_packets, rx_errs, rx_drop, rx_fifo, rx_frame, rx_compressed, rx_multicast;
    long tx_packets, tx_errs, tx_drop, tx_fifo, tx_colls, tx_carrier, tx_compressed;
    iss >> out.rx_bytes >> rx_packets >> rx_errs >> rx_drop >> rx_fifo >> rx_frame >>
        rx_compressed >> rx_multicast >> out.tx_bytes >> tx_packets >> tx_errs >>
        tx_drop >> tx_fifo >> tx_colls >> tx_carrier >> tx_compressed;
    return !iss.fail();
}

inline std::vector<NetDev> parse_netdev(std::istream& in) {
    std::vector<NetDev> devs;
    std::string line;
    std::getline(in, line);  // ヘッダ 1 行目
    std::getline(in, line);  // ヘッダ 2 行目
    while (std::getline(in, line)) {
        NetDev d;
        if (parse_netdev_line(line, d)) devs.push_back(d);
    }
    return devs;
}

// 名前でインターフェイスを探す(出現順は変わりうるので、位置ではなく名前で対応させる)
inline const NetDev* find_dev(const std::vector<NetDev>& devs, const std::string& name) {
    for (const auto& d : devs) {
        if (d.name == name) return &d;
    }
    return nullptr;
}

// ===== Disk (/proc/diskstats) =====
// 各行: major minor name reads reads_merged sectors_read ms_reading
//       writes writes_merged sectors_written ...  (1 セクタ = 512 バイト)

struct DiskStat {
    std::string name;
    long reads = 0;
    long sectors_read = 0;
    long sectors_written = 0;
};

inline std::vector<DiskStat> parse_diskstats(std::istream& in) {
    std::vector<DiskStat> disks;
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream iss(line);
        int major, minor;
        DiskStat d;
        long reads_merged, ms_reading, writes_completed, writes_merged;
        iss >> major >> minor >> d.name >> d.reads >> reads_merged >> d.sectors_read >>
            ms_reading >> writes_completed >> writes_merged >> d.sectors_written;
        if (!iss.fail()) disks.push_back(d);
    }
    return disks;
}

}  // namespace probe
