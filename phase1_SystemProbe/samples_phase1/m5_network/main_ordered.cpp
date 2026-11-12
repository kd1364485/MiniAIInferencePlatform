// ★ samples/phase1/m5_network/main.cpp を
//   SystemProbe/src/main.cpp の並び順に合わせて並べ替えた参照用コピー。
//   新規追加部分は【M5 NEW】とマークしてある。

#include <algorithm>
#include <chrono>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

// ===== Thread =====

int threads_by_task(const std::string& pid) {
    int count = 0;
    try {
        for ([[maybe_unused]] const auto& e : fs::directory_iterator("/proc/" + pid + "/task")) {
            ++count;
        }
    } catch (const fs::filesystem_error&) {
        // 走査中にプロセスが消えた
    }
    return count;
}

int threads_by_status(const std::string& pid) {
    std::ifstream file("/proc/" + pid + "/status");
    if (!file) return 0;
    std::string line;
    while (std::getline(file, line)) {
        if (line.starts_with("Threads:")) {
            size_t i = line.find_first_not_of(" \t", 8);
            if (i != std::string::npos) {
                return std::stoi(line.substr(i));
            }
        }
    }
    return 0;
}

std::string process_name(const std::string& pid) {
    std::ifstream file("/proc/" + pid + "/comm");
    std::string name;
    std::getline(file, name);
    return name;
}

// ===== Process (/proc/<pid>/) =====

bool is_pid_name(const std::string& name) {
    return std::all_of(name.begin(), name.end(), ::isdigit);
}

char process_state(const std::string& pid) {
    std::ifstream file("/proc/" + pid + "/status");
    std::string line;
    while (std::getline(file, line)) {
        if (line.starts_with("State:")) {
            size_t i = line.find_first_not_of(" \t", 6);
            return (i == std::string::npos) ? '?' : line[i];
        }
    }
    return '?';
}

// ===== Memory (/proc/meminfo) =====

long meminfo_kb(const std::string& key) {
    std::ifstream file("/proc/meminfo");
    std::string line;
    while (std::getline(file, line)) {
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

std::string cpu_model_name() {
    std::ifstream file("/proc/cpuinfo");
    std::string line;
    while (std::getline(file, line)) {
        if (line.starts_with("model name")) {
            return line.substr(line.find(':') + 2);
        }
    }
    return "unknown";
}

int logical_cpus() {
    std::ifstream file("/proc/cpuinfo");
    std::string line;
    int count = 0;
    while (std::getline(file, line)) {
        if (line.starts_with("processor")) {
            ++count;
        }
    }
    return count;
}

struct CpuTimes {
    long user = 0, nice = 0, system = 0, idle = 0;
    long iowait = 0, irq = 0, softirq = 0, steal = 0;
};

CpuTimes read_cpu_times() {
    std::ifstream file("/proc/stat");
    std::string line;
    std::getline(file, line);
    std::istringstream iss(line);
    std::string cpu;
    CpuTimes t;
    iss >> cpu >> t.user >> t.nice >> t.system >> t.idle
        >> t.iowait >> t.irq >> t.softirq >> t.steal;
    return t;
}

long total_time(const CpuTimes& t) {
    return t.user + t.nice + t.system + t.idle
         + t.iowait + t.irq + t.softirq + t.steal;
}

long idle_time(const CpuTimes& t) {
    return t.idle + t.iowait;
}

double cpu_usage_percent(const CpuTimes& a, const CpuTimes& b) {
    long total_d = total_time(b) - total_time(a);
    long idle_d  = idle_time(b)  - idle_time(a);
    if (total_d == 0) return 0.0;
    return 100.0 * (total_d - idle_d) / total_d;
}

// ===== Network (/proc/net/dev) =====   【M5 NEW：ここから新規】
//
// /proc/net/dev の形式:
//   先頭2行はヘッダ（読み飛ばす）
//   3行目以降: "    eth0: rx_bytes rx_packets ... tx_bytes tx_packets ..."
//   ':' の左側がインターフェース名（先頭に空白あり）、右側に数値16個。
//   左8個=Receive、右8個=Transmit。値は起動からの累積カウンタ。

struct NetDev {
    std::string name;
    long rx_bytes = 0; // 受信バイト数(累積)
    long tx_bytes = 0; // 送信バイト数(累積)
};

// 1行をパースして out に入れる。失敗したら false。
bool parse_netdev_line(const std::string& line, NetDev& out) {
    size_t colon = line.find(':');
    if (colon == std::string::npos) return false;

    // ':'の左側が名前。"    eth0" のように先頭空白が付くので trim する
    std::string name = line.substr(0, colon);
    size_t first = name.find_first_not_of(" \t");
    if (first == std::string::npos) return false;
    out.name = name.substr(first);

    // ':'の右側は数値16個。使うのは1個目(rx_bytes)と9個目(tx_bytes)だけだが
    // 読み飛ばすために全部変数を用意する
    std::istringstream iss(line.substr(colon + 1));
    long rx_packets, rx_errs, rx_drop, rx_fifo, rx_frame, rx_compressed, rx_multicast;
    long tx_packets, tx_errs, tx_drop, tx_fifo, tx_colls, tx_carrier, tx_compressed;
    iss >> out.rx_bytes
        >> rx_packets >> rx_errs >> rx_drop >> rx_fifo
        >> rx_frame >> rx_compressed >> rx_multicast
        >> out.tx_bytes
        >> tx_packets >> tx_errs >> tx_drop >> tx_fifo
        >> tx_colls >> tx_carrier >> tx_compressed;
    return !iss.fail();
}

std::vector<NetDev> read_netdev() {
    std::vector<NetDev> devs;
    std::ifstream file("/proc/net/dev");
    std::string line;
    std::getline(file, line); // ヘッダ1行目を捨てる
    std::getline(file, line); // ヘッダ2行目を捨てる
    while (std::getline(file, line)) {
        NetDev d;
        if (parse_netdev_line(line, d)) {
            devs.push_back(d);
        }
    }
    return devs;
}

// 名前でインターフェースを探す。2回目の読み取りと対応付けるために使う
// （インターフェースの出現順は変わり得るので、位置ではなく名前で対応させる）
const NetDev* find_dev(const std::vector<NetDev>& devs, const std::string& name) {
    for (const auto& d : devs) {
        if (d.name == name) return &d;
    }
    return nullptr;
}

int main() {
    // ----- Memory -----
    long total_kb = meminfo_kb("MemTotal");
    long avail_kb = meminfo_kb("MemAvailable");
    long free_kb  = meminfo_kb("MemFree");

    std::cout << "Memory\n";
    std::cout << "  Total:     " << total_kb / 1024 << " MiB\n";
    std::cout << "  Used:      " << (total_kb - avail_kb) / 1024 << " MiB\n";
    std::cout << "  Available: " << avail_kb / 1024 << " MiB\n";
    std::cout << "  Free:      " << free_kb  / 1024 << " MiB\n";

    // ----- CPU -----
    std::cout << "CPU\n";
    std::cout << "  Model:   " << cpu_model_name() << '\n';
    std::cout << "  Logical: " << logical_cpus() << " cores\n";

    // ----- Processes + Threads -----
    int total_procs = 0;
    int total_threads_task = 0;
    int total_threads_status = 0;
    int thread_mismatches = 0;
    std::map<char, int> states;
    std::vector<std::pair<std::string, int>> thread_rows;

    for (const auto& entry : fs::directory_iterator("/proc")) {
        if (!entry.is_directory()) continue;
        std::string name = entry.path().filename().string();
        if (!is_pid_name(name)) continue;

        ++total_procs;
        states[process_state(name)]++;

        int t_task   = threads_by_task(name);
        int t_status = threads_by_status(name);
        if (t_task != t_status) ++thread_mismatches;
        total_threads_task   += t_task;
        total_threads_status += t_status;

        thread_rows.push_back({process_name(name), t_task});
    }

    std::cout << "Processes\n";
    std::cout << "  Total:   " << total_procs << '\n';
    std::cout << "  States: ";
    for (const auto& [state, count] : states) {
        std::cout << state << "=" << count << " ";
    }
    std::cout << '\n';

    std::cout << "Threads\n";
    std::cout << "  Total (task):   " << total_threads_task << '\n';
    std::cout << "  Total (status): " << total_threads_status << '\n';
    std::cout << "  Mismatches:     " << thread_mismatches << '\n';
    std::cout << "  Top processes:\n";
    std::sort(thread_rows.begin(), thread_rows.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });
    for (size_t i = 0; i < 5 && i < thread_rows.size(); ++i) {
        std::cout << "    " << thread_rows[i].first << "  "
                  << thread_rows[i].second << " threads\n";
    }

    // ----- CPU usage + Network rate -----   【M5 NEW：計測区間】
    // どちらも累積カウンタなので、同じ計測区間(500ms)で差分を取る。
    // CPU使用率とネットワークレートは同じパターンで計算できる。
    CpuTimes t1 = read_cpu_times();
    auto devs1 = read_netdev();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    CpuTimes t2 = read_cpu_times();
    auto devs2 = read_netdev();

    std::cout << "  Usage:   " << cpu_usage_percent(t1, t2) << " %\n";

    std::cout << "Network\n";              // 【M5 NEW：出力ブロック】
    for (const auto& a : devs1) {
        const NetDev* b = find_dev(devs2, a.name);
        double rx_rate = 0, tx_rate = 0;
        if (b) {
            // 0.5秒区間の差分を2倍して bytes/sec にする
            rx_rate = (b->rx_bytes - a.rx_bytes) / 0.5;
            tx_rate = (b->tx_bytes - a.tx_bytes) / 0.5;
        }
        std::cout << "  " << a.name
                  << ": RX " << a.rx_bytes / 1024 << " KiB"
                  << "  TX " << a.tx_bytes / 1024 << " KiB"
                  << "  (RX " << rx_rate / 1024 << " KiB/s"
                  << "  TX " << tx_rate / 1024 << " KiB/s)\n";
    }
}
