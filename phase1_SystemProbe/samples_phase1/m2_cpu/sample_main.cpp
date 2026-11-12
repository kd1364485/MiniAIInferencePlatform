#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

// ===== Memory (/proc/meminfo) =====

// 各行は "Key:    value kB" 形式。指定キーの値(kB)を返す。
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

// "model name : Intel(R) ..." の右側を取り出す
std::string cpu_model_name() {
    std::ifstream file("/proc/cpuinfo");
    std::string line;
    while (std::getline(file, line)) {
        if (line.starts_with("model name")) {
            return line.substr(line.find(':') + 2); // ": " を飛ばす
        }
    }
    return "unknown";
}

// "processor : N" 行の数 = 論理CPU数(ハードウェアスレッド数)
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

// /proc/stat の1行目 "cpu  user nice system idle iowait irq softirq steal"
// は起動からの累積時間(jiffies単位)。瞬間値ではなくカウンタ。
struct CpuTimes {
    long user = 0, nice = 0, system = 0, idle = 0;
    long iowait = 0, irq = 0, softirq = 0, steal = 0;
};

CpuTimes read_cpu_times() {
    std::ifstream file("/proc/stat");
    std::string line;
    std::getline(file, line); // 1行目だけ使う
    std::istringstream iss(line);
    std::string cpu; // "cpu" という文字を捨てる
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

// 2回読んで差分から使用率%を出す。
// カウンタは累積なので、1回の読み取りだけでは%は計算できない。
double cpu_usage_percent(const CpuTimes& a, const CpuTimes& b) {
    long total_d = total_time(b) - total_time(a);
    long idle_d  = idle_time(b)  - idle_time(a);
    if (total_d == 0) return 0.0;
    return 100.0 * (total_d - idle_d) / total_d;
}

int main() {
    long total_kb = meminfo_kb("MemTotal");
    long avail_kb = meminfo_kb("MemAvailable");
    long free_kb  = meminfo_kb("MemFree");

    std::cout << "Memory\n";
    std::cout << "  Total:     " << total_kb / 1024 << " MiB\n";
    std::cout << "  Used:      " << (total_kb - avail_kb) / 1024 << " MiB\n";
    std::cout << "  Available: " << avail_kb / 1024 << " MiB\n";
    std::cout << "  Free:      " << free_kb  / 1024 << " MiB\n";

    std::cout << "CPU\n";
    std::cout << "  Model:   " << cpu_model_name() << '\n';
    std::cout << "  Logical: " << logical_cpus() << " cores\n";

    CpuTimes t1 = read_cpu_times();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    CpuTimes t2 = read_cpu_times();
    std::cout << "  Usage:   " << cpu_usage_percent(t1, t2) << " %\n";
}
