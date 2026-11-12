#include <algorithm>
#include <chrono>
#include <iostream>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include "collect.h"

// /proc の読み取りと解釈は collect.h / parse.h に切り出した。
// このファイルは、集めた値を表示するだけ。

int main() {
    using namespace probe;

    long total_kb = meminfo_kb("MemTotal");
    long avail_kb = meminfo_kb("MemAvailable");
    long free_kb = meminfo_kb("MemFree");
    long used_kb = total_kb - avail_kb;

    // kB -> MiB で表示
    std::cout << "Memory\n";
    std::cout << "  Total:     " << total_kb / 1024 << " MiB\n";
    std::cout << "  Used:      " << used_kb / 1024 << " MiB\n";
    std::cout << "  Available: " << avail_kb / 1024 << " MiB\n";
    std::cout << "  Free:      " << free_kb / 1024 << " MiB\n";

    std::cout << "CPU\n";
    std::cout << " Model:      " << cpu_model_name() << "\n";
    std::cout << " Logical:    " << logical_cpus() << " cores\n";

    // Process
    int total = 0;
    int total_thread_task = 0;
    int total_thread_status = 0;
    int total_failedThreads = 0;
    std::vector<std::pair<std::string, int>> rows;  // (名前, スレッド数)
    std::map<char, int> states;
    for (const auto& entry : fs::directory_iterator("/proc")) {
        if (!entry.is_directory()) continue;
        // ディレクトリ名が数字だけかどうかでプロセスか判定
        std::string name = entry.path().filename().string();
        if (!is_pid_name(name)) continue;

        ++total;
        // /proc/<pid>/status の "State:" 行から状態コード(S や R)を1文字取る
        states[process_state(name)]++;

        // スレッド数を 2 通りの方法で数えて、結果が一致するか確かめる
        int count_task = threads_by_task(name);
        int count_status = threads_by_status(name);
        if (count_task != count_status) {
            ++total_failedThreads;
        }

        rows.push_back({process_name(name), count_task});

        total_thread_task += count_task;
        total_thread_status += count_status;
    }

    std::cout << "Processes\n";
    std::cout << " Total: " << total << "\n";
    std::cout << " Total(task):" << total_thread_task << "\n";
    std::cout << " Total(status):" << total_thread_status << "\n";
    std::cout << " Failed Threads: " << total_failedThreads << "\n";
    std::cout << " States: ";
    std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;  // スレッド数で降順ソート
    });

    // 状態ごとの数を表示 S=120 みたいに並べる
    for (const auto& [state, count] : states) {
        std::cout << state << "=" << count << " ";
    }
    std::cout << "\n";

    std::cout << "\n\nTop 5 processes by thread count:\n";
    for (size_t i = 0; i < 5 && i < rows.size(); i++) {
        std::cout << " " << rows[i].first << " " << rows[i].second << " threads\n";
    }

    // ----- CPU usage + Network rate -----
    // どちらも累積カウンタなので、同じ計測区間(500ms)で差分を取る。
    CpuTimes t1 = read_cpu_times();
    auto devs1 = read_netdev();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    CpuTimes t2 = read_cpu_times();
    auto devs2 = read_netdev();

    std::cout << "Usage:  " << cpu_usage_percent(t1, t2) << "%\n";

    std::cout << "Network\n";
    for (const auto& a : devs1) {
        const NetDev* b = find_dev(devs2, a.name);
        double rx_rate = 0.0, tx_rate = 0.0;
        if (b) {
            // 0.5 秒区間の差分を 2 倍して bytes/sec にする
            rx_rate = (b->rx_bytes - a.rx_bytes) / 0.5;
            tx_rate = (b->tx_bytes - a.tx_bytes) / 0.5;
        }
        std::cout << " " << a.name << ": RX " << a.rx_bytes / 1024 << " KiB"
                  << " TX " << a.tx_bytes / 1024 << " KiB"
                  << " (RX " << rx_rate / 1024 << " KiB/s"
                  << " TX " << tx_rate / 1024 << " KiB/s)\n";
    }

    // ----- Disk -----
    auto disks = read_diskstats();
    std::cout << "Disk\n";
    for (const auto& d : disks) {
        if (d.reads == 0 && d.sectors_written == 0) continue;

        // 1 セクターは 512 バイト → KiB は * 512 / 1024
        std::cout << " " << d.name << ": reads " << d.reads << " R "
                  << d.sectors_read * 512 / 1024 << " KiB"
                  << " W " << d.sectors_written * 512 / 1024 << " KiB\n";
    }
}
