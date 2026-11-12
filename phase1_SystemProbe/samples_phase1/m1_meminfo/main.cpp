#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

// /proc/meminfo の各行は "Key:    value kB" 形式。
// 指定したキーの値(kB)を返す。見つからなければ -1。
long meminfo_kb(const std::string& key) {
    std::ifstream file("/proc/meminfo");
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string k;
        long value = 0;
        std::string unit; // "kB"。HugePages系など単位なしの行もある
        iss >> k >> value >> unit;
        if (k == key + ":") {
            return value;
        }
    }
    return -1;
}

int main() {
    long total_kb = meminfo_kb("MemTotal");
    long avail_kb = meminfo_kb("MemAvailable");
    long free_kb  = meminfo_kb("MemFree");
    long used_kb  = total_kb - avail_kb;

    // kB -> MiB で表示
    std::cout << "Memory\n";
    std::cout << "  Total:     " << total_kb / 1024 << " MiB\n";
    std::cout << "  Used:      " << used_kb  / 1024 << " MiB\n";
    std::cout << "  Available: " << avail_kb / 1024 << " MiB\n";
    std::cout << "  Free:      " << free_kb  / 1024 << " MiB\n";
}
