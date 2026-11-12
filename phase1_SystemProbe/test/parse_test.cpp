// parse_test.cpp — parse.h の単体テスト
// /proc の中身を模した文字列を std::istringstream に入れて渡す。実機の /proc には触れない。
#include <sstream>

#include "check.h"
#include "parse.h"

using namespace probe;

static void test_is_pid_name() {
    CHECK(is_pid_name("1"));
    CHECK(is_pid_name("123456"));
    CHECK(!is_pid_name("self"));      // /proc/self はプロセスではなくリンク
    CHECK(!is_pid_name("12a"));
    CHECK(!is_pid_name("cpuinfo"));
    CHECK(!is_pid_name(""));          // 空文字は false(std::all_of は空で true を返すため明示的に除外)
}

static void test_threads_and_state() {
    std::istringstream status(
        "Name:\tbash\n"
        "Umask:\t0022\n"
        "State:\tS (sleeping)\n"
        "Tgid:\t812\n"
        "Threads:\t7\n"
        "SigQ:\t0/15000\n");
    CHECK_EQ(parse_threads(status), 7);

    std::istringstream status2(
        "Name:\tbash\n"
        "State:\tR (running)\n");
    CHECK_EQ(parse_state(status2), 'R');

    // 行がない場合
    std::istringstream empty("Name:\tx\n");
    CHECK_EQ(parse_threads(empty), 0);
    std::istringstream empty2("Name:\tx\n");
    CHECK_EQ(parse_state(empty2), '?');

    // 値が空の行("State:" だけ)
    std::istringstream blank("State:\n");
    CHECK_EQ(parse_state(blank), '?');
}

static void test_meminfo() {
    const char* text =
        "MemTotal:       16384000 kB\n"
        "MemFree:         1024000 kB\n"
        "MemAvailable:    8192000 kB\n"
        "Buffers:          204800 kB\n"
        "HugePages_Total:       0\n"      // 単位のない行
        "Hugepagesize:       2048 kB\n";

    std::istringstream a(text);
    CHECK_EQ(parse_meminfo_kb(a, "MemTotal"), 16384000L);
    std::istringstream b(text);
    CHECK_EQ(parse_meminfo_kb(b, "MemAvailable"), 8192000L);
    std::istringstream c(text);
    CHECK_EQ(parse_meminfo_kb(c, "HugePages_Total"), 0L);   // 単位なしでも読める
    std::istringstream d(text);
    CHECK_EQ(parse_meminfo_kb(d, "NoSuchKey"), -1L);        // なければ -1
    std::istringstream e(text);
    CHECK_EQ(parse_meminfo_kb(e, "Mem"), -1L);              // 前方一致では拾わない
    std::istringstream f("");
    CHECK_EQ(parse_meminfo_kb(f, "MemTotal"), -1L);         // 空入力
}

static void test_cpuinfo() {
    const char* text =
        "processor\t: 0\n"
        "vendor_id\t: GenuineIntel\n"
        "model name\t: Intel(R) Core(TM) i7-12700H\n"
        "cpu MHz\t\t: 2688.000\n"
        "\n"
        "processor\t: 1\n"
        "model name\t: Intel(R) Core(TM) i7-12700H\n";

    std::istringstream a(text);
    CHECK_EQ(parse_cpu_model(a), std::string("Intel(R) Core(TM) i7-12700H"));
    std::istringstream b(text);
    CHECK_EQ(parse_logical_cpus(b), 2);

    std::istringstream none("vendor_id\t: x\n");
    CHECK_EQ(parse_cpu_model(none), std::string("unknown"));
    std::istringstream none2("");
    CHECK_EQ(parse_logical_cpus(none2), 0);

    // コロンのない "model name" 行は読み飛ばす
    std::istringstream broken("model name no colon\nmodel name\t: AMD EPYC\n");
    CHECK_EQ(parse_cpu_model(broken), std::string("AMD EPYC"));
}

static void test_cpu_times_and_usage() {
    std::istringstream a("cpu  100 20 30 400 50 6 7 8\ncpu0 1 2 3 4 5 6 7 8\n");
    CpuTimes t = parse_cpu_times(a);
    CHECK_EQ(t.user, 100L);
    CHECK_EQ(t.nice, 20L);
    CHECK_EQ(t.system, 30L);
    CHECK_EQ(t.idle, 400L);
    CHECK_EQ(t.iowait, 50L);
    CHECK_EQ(t.irq, 6L);
    CHECK_EQ(t.softirq, 7L);
    CHECK_EQ(t.steal, 8L);
    CHECK_EQ(total_time(t), 621L);
    CHECK_EQ(idle_time(t), 450L);   // idle + iowait

    // 差分から使用率: 合計 +100 のうち idle が +50 なら 50%
    CpuTimes x, y;
    x.user = 100; x.idle = 900;
    y.user = 150; y.idle = 950;
    CHECK_NEAR(cpu_usage_percent(x, y), 50.0, 1e-9);

    // iowait の増加は「使用」ではなく idle 側
    CpuTimes p, q;
    p.user = 0; p.idle = 0; p.iowait = 0;
    q.user = 0; q.idle = 0; q.iowait = 100;
    CHECK_NEAR(cpu_usage_percent(p, q), 0.0, 1e-9);

    // 全部 user なら 100%
    CpuTimes r, s;
    s.user = 200;
    CHECK_NEAR(cpu_usage_percent(r, s), 100.0, 1e-9);

    // 変化がなければ 0(0 除算しない)
    CHECK_NEAR(cpu_usage_percent(x, x), 0.0, 1e-9);
}

static void test_netdev() {
    const char* text =
        "Inter-|   Receive                                                |  Transmit\n"
        " face |bytes    packets errs drop fifo frame compressed multicast|bytes    packets errs drop fifo colls carrier compressed\n"
        "    lo:  500000    1000    0    0    0     0          0         0   500000    1000    0    0    0     0       0          0\n"
        "  eth0: 123456789  90000    1    2    0     0          0         3 987654321   80000    0    0    0     0       0          0\n"
        "docker0:       0       0    0    0    0     0          0         0        0       0    0    0    0     0       0          0\n";

    std::istringstream a(text);
    auto devs = parse_netdev(a);
    CHECK_EQ(devs.size(), size_t(3));
    CHECK_EQ(devs[0].name, std::string("lo"));          // 先頭の空白を取り除く
    CHECK_EQ(devs[0].rx_bytes, 500000L);
    CHECK_EQ(devs[1].name, std::string("eth0"));
    CHECK_EQ(devs[1].rx_bytes, 123456789L);
    CHECK_EQ(devs[1].tx_bytes, 987654321L);             // 9 個目が送信バイト
    CHECK_EQ(devs[2].name, std::string("docker0"));

    const NetDev* e = find_dev(devs, "eth0");
    CHECK(e != nullptr);
    CHECK(find_dev(devs, "wlan0") == nullptr);

    // 壊れた行
    NetDev d;
    CHECK(!parse_netdev_line("no colon here", d));
    CHECK(!parse_netdev_line("   : 1 2 3", d));         // 名前が空
    CHECK(!parse_netdev_line("eth0: 1 2 3", d));         // 数値が足りない
    CHECK(!parse_netdev_line("eth0: a b c d e f g h i j k l m n o p", d)); // 数値でない

    // ヘッダ 2 行だけなら空
    std::istringstream headers_only("h1\nh2\n");
    CHECK_EQ(parse_netdev(headers_only).size(), size_t(0));
}

static void test_diskstats() {
    const char* text =
        "   8       0 sda 1000 20 30000 400 500 10 6000 70 0 800 470 0 0 0 0\n"
        "   8       1 sda1 10 0 80 5 0 0 0 0 0 5 5 0 0 0 0\n"
        " 259       0 nvme0n1 0 0 0 0 0 0 0 0 0 0 0\n"
        "garbage line\n"
        "   7       0 loop0 5 0 10 0 0 0 0 0 0 0 0\n";

    std::istringstream a(text);
    auto disks = parse_diskstats(a);
    CHECK_EQ(disks.size(), size_t(4));                   // "garbage line" は読み飛ばす
    CHECK_EQ(disks[0].name, std::string("sda"));
    CHECK_EQ(disks[0].reads, 1000L);                     // 4 番目の値
    CHECK_EQ(disks[0].sectors_read, 30000L);             // 6 番目の値
    CHECK_EQ(disks[0].sectors_written, 6000L);           // 10 番目の値
    CHECK_EQ(disks[2].name, std::string("nvme0n1"));
    CHECK_EQ(disks[2].reads, 0L);

    std::istringstream empty("");
    CHECK_EQ(parse_diskstats(empty).size(), size_t(0));
}

int main() {
    test_is_pid_name();
    test_threads_and_state();
    test_meminfo();
    test_cpuinfo();
    test_cpu_times_and_usage();
    test_netdev();
    test_diskstats();
    return check_report("parse_test");
}
