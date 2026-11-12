// collect_test.cpp — collect.h を実機の /proc で確かめるテスト
// 値そのものは環境で変わるので、「あり得る範囲か」「別の方法と一致するか」を確かめる。
#include <atomic>
#include <chrono>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

#include "check.h"
#include "collect.h"

using namespace probe;

static void test_memory_is_sane() {
    long total = meminfo_kb("MemTotal");
    long avail = meminfo_kb("MemAvailable");
    long free_kb = meminfo_kb("MemFree");
    CHECK(total > 0);
    CHECK(avail > 0);
    CHECK(avail <= total);
    CHECK(free_kb >= 0);
    CHECK(free_kb <= avail);   // MemAvailable は MemFree 以上(回収できる分を含む)
}

static void test_cpu_is_sane() {
    CHECK(logical_cpus() >= 1);
    CHECK_EQ(logical_cpus(), logical_cpus_online());   // cpuinfo を数える方法と sysconf は同じ値
    CHECK(!cpu_model_name().empty());

    CpuTimes a = read_cpu_times();
    CHECK(total_time(a) > 0);
    // 累積カウンタなので、あとから読んだ値が小さくなることはない
    CpuTimes b = read_cpu_times();
    CHECK(total_time(b) >= total_time(a));
    double usage = cpu_usage_percent(a, b);
    CHECK(usage >= 0.0 && usage <= 100.0);
}

// 自分自身のスレッド数が、スレッドを足すと正しく増えること(2 つの方式の両方で確認)
static void test_thread_count_follows_reality() {
    const std::string self = std::to_string(getpid());
    const int base_task = threads_by_task(self);
    const int base_status = threads_by_status(self);
    CHECK(base_task >= 1);
    CHECK_EQ(base_task, base_status);    // 2 つの数え方が一致する

    constexpr int kExtra = 4;
    std::atomic<bool> stop{false};
    std::vector<std::thread> workers;
    for (int i = 0; i < kExtra; ++i) {
        workers.emplace_back([&] {
            while (!stop.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        });
    }
    // 起動直後は /proc に反映されるまで少し待つことがあるので、短く待って確かめる
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    CHECK_EQ(threads_by_task(self), base_task + kExtra);
    CHECK_EQ(threads_by_status(self), base_status + kExtra);

    stop = true;
    for (auto& t : workers) t.join();
}

static void test_process_listing() {
    const std::string self = std::to_string(getpid());
    CHECK(is_pid_name(self));
    CHECK(!process_name(self).empty());
    char s = process_state(self);
    CHECK(s == 'R' || s == 'S');         // 自分は走っているか、(一瞬)眠っている
    CHECK_EQ(process_state("999999999"), '?');     // 存在しない PID
    CHECK_EQ(threads_by_status("999999999"), 0);
    CHECK_EQ(process_name("999999999"), std::string(""));
}

static void test_net_and_disk() {
    auto devs = read_netdev();
    CHECK(!devs.empty());
    CHECK(find_dev(devs, "lo") != nullptr);   // ループバックはほぼ必ずある
    for (const auto& d : devs) {
        CHECK(d.rx_bytes >= 0);
        CHECK(d.tx_bytes >= 0);
    }
    auto disks = read_diskstats();
    for (const auto& d : disks) {
        CHECK(d.reads >= 0);
        CHECK(d.sectors_read >= 0);
        CHECK(d.sectors_written >= 0);
    }
}

int main() {
    test_memory_is_sane();
    test_cpu_is_sane();
    test_thread_count_follows_reality();
    test_process_listing();
    test_net_and_disk();
    return check_report("collect_test");
}
