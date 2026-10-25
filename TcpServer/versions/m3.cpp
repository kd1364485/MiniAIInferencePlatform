// Phase 2 M3: thread pool echo server (保存版)
#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>

class FdQueue {
public:
    void push(int fd) {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            q_.push(fd);
        }
        cv_.notify_one();
    }
    int pop() {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this] { return !q_.empty(); });
        int fd = q_.front();
        q_.pop();
        return fd;
    }
private:
    std::queue<int> q_;
    std::mutex mtx_;
    std::condition_variable cv_;
};

void handle_client(int client_fd) {
    char buf[4096];
    ssize_t n;
    while ((n = recv(client_fd, buf, sizeof(buf), 0)) > 0) {
        send(client_fd, buf, n, 0);
    }
    close(client_fd);
}

void worker(FdQueue& q) {
    while (true) {
        int fd = q.pop();
        handle_client(fd);
    }
}

int main(int argc, char** argv) {
    int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(8080);
    if (bind(listen_fd, (sockaddr*)&addr, sizeof(addr)) < 0) { perror("bind"); return 1; }
    if (listen(listen_fd, 5) < 0) { perror("listen"); return 1; }

    int kWorkers = (argc > 1) ? std::atoi(argv[1]) : 4;
    FdQueue queue;
    std::vector<std::thread> workers;
    for (int i = 0; i < kWorkers; ++i) {
        workers.emplace_back(worker, std::ref(queue));
    }

    while (true) {
        int client_fd = accept(listen_fd, nullptr, nullptr);
        if (client_fd < 0) continue;
        queue.push(client_fd);
    }
}
