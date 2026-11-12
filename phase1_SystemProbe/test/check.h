// check.h — 外部ライブラリなしで使える、最小のテスト用マクロ
// 失敗してもそこで止めず、最後に失敗数を返す(1 回の実行で全部の失敗が分かる)
#pragma once

#include <cmath>
#include <iostream>

namespace check {
inline int failures = 0;
inline int total = 0;
}  // namespace check

#define CHECK(cond)                                                              \
    do {                                                                         \
        ++check::total;                                                          \
        if (!(cond)) {                                                           \
            ++check::failures;                                                   \
            std::cerr << __FILE__ << ":" << __LINE__ << "  CHECK failed: " #cond \
                      << "\n";                                                   \
        }                                                                        \
    } while (0)

#define CHECK_EQ(a, b)                                                          \
    do {                                                                        \
        ++check::total;                                                         \
        auto&& va = (a);                                                        \
        auto&& vb = (b);                                                        \
        if (!(va == vb)) {                                                      \
            ++check::failures;                                                  \
            std::cerr << __FILE__ << ":" << __LINE__ << "  CHECK_EQ failed: " #a \
                      << " == " #b << "  (" << va << " vs " << vb << ")\n";     \
        }                                                                       \
    } while (0)

#define CHECK_NEAR(a, b, eps)                                                   \
    do {                                                                        \
        ++check::total;                                                         \
        double va = (a), vb = (b);                                              \
        if (!(std::fabs(va - vb) <= (eps))) {                                   \
            ++check::failures;                                                  \
            std::cerr << __FILE__ << ":" << __LINE__ << "  CHECK_NEAR failed: " #a \
                      << " ~ " #b << "  (" << va << " vs " << vb << ")\n";      \
        }                                                                       \
    } while (0)

inline int check_report(const char* name) {
    std::cout << name << ": " << (check::total - check::failures) << "/" << check::total
              << " passed\n";
    return check::failures == 0 ? 0 : 1;
}
