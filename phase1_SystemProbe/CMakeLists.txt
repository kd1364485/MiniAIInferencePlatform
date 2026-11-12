cmake_minimum_required(VERSION 3.16)
project(systemprobe VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 指定がなければ Release でビルドする(ベンチの数字が -O0 で歪まないように)。
# gdb で調べたいときは -DCMAKE_BUILD_TYPE=Debug を付ける。
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

add_compile_options(-Wall -Wextra)

# 本体
add_executable(systemprobe src/main.cpp)
target_include_directories(systemprobe PRIVATE src)

# 単体テスト(ctest で実行)
enable_testing()

add_executable(parse_test test/parse_test.cpp)
target_include_directories(parse_test PRIVATE src test)
add_test(NAME parse_test COMMAND parse_test)

add_executable(collect_test test/collect_test.cpp)
target_include_directories(collect_test PRIVATE src test)
find_package(Threads REQUIRED)
target_link_libraries(collect_test PRIVATE Threads::Threads)
add_test(NAME collect_test COMMAND collect_test)

# ベンチマーク(ctest には入れない。./bench を手で実行する)
add_executable(bench bench/bench.cpp)
target_include_directories(bench PRIVATE src)

# strace / perf で観察するための、1 つの処理だけを繰り返すプログラム
add_executable(syscalls bench/syscalls.cpp)
target_include_directories(syscalls PRIVATE src)
