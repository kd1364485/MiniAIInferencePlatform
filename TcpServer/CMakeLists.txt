cmake_minimum_required(VERSION 3.16)
project(tcpserver VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(tcpserver src/main.cpp)
add_executable(bench src/bench.cpp)

# ===== Phase 6: ONNX Runtime 推論サーバー =====
# ONNX Runtime(C++ 用の配布パッケージ)の場所。既定は従来どおり ../OnnxLearn/ort。
# 別の場所に置いたなら  cmake -DORT_DIR=/path/to/onnxruntime-linux-x64-x.y.z ..  で上書きする。
set(ORT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../OnnxLearn/ort CACHE PATH "ONNX Runtime package directory")
if(NOT EXISTS ${ORT_DIR}/include/onnxruntime_cxx_api.h)
  message(FATAL_ERROR "ONNX Runtime が見つかりません: ${ORT_DIR}\n"
                      "配布パッケージを展開し、cmake -DORT_DIR=<展開先> .. で指定してください。")
endif()
find_package(Threads REQUIRED)
add_executable(infer_server src/infer_server.cpp)
target_include_directories(infer_server PRIVATE ${ORT_DIR}/include)
target_link_directories(infer_server PRIVATE ${ORT_DIR}/lib)
target_link_libraries(infer_server PRIVATE onnxruntime Threads::Threads)
set_target_properties(infer_server PROPERTIES BUILD_RPATH ${ORT_DIR}/lib)
add_executable(infer_bench src/infer_bench.cpp)
target_link_libraries(infer_bench PRIVATE Threads::Threads)

# ネットワークなしで Run() だけを測る切り分け用ベンチ
add_executable(run_bench src/run_bench.cpp)
target_include_directories(run_bench PRIVATE ${ORT_DIR}/include)
target_link_directories(run_bench PRIVATE ${ORT_DIR}/lib)
target_link_libraries(run_bench PRIVATE onnxruntime Threads::Threads)
set_target_properties(run_bench PROPERTIES BUILD_RPATH ${ORT_DIR}/lib)
