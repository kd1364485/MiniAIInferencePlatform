cmake_minimum_required(VERSION 3.16)
project(tcpserver VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(tcpserver src/main.cpp)
add_executable(bench src/bench.cpp)

# ===== Phase 6: ONNX Runtime 推論サーバー =====
set(ORT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../OnnxLearn/ort)
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
