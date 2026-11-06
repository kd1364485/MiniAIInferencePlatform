cmake_minimum_required(VERSION 3.8)
project(m0_hello)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

add_executable(m0_hello src/m0_hello.cpp)
target_include_directories(m0_hello PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>)
target_compile_features(m0_hello PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
ament_target_dependencies(
  m0_hello
  "rclcpp"
  "std_msgs"
)

install(TARGETS m0_hello
  DESTINATION lib/${PROJECT_NAME})

# M0 の環境確認用のサンプルなので、自動の lint(ament_lint_auto)は付けない

ament_package()
