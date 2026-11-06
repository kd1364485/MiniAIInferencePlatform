cmake_minimum_required(VERSION 3.8)
project(pcl_pipeline)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# 警告は C++ だけに付ける(nvcc に -Wpedantic を渡すと、生成コードで大量の警告が出る)
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>")

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

# 点群の処理そのもの(ROS に依存しない)。共有ライブラリに入れるので PIC にする
add_library(pipeline_core STATIC src/filters.cpp)
set_target_properties(pipeline_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(pipeline_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(pipeline_core PUBLIC cxx_std_17)

# M5: voxel の CUDA 版。CUDA が見つかったときだけ作る(見つからなければ CPU 版だけ)
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  # Ubuntu の CUDA 12.0 は gcc 13 に対応していないので、gcc 12 を使う
  if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER AND EXISTS /usr/bin/g++-12)
    set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-12)
  endif()
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(PCL_PIPELINE_WITH_CUDA ON)
  add_library(pipeline_cuda STATIC src/voxel_cuda.cu)
  set_target_properties(pipeline_cuda PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_STANDARD 17)
  target_link_libraries(pipeline_cuda PUBLIC pipeline_core CUDA::cudart)
  target_compile_definitions(pipeline_cuda PUBLIC PCL_PIPELINE_WITH_CUDA)

  add_executable(voxel_bench src/voxel_bench.cpp)
  target_link_libraries(voxel_bench pipeline_cuda)
  install(TARGETS voxel_bench DESTINATION lib/${PROJECT_NAME})
  message(STATUS "pcl_pipeline: CUDA enabled (${CMAKE_CUDA_COMPILER_VERSION})")
else()
  message(STATUS "pcl_pipeline: CUDA not found, CPU only")
endif()

# ノードはすべて部品(component)として 1 つの共有ライブラリに入れる(M3)。
#  - 1 プロセスにまとめる: component_container に読み込む(launch/composed.launch.py)
#  - 別プロセス: rclcpp_components_register_node が、部品ごとの単独の実行ファイルも作る(launch/pipeline.launch.py)
add_library(pipeline_components SHARED
  src/sensor_sim_node.cpp src/crop_box_node.cpp src/voxel_downsample_node.cpp
  src/ground_removal_node.cpp src/cluster_node.cpp src/sink_node.cpp)
target_link_libraries(pipeline_components pipeline_core)
if(PCL_PIPELINE_WITH_CUDA)
  target_link_libraries(pipeline_components pipeline_cuda)
endif()
ament_target_dependencies(pipeline_components rclcpp rclcpp_components sensor_msgs visualization_msgs)

rclcpp_components_register_node(pipeline_components PLUGIN "pcl_pipeline::SensorSim" EXECUTABLE sensor_sim)
rclcpp_components_register_node(pipeline_components PLUGIN "pcl_pipeline::CropBox" EXECUTABLE crop_box)
rclcpp_components_register_node(pipeline_components PLUGIN "pcl_pipeline::VoxelDownsample" EXECUTABLE voxel_downsample)
rclcpp_components_register_node(pipeline_components PLUGIN "pcl_pipeline::GroundRemoval" EXECUTABLE ground_removal)
rclcpp_components_register_node(pipeline_components PLUGIN "pcl_pipeline::ClusterNode" EXECUTABLE cluster)
rclcpp_components_register_node(pipeline_components PLUGIN "pcl_pipeline::Sink" EXECUTABLE sink)

install(TARGETS pipeline_components
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
install(DIRECTORY launch rviz DESTINATION share/${PROJECT_NAME})

# M6: テスト(colcon test)
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(launch_testing_ament_cmake REQUIRED)
  # 点群の処理の単体テスト
  ament_add_gtest(test_filters test/test_filters.cpp)
  target_link_libraries(test_filters pipeline_core)
  # CUDA 版が CPU 版と同じか(GPU がなければスキップ)
  if(PCL_PIPELINE_WITH_CUDA)
    ament_add_gtest(test_voxel_cuda test/test_voxel_cuda.cpp)
    target_link_libraries(test_voxel_cuda pipeline_cuda)
  endif()
  # パイプライン全体を起動して、箱が届くかの結合テスト
  add_launch_test(test/test_pipeline_launch.py TIMEOUT 60)
endif()

ament_package()
