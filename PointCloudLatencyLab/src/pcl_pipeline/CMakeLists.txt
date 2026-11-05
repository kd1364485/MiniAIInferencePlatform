cmake_minimum_required(VERSION 3.8)
project(pcl_pipeline)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra -Wpedantic)

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

# ノードはすべて部品(component)として 1 つの共有ライブラリに入れる(M3)。
#  - 1 プロセスにまとめる: component_container に読み込む(launch/composed.launch.py)
#  - 別プロセス: rclcpp_components_register_node が、部品ごとの単独の実行ファイルも作る(launch/pipeline.launch.py)
add_library(pipeline_components SHARED
  src/sensor_sim_node.cpp src/crop_box_node.cpp src/voxel_downsample_node.cpp
  src/ground_removal_node.cpp src/cluster_node.cpp src/sink_node.cpp)
target_link_libraries(pipeline_components pipeline_core)
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

ament_package()
