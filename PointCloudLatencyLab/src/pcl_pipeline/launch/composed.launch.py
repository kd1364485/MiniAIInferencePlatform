"""M3: 5 つのノードを 1 つのプロセス(component container)にまとめて起動する。

使い方:
  ros2 launch pcl_pipeline composed.launch.py [ipc:=true] [container:=component_container]
      [rings:=32] [rate_hz:=10.0] [csv_dir:=/path/to/dir]

  ipc       : true なら intra-process 通信(同じプロセスの中は DDS を通らず、unique_ptr をそのまま渡す)
  container : component_container(1 スレッドで全ノードを回す)/ component_container_mt(複数スレッド)
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    args = [
        DeclareLaunchArgument('ipc', default_value='true'),
        DeclareLaunchArgument('container', default_value='component_container'),
        DeclareLaunchArgument('rings', default_value='32'),
        DeclareLaunchArgument('azimuth_steps', default_value='1800'),
        DeclareLaunchArgument('rate_hz', default_value='10.0'),
        DeclareLaunchArgument('csv_dir', default_value=''),
        # true なら点群のトピックを reliable にする(既定は best_effort)
        DeclareLaunchArgument('qos_reliable', default_value='false'),
        # true なら voxel_downsample を CUDA 版にする(M5)
        DeclareLaunchArgument('use_cuda', default_value='false'),
    ]
    csv = {'csv_dir': ParameterValue(LaunchConfiguration('csv_dir'), value_type=str),
           'qos_reliable': ParameterValue(LaunchConfiguration('qos_reliable'), value_type=bool)}
    ipc = [{'use_intra_process_comms': ParameterValue(LaunchConfiguration('ipc'), value_type=bool)}]
    voxel_params = {'use_cuda': ParameterValue(LaunchConfiguration('use_cuda'), value_type=bool)}
    sim_params = {
        'rings': ParameterValue(LaunchConfiguration('rings'), value_type=int),
        'azimuth_steps': ParameterValue(LaunchConfiguration('azimuth_steps'), value_type=int),
        'rate_hz': ParameterValue(LaunchConfiguration('rate_hz'), value_type=float),
    }

    def part(plugin, name, params=None):
        return ComposableNode(package='pcl_pipeline', plugin='pcl_pipeline::' + plugin, name=name,
                              parameters=[csv, params or {}], extra_arguments=ipc)

    container = ComposableNodeContainer(
        name='pipeline_container', namespace='', package='rclcpp_components',
        executable=LaunchConfiguration('container'), output='screen',
        composable_node_descriptions=[
            part('SensorSim', 'sensor_sim', sim_params),
            part('CropBox', 'crop_box'),
            part('VoxelDownsample', 'voxel_downsample', voxel_params),
            part('GroundRemoval', 'ground_removal'),
            part('ClusterNode', 'cluster'),
        ])
    return LaunchDescription(args + [container])
