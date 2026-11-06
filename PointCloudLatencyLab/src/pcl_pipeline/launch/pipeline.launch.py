"""5 つのノードを、それぞれ別のプロセスで起動する(M1)。M2 の測定用に CSV と sink を足せる。

使い方:
  ros2 launch pcl_pipeline pipeline.launch.py [rings:=64] [azimuth_steps:=1800] [rate_hz:=10.0]
      [rviz:=false] [csv_dir:=/path/to/dir] [sink:=true] [sink_topic:=sensor/points]
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    args = [
        DeclareLaunchArgument('rings', default_value='32'),
        DeclareLaunchArgument('azimuth_steps', default_value='1800'),
        DeclareLaunchArgument('rate_hz', default_value='10.0'),
        DeclareLaunchArgument('rviz', default_value='true'),
        # 空でなければ、各ノードが <csv_dir>/<ノード名>.csv に 1 フレーム 1 行を書く
        DeclareLaunchArgument('csv_dir', default_value=''),
        # true なら点群のトピックを reliable にする(既定は best_effort)
        DeclareLaunchArgument('qos_reliable', default_value='false'),
        # true なら voxel_downsample を CUDA 版にする(M5)
        DeclareLaunchArgument('use_cuda', default_value='false'),
        # 受け取るだけのノードを足す(RViz2 の「受け取る相手が増える」分だけを再現する)
        DeclareLaunchArgument('sink', default_value='false'),
        DeclareLaunchArgument('sink_topic', default_value='sensor/points'),
    ]
    csv = {'csv_dir': ParameterValue(LaunchConfiguration('csv_dir'), value_type=str),
           'qos_reliable': ParameterValue(LaunchConfiguration('qos_reliable'), value_type=bool)}
    voxel_params = {'use_cuda': ParameterValue(LaunchConfiguration('use_cuda'), value_type=bool)}
    sim_params = {
        'rings': ParameterValue(LaunchConfiguration('rings'), value_type=int),
        'azimuth_steps': ParameterValue(LaunchConfiguration('azimuth_steps'), value_type=int),
        'rate_hz': ParameterValue(LaunchConfiguration('rate_hz'), value_type=float),
    }

    def stage(executable, params=None):
        return Node(package='pcl_pipeline', executable=executable, name=executable,
                    parameters=[csv, params or {}], output='screen')

    rviz_config = PathJoinSubstitution([FindPackageShare('pcl_pipeline'), 'rviz', 'pipeline.rviz'])
    return LaunchDescription(args + [
        stage('sensor_sim', sim_params),
        stage('crop_box'),
        stage('voxel_downsample', voxel_params),
        stage('ground_removal'),
        stage('cluster'),
        Node(package='pcl_pipeline', executable='sink', name='sink', output='screen',
             parameters=[{'topic': LaunchConfiguration('sink_topic')}],
             condition=IfCondition(LaunchConfiguration('sink'))),
        Node(package='rviz2', executable='rviz2', arguments=['-d', rviz_config],
             condition=IfCondition(LaunchConfiguration('rviz')), output='log'),
    ])
