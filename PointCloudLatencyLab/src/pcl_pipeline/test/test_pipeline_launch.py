"""M6: パイプライン全体の結合テスト(launch_testing)。

1 プロセス + intra-process のパイプライン(composed.launch.py)を起動し、
/objects/markers に箱が届くこと、箱の数が車の台数(8 台)に近いことを確かめる。
"""
import os
import time
import unittest

from ament_index_python.packages import get_package_share_directory
import launch
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
import launch_testing
import launch_testing.actions
import pytest
import rclpy
from visualization_msgs.msg import Marker, MarkerArray


@pytest.mark.launch_test
def generate_test_description():
    share = get_package_share_directory('pcl_pipeline')
    pipeline = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(share, 'launch', 'composed.launch.py')),
        launch_arguments={'ipc': 'true'}.items())
    return launch.LaunchDescription([pipeline, launch_testing.actions.ReadyToTest()])


class TestPipelineOutput(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def test_markers_arrive_with_plausible_count(self):
        node = rclpy.create_node('test_pipeline_listener')
        counts = []

        def on_markers(msg):
            counts.append(sum(1 for m in msg.markers if m.action == Marker.ADD))

        sub = node.create_subscription(MarkerArray, 'objects/markers', on_markers, 10)
        try:
            deadline = time.time() + 15.0
            while time.time() < deadline and len(counts) < 20:
                rclpy.spin_once(node, timeout_sec=0.1)
        finally:
            node.destroy_subscription(sub)
            node.destroy_node()

        self.assertGreaterEqual(len(counts), 20, 'markers did not arrive (20 frames in 15 s)')
        median = sorted(counts)[len(counts) // 2]
        # 車は 8 台。遠い・隠れている車は欠け、近い車が 2 つに割れることもあるので、幅を持たせる
        self.assertGreaterEqual(median, 4, f'too few clusters: {counts}')
        self.assertLessEqual(median, 14, f'too many clusters: {counts}')


@launch_testing.post_shutdown_test()
class TestShutdown(unittest.TestCase):

    def test_exit_codes(self, proc_info):
        # Ctrl+C で止めたときの終了コード(0 か SIGINT)だけを許す
        launch_testing.asserts.assertExitCodes(proc_info, allowable_exit_codes=[0, -2, 130])
