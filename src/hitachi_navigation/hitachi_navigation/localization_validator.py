#!/usr/bin/env python3
"""
Manual localization accuracy checker.

Subscribes to /robot_pose_map and keeps a rolling window of recent
samples. Place the robot at a spot, measure its true (x, y, yaw) with
a tape measure using the same origin/axes as aruco_locations.csv,
type it in, and this prints the error against the averaged live
estimate. Keeps a running summary across every spot you check.
"""

import math
import statistics
import threading

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose2D


def normalize_angle_deg(angle):
    """Wrap an angle to [-180, 180]."""
    while angle > 180.0:
        angle -= 360.0
    while angle < -180.0:
        angle += 360.0
    return angle


class LocalizationValidator(Node):

    def __init__(self):
        super().__init__('localization_validator')

        self.declare_parameter('average_window', 10)  # samples (~1s at 10Hz)
        self.average_window = int(self.get_parameter('average_window').value)

        self.recent_x = []
        self.recent_y = []
        self.recent_yaw_deg = []

        self.create_subscription(
            Pose2D, '/robot_pose_map', self.pose_callback, 10
        )

        self.get_logger().info(
            f'LocalizationValidator ready - averaging last '
            f'{self.average_window} /robot_pose_map samples per check'
        )

    def pose_callback(self, msg):
        self.recent_x.append(msg.x)
        self.recent_y.append(msg.y)
        self.recent_yaw_deg.append(math.degrees(msg.theta))

        if len(self.recent_x) > self.average_window:
            self.recent_x.pop(0)
            self.recent_y.pop(0)
            self.recent_yaw_deg.pop(0)

    def averaged_pose(self):
        """Mean (x, y) and circular mean yaw over the recent window."""

        if not self.recent_x:
            return None

        sin_sum = sum(math.sin(math.radians(y)) for y in self.recent_yaw_deg)
        cos_sum = sum(math.cos(math.radians(y)) for y in self.recent_yaw_deg)
        mean_yaw = math.degrees(math.atan2(sin_sum, cos_sum))

        return (
            statistics.mean(self.recent_x),
            statistics.mean(self.recent_y),
            mean_yaw,
            len(self.recent_x),
        )


def main(args=None):

    rclpy.init(args=args)
    node = LocalizationValidator()

    spin_thread = threading.Thread(
        target=rclpy.spin, args=(node,), daemon=True
    )
    spin_thread.start()

    results = []

    print()
    print('=' * 64)
    print('LOCALIZATION VALIDATOR')
    print('Place the robot at a known spot, measure its true')
    print('(x, y, yaw_deg) with a tape measure, then enter it here.')
    print('Use the same origin/axes as aruco_locations.csv.')
    print('Type "q" to quit and see the summary.')
    print('=' * 64)

    try:
        while True:
            line = input(
                '\nMeasured ground truth  x y yaw_deg  (or q): '
            ).strip()

            if line.lower() in ('q', 'quit', 'exit'):
                break

            parts = line.split()

            if len(parts) != 3:
                print('Enter three numbers: x y yaw_deg (e.g. "1.20 0.35 90")')
                continue

            try:
                true_x, true_y, true_yaw = (float(v) for v in parts)
            except ValueError:
                print('Could not parse - enter three numbers.')
                continue

            avg = node.averaged_pose()

            if avg is None:
                print(
                    'No /robot_pose_map samples received yet - '
                    'is robot_localizer running?'
                )
                continue

            map_x, map_y, map_yaw, n = avg

            dx = true_x - map_x
            dy = true_y - map_y
            dist_err = math.hypot(dx, dy)
            dyaw = normalize_angle_deg(true_yaw - map_yaw)

            results.append(
                (true_x, true_y, true_yaw, map_x, map_y, map_yaw, dist_err, dyaw)
            )

            print(
                f'  map (avg of {n}) = ({map_x:.3f}, {map_y:.3f}, '
                f'{map_yaw:.1f}deg)\n'
                f'  error            = {dist_err * 100:.1f}cm, {dyaw:+.1f}deg '
                f'(dx={dx * 100:+.1f}cm, dy={dy * 100:+.1f}cm)'
            )

    except (KeyboardInterrupt, EOFError):
        pass

    if results:
        print('\n' + '=' * 64)
        print('SUMMARY')
        print('=' * 64)
        print(
            f'{"true(x,y,yaw)":<24}{"map(x,y,yaw)":<24}'
            f'{"err(cm)":<10}{"err(deg)":<10}'
        )

        for (tx, ty, tyaw, mx, my, myaw, derr, dyaw) in results:
            print(
                f'({tx:.2f},{ty:.2f},{tyaw:.0f}){"":<9}'
                f'({mx:.2f},{my:.2f},{myaw:.0f}){"":<9}'
                f'{derr * 100:<10.1f}{dyaw:<10.1f}'
            )

        dist_errors = [r[6] for r in results]
        yaw_errors = [abs(r[7]) for r in results]

        print(
            f'\nPosition error: mean={statistics.mean(dist_errors) * 100:.1f}cm '
            f'max={max(dist_errors) * 100:.1f}cm'
        )
        print(
            f'Heading error:  mean={statistics.mean(yaw_errors):.1f}deg '
            f'max={max(yaw_errors):.1f}deg'
        )

    node.destroy_node()

    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
