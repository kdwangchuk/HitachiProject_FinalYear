#!/usr/bin/env python3

import math

import cv2
import numpy as np

import rclpy

from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, Float32, Int32


class HardwareArucoDetector(Node):

    def __init__(self):
        super().__init__('hardware_aruco_detector')

        # ==========================================================
        # PARAMETERS
        # ==========================================================

        self.declare_parameter(
            'camera_topic',
            '/webcam/image/compressed'
        )

        # IMPORTANT:
        # Change this to the REAL printed black-square side length.
        #
        # Example:
        # 20 cm marker -> 0.20
        # 15 cm marker -> 0.15
        self.declare_parameter(
            'marker_size',
            0.03
        )

        # Physical distance at which marker becomes CLOSE.
        self.declare_parameter(
            'close_distance',
            0.20
        )

        self.camera_topic = str(
            self.get_parameter(
                'camera_topic'
            ).value
        )

        self.marker_size = float(
            self.get_parameter(
                'marker_size'
            ).value
        )

        self.close_distance = float(
            self.get_parameter(
                'close_distance'
            ).value
        )

        # ==========================================================
        # CALIBRATED LOGITECH C930e CAMERA
        # ==========================================================

        self.camera_matrix = np.array(
            [
                [
                    499.682974,
                    0.0,
                    334.735357
                ],
                [
                    0.0,
                    499.022855,
                    259.555019
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            dtype=np.float64
        )

        self.dist_coeffs = np.array(
            [
                0.072010,
                -0.124825,
                -0.004650,
                0.004907,
                0.000000
            ],
            dtype=np.float64
        )

        # ==========================================================
        # ARUCO DICTIONARY
        #
        # Must match your printed markers.
        # Your simulation used DICT_4X4_50.
        # ==========================================================

        self.dictionary = (
            cv2.aruco.Dictionary_get(
                cv2.aruco.DICT_4X4_50
            )
        )

        self.detector_parameters = (
            cv2.aruco.DetectorParameters_create()
        )

        # ==========================================================
        # SUBSCRIBER
        # ==========================================================

        self.image_sub = self.create_subscription(
            CompressedImage,
            self.camera_topic,
            self.image_callback,
            10
        )

        # ==========================================================
        # PUBLISHERS
        # ==========================================================

        self.id_pub = self.create_publisher(
            Int32,
            '/aruco_id',
            10
        )

        self.distance_pub = self.create_publisher(
            Float32,
            '/aruco_distance',
            10
        )

        self.close_id_pub = self.create_publisher(
            Int32,
            '/aruco_close_id',
            10
        )

        # NOTE: line_follower_test.py subscribes to this topic to
        # slow down / go straight while approaching a marker.
        # It was never published before - that is why the
        # "ARUCO APPROACH" slow-down never triggered.
        self.visible_pub = self.create_publisher(
            Bool,
            '/aruco_visible',
            10
        )

        self.pose_pub = self.create_publisher(
            PoseStamped,
            '/aruco_pose_camera',
            10
        )

        self.debug_pub = self.create_publisher(
            CompressedImage,
            '/aruco_debug_image/compressed',
            10
        )

        # Prevent CLOSE from publishing every single frame.
        self.last_close_id = None

        self.frame_count = 0

        self.get_logger().info(
            'Hardware ArUco detector started'
        )

        self.get_logger().info(
            f'Camera topic: {self.camera_topic}'
        )

        self.get_logger().info(
            f'Marker size: {self.marker_size:.3f} m'
        )

        self.get_logger().info(
            f'Close distance: {self.close_distance:.3f} m'
        )

        self.get_logger().info(
            'NO MOTOR COMMANDS ARE PUBLISHED'
        )

    # ==============================================================
    # ROTATION MATRIX -> QUATERNION
    # ==============================================================

    def rotation_matrix_to_quaternion(
        self,
        rotation_matrix
    ):

        m = rotation_matrix

        trace = (
            m[0, 0]
            + m[1, 1]
            + m[2, 2]
        )

        if trace > 0.0:

            s = math.sqrt(
                trace + 1.0
            ) * 2.0

            qw = 0.25 * s
            qx = (
                m[2, 1] - m[1, 2]
            ) / s
            qy = (
                m[0, 2] - m[2, 0]
            ) / s
            qz = (
                m[1, 0] - m[0, 1]
            ) / s

        elif (
            m[0, 0] > m[1, 1]
            and
            m[0, 0] > m[2, 2]
        ):

            s = math.sqrt(
                1.0
                + m[0, 0]
                - m[1, 1]
                - m[2, 2]
            ) * 2.0

            qw = (
                m[2, 1] - m[1, 2]
            ) / s

            qx = 0.25 * s

            qy = (
                m[0, 1] + m[1, 0]
            ) / s

            qz = (
                m[0, 2] + m[2, 0]
            ) / s

        elif m[1, 1] > m[2, 2]:

            s = math.sqrt(
                1.0
                + m[1, 1]
                - m[0, 0]
                - m[2, 2]
            ) * 2.0

            qw = (
                m[0, 2] - m[2, 0]
            ) / s

            qx = (
                m[0, 1] + m[1, 0]
            ) / s

            qy = 0.25 * s

            qz = (
                m[1, 2] + m[2, 1]
            ) / s

        else:

            s = math.sqrt(
                1.0
                + m[2, 2]
                - m[0, 0]
                - m[1, 1]
            ) * 2.0

            qw = (
                m[1, 0] - m[0, 1]
            ) / s

            qx = (
                m[0, 2] + m[2, 0]
            ) / s

            qy = (
                m[1, 2] + m[2, 1]
            ) / s

            qz = 0.25 * s

        return (
            qx,
            qy,
            qz,
            qw
        )

    # ==============================================================
    # PUBLISH POSE
    # ==============================================================

    def publish_pose(
        self,
        marker_id,
        rvec,
        tvec,
        source_header
    ):

        rotation_matrix, _ = (
            cv2.Rodrigues(
                rvec
            )
        )

        (
            qx,
            qy,
            qz,
            qw
        ) = self.rotation_matrix_to_quaternion(
            rotation_matrix
        )

        pose = PoseStamped()

        pose.header = source_header

        # Marker position is relative to camera.
        pose.header.frame_id = (
            'camera_link'
        )

        pose.pose.position.x = float(
            tvec[0]
        )

        pose.pose.position.y = float(
            tvec[1]
        )

        pose.pose.position.z = float(
            tvec[2]
        )

        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw

        self.pose_pub.publish(
            pose
        )

    # ==============================================================
    # IMAGE CALLBACK
    # ==============================================================

    def image_callback(
        self,
        msg
    ):

        # ----------------------------------------------------------
        # Decode compressed JPEG
        # ----------------------------------------------------------

        np_arr = np.frombuffer(
            msg.data,
            dtype=np.uint8
        )

        frame = cv2.imdecode(
            np_arr,
            cv2.IMREAD_COLOR
        )

        if frame is None:

            self.get_logger().warning(
                'Failed to decode webcam frame'
            )

            return

        self.frame_count += 1

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        # ----------------------------------------------------------
        # Detect ArUco markers
        # ----------------------------------------------------------

        (
            corners,
            ids,
            rejected
        ) = cv2.aruco.detectMarkers(
            gray,
            self.dictionary,
            parameters=self.detector_parameters
        )

        debug = frame.copy()

        close_marker_visible = False

        # ----------------------------------------------------------
        # MARKERS FOUND
        # ----------------------------------------------------------

        markers_found = ids is not None and len(ids) > 0

        visible_msg = Bool()
        visible_msg.data = bool(markers_found)
        self.visible_pub.publish(visible_msg)

        if markers_found:

            cv2.aruco.drawDetectedMarkers(
                debug,
                corners,
                ids
            )

            # ------------------------------------------------------
            # Estimate marker position
            # ------------------------------------------------------

            (
                rvecs,
                tvecs,
                _
            ) = cv2.aruco.estimatePoseSingleMarkers(
                corners,
                self.marker_size,
                self.camera_matrix,
                self.dist_coeffs
            )

            # ------------------------------------------------------
            # Process every detected marker
            # ------------------------------------------------------

            detected_results = []

            for index, raw_id in enumerate(
                ids.flatten()
            ):

                marker_id = int(
                    raw_id
                )

                rvec = rvecs[
                    index
                ][0]

                tvec = tvecs[
                    index
                ][0]

                # Camera-coordinate translation:
                #
                # X = left/right
                # Y = up/down
                # Z = forward from camera

                tx = float(
                    tvec[0]
                )

                ty = float(
                    tvec[1]
                )

                tz = float(
                    tvec[2]
                )

                # True 3D Euclidean distance
                distance = math.sqrt(
                    tx * tx
                    + ty * ty
                    + tz * tz
                )

                detected_results.append(
                    (
                        distance,
                        marker_id,
                        rvec,
                        tvec
                    )
                )

                # Draw coordinate axis
                cv2.drawFrameAxes(
                    debug,
                    self.camera_matrix,
                    self.dist_coeffs,
                    rvec,
                    tvec,
                    self.marker_size * 0.5
                )

                # Marker center
                polygon = (
                    corners[index]
                    .reshape(4, 2)
                )

                center = np.mean(
                    polygon,
                    axis=0
                )

                cx = int(
                    center[0]
                )

                cy = int(
                    center[1]
                )

                # Green if physically close.
                if distance <= self.close_distance:

                    color = (
                        0,
                        255,
                        0
                    )

                    status = 'CLOSE'

                else:

                    color = (
                        0,
                        165,
                        255
                    )

                    status = 'FAR'

                cv2.putText(
                    debug,
                    (
                        f'ID {marker_id} '
                        f'{distance:.3f}m '
                        f'{status}'
                    ),
                    (
                        max(
                            5,
                            cx - 100
                        ),
                        max(
                            25,
                            cy - 30
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    color,
                    2
                )

            # ------------------------------------------------------
            # Select NEAREST marker
            # ------------------------------------------------------

            detected_results.sort(
                key=lambda result: result[0]
            )

            (
                nearest_distance,
                nearest_id,
                nearest_rvec,
                nearest_tvec
            ) = detected_results[0]

            # Publish ID
            id_msg = Int32()

            id_msg.data = (
                nearest_id
            )

            self.id_pub.publish(
                id_msg
            )

            # Publish physical distance
            distance_msg = Float32()

            distance_msg.data = float(
                nearest_distance
            )

            self.distance_pub.publish(
                distance_msg
            )

            # Publish camera-relative pose
            self.publish_pose(
                nearest_id,
                nearest_rvec,
                nearest_tvec,
                msg.header
            )

            # ------------------------------------------------------
            # CLOSE EVENT
            # ------------------------------------------------------

            if (
                nearest_distance
                <= self.close_distance
            ):

                close_marker_visible = True

                # Publish one close event when entering close range.
                if (
                    self.last_close_id
                    != nearest_id
                ):

                    close_msg = Int32()

                    close_msg.data = (
                        nearest_id
                    )

                    self.close_id_pub.publish(
                        close_msg
                    )

                    self.last_close_id = (
                        nearest_id
                    )

                    self.get_logger().info(
                        f'CLOSE ARUCO '
                        f'ID={nearest_id} | '
                        f'distance='
                        f'{nearest_distance:.3f} m'
                    )

            # If nearest marker is not close anymore,
            # allow another close event later.
            if not close_marker_visible:

                self.last_close_id = None

            cv2.putText(
                debug,
                (
                    f'NEAREST ID={nearest_id} '
                    f'D={nearest_distance:.3f}m'
                ),
                (
                    10,
                    30
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                (
                    0,
                    255,
                    0
                )
                if close_marker_visible
                else (
                    0,
                    165,
                    255
                ),
                2
            )

        else:

            # No markers currently visible.
            self.last_close_id = None

            cv2.putText(
                debug,
                'NO ARUCO DETECTED',
                (
                    10,
                    30
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                (
                    0,
                    0,
                    255
                ),
                2
            )

        # ----------------------------------------------------------
        # Publish compressed debug image
        # ----------------------------------------------------------

        success, encoded = cv2.imencode(
            '.jpg',
            debug
        )

        if success:

            debug_msg = CompressedImage()

            debug_msg.header = (
                msg.header
            )

            debug_msg.format = (
                'jpeg'
            )

            debug_msg.data = (
                encoded.tobytes()
            )

            self.debug_pub.publish(
                debug_msg
            )


def main(args=None):

    rclpy.init(args=args)

    node = HardwareArucoDetector()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()