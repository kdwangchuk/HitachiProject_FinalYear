#!/usr/bin/env python3

import csv
import math
import os
import time

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

from std_msgs.msg import Int32, Float32, Bool, ColorRGBA
from geometry_msgs.msg import PoseStamped, Pose2D, Vector3
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray

from hitachi_navigation.marker_geometry import (
    marker_in_base_frame,
    marker_pose_from_robot,
    robot_pose_from_marker,
)
from hitachi_navigation.pose_ekf import PoseEKF


def normalize_angle(angle):
    """Wrap an angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def quaternion_to_yaw(q):
    """Yaw angle (rad) from a geometry_msgs Quaternion."""
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    )


def yaw_to_quaternion(yaw):
    """(x, y, z, w) for a geometry_msgs Quaternion from a yaw angle."""
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


class RobotLocalizer(Node):
    """
    Fuses wheel odometry (/tb3_1/odom - continuous, drifts) with
    ArUco marker detections (/aruco_id, /aruco_pose_camera - occasional,
    drift-free) to estimate the robot's pose in the map frame, using
    the known marker positions in aruco_locations.csv.

    Markers are assumed to lie FLAT ON THE FLOOR (see marker_geometry.py
    for the camera/marker conventions, including camera pitch).

    No TF tree is required (hardware currently broadcasts none).

    use_ekf=True (default): an extended Kalman filter (pose_ekf.py,
    after Bach et al., IEEE Access 2023). Odometry drives the
    prediction step, every accepted marker fix is a correction, and the
    filter's covariance says how much the pose can be trusted.

    use_ekf=False: the older hard snap - each fix overwrites a plain
    SE(2) map<->odom offset, so the published pose is odom carried
    through that offset.
    """

    def __init__(self):
        super().__init__('robot_localizer')

        # =====================================================
        # PARAMETERS - MARKER MAP
        # =====================================================

        default_csv = os.path.join(
            get_package_share_directory('hitachi_navigation'),
            'config',
            'aruco_locations.csv'
        )
        self.declare_parameter('aruco_locations_csv', default_csv)
        self.aruco_locations_csv = str(
            self.get_parameter('aruco_locations_csv').value
        )
        self.marker_map = self.load_marker_locations(self.aruco_locations_csv)

        # =====================================================
        # PARAMETERS - CAMERA MOUNTING OFFSET
        #
        # base_footprint -> camera, in the robot's body frame.
        # NOT calibrated yet - defaults to zero (camera assumed at
        # the robot's origin, facing forward). Measure the real
        # mount position/heading and set these before trusting fixes.
        # =====================================================

        self.declare_parameter('camera_forward_offset', 0.0)  # meters
        self.declare_parameter('camera_left_offset', 0.0)      # meters
        self.declare_parameter('camera_yaw_offset', 0.0)       # radians

        self.camera_forward_offset = float(
            self.get_parameter('camera_forward_offset').value
        )
        self.camera_left_offset = float(
            self.get_parameter('camera_left_offset').value
        )
        self.camera_yaw_offset = float(
            self.get_parameter('camera_yaw_offset').value
        )

        # =====================================================
        # PARAMETERS - FIX RELIABILITY
        # =====================================================

        # Only trust a marker fix when close enough. A 3cm marker's
        # pose estimate gets noisy at range (see aruco_detector.py
        # marker_size fix) - this bounds how far out we'll accept it.
        self.declare_parameter('max_fix_distance', 0.40)  # meters
        self.max_fix_distance = float(
            self.get_parameter('max_fix_distance').value
        )

        # Reject a detection if it's older than this (subscriber
        # queue lag / stale data).
        self.declare_parameter('max_fix_age', 0.3)  # seconds
        self.max_fix_age = float(
            self.get_parameter('max_fix_age').value
        )

        # How long after the last successful marker fix the pose is
        # still considered "localized" (trustworthy) rather than
        # "coasting" on odometry drift alone.
        self.declare_parameter('localized_timeout', 1.0)  # seconds
        self.localized_timeout = float(
            self.get_parameter('localized_timeout').value
        )

        # Camera tilt for FLOOR markers. The marker distance the camera
        # reports is the straight line to the tag, which includes the
        # camera's height above the floor, so the tilt is needed to get
        # the ground distance. pitch = angle the optical axis points
        # DOWN from horizontal. The default is only an estimate read
        # off a debug-image screenshot - tune it until the logged
        # "cam_h" matches camera_height (see the LOCALIZATION FIX log).
        self.declare_parameter('camera_pitch_deg', 25.0)  # degrees, down
        self.declare_parameter('camera_height', 0.137)    # m, diagnostics
        self.camera_pitch = math.radians(
            float(self.get_parameter('camera_pitch_deg').value)
        )
        self.camera_height = float(
            self.get_parameter('camera_height').value
        )

        # A floor marker's normal must point roughly straight up. If a
        # detection's normal doesn't (planar-pose ambiguity flips it on
        # small markers), the whole detection is unreliable - drop it.
        # 1.0 = perfectly flat, 0.5 = 60 degrees off.
        self.declare_parameter('min_normal_up', 0.5)
        self.min_normal_up = float(
            self.get_parameter('min_normal_up').value
        )

        # Extra fixed bias added to every marker's heading in the robot
        # frame (radians). Normally leave at 0.0 and set the yaw column
        # of aruco_locations.csv correctly instead.
        self.declare_parameter('marker_yaw_correction', 0.0)  # radians
        self.marker_yaw_correction = float(
            self.get_parameter('marker_yaw_correction').value
        )

        # =====================================================
        # PARAMETERS - EXTENDED KALMAN FILTER
        # =====================================================

        self.declare_parameter('use_ekf', True)
        self.use_ekf = bool(self.get_parameter('use_ekf').value)

        # Process noise: how fast odometry uncertainty grows.
        self.declare_parameter('ekf_q_xy_per_m', 0.002)      # m^2 per m
        self.declare_parameter('ekf_q_theta_per_rad', 0.01)  # rad^2 per rad
        self.declare_parameter('ekf_q_theta_per_m', 0.001)   # rad^2 per m
        self.declare_parameter('ekf_q_per_sec', 1.0e-5)      # baseline / s

        # Measurement noise of a marker fix at ekf_r_distance_ref;
        # grows with the square of the marker's distance beyond that.
        self.declare_parameter('ekf_r_xy', 0.0025)           # m^2 (5 cm)
        self.declare_parameter('ekf_r_theta', 0.008)         # rad^2 (5 deg)
        self.declare_parameter('ekf_r_distance_ref', 0.20)   # m

        # Reject a fix whose normalised innovation (Mahalanobis d^2,
        # 3 dof) exceeds this; 16.27 is the 99.9% chi-square point.
        self.declare_parameter('ekf_gate', 16.27)

        # "localized" while the filter's own uncertainty stays below:
        self.declare_parameter('ekf_localized_std_xy', 0.15)     # m
        self.declare_parameter('ekf_localized_std_theta', 0.20)  # rad

        self.ekf = PoseEKF(
            q_xy_per_m=float(self.get_parameter('ekf_q_xy_per_m').value),
            q_theta_per_rad=float(
                self.get_parameter('ekf_q_theta_per_rad').value
            ),
            q_theta_per_m=float(
                self.get_parameter('ekf_q_theta_per_m').value
            ),
            q_per_sec=float(self.get_parameter('ekf_q_per_sec').value),
            r_xy=float(self.get_parameter('ekf_r_xy').value),
            r_theta=float(self.get_parameter('ekf_r_theta').value),
            r_distance_ref=float(
                self.get_parameter('ekf_r_distance_ref').value
            ),
            gate=float(self.get_parameter('ekf_gate').value),
        )
        self.ekf_localized_std_xy = float(
            self.get_parameter('ekf_localized_std_xy').value
        )
        self.ekf_localized_std_theta = float(
            self.get_parameter('ekf_localized_std_theta').value
        )

        # Hard sanity bound: reject any fix that lands outside the
        # physically possible track extent. A single bad detection
        # (motion blur, a misread corner) can otherwise get fused
        # with full confidence and lock the offset onto a wrong
        # belief that persists until the next good fix. Defaults are
        # wide open (disabled) until you set them to your track's
        # real bounds.
        self.declare_parameter('fix_bounds_x_min', -1.0e6)
        self.declare_parameter('fix_bounds_x_max', 1.0e6)
        self.declare_parameter('fix_bounds_y_min', -1.0e6)
        self.declare_parameter('fix_bounds_y_max', 1.0e6)
        self.fix_bounds_x_min = float(
            self.get_parameter('fix_bounds_x_min').value
        )
        self.fix_bounds_x_max = float(
            self.get_parameter('fix_bounds_x_max').value
        )
        self.fix_bounds_y_min = float(
            self.get_parameter('fix_bounds_y_min').value
        )
        self.fix_bounds_y_max = float(
            self.get_parameter('fix_bounds_y_max').value
        )

        # =====================================================
        # STATE - LATEST DETECTION
        # =====================================================

        self.latest_marker_id = None
        self.latest_marker_pose = None  # geometry_msgs/Pose, camera frame
        self.latest_marker_distance = None
        self.latest_marker_stamp = None

        # Each new /aruco_pose_camera message bumps detection_seq. A
        # Kalman filter must not fuse the same measurement twice (it
        # would become over-confident), so every consumer remembers the
        # last sequence number it processed.
        self.detection_seq = 0
        self.last_fix_seq = -1
        self.last_survey_seq = -1
        self.latest_diag = None  # MarkerInBase of the latest detection

        # =====================================================
        # STATE - ODOMETRY
        # =====================================================

        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_yaw = 0.0
        self.odom_received = False
        self.last_odom_time = time.monotonic()

        # =====================================================
        # STATE - MAP<->ODOM OFFSET
        #
        # map_pose = odom_pose (+) offset (SE(2) composition).
        # Starts at zero (map assumed == odom) until the first fix.
        # =====================================================

        self.offset_x = 0.0
        self.offset_y = 0.0
        self.offset_yaw = 0.0
        self.have_fix = False
        self.last_fix_wall_time = None

        # Periodic heartbeat so "no marker in range" (silent, expected)
        # is visibly different from "node stopped/crashed".
        self.status_log_interval = 2.0  # seconds
        self.last_status_log = time.monotonic()

        # =====================================================
        # STATE - AUTO-SURVEYING NEW (UNKNOWN) MARKERS
        #
        # While the robot's own position is currently trustworthy
        # (is_localized), any detected marker NOT already in
        # marker_map gets its implied map position computed and
        # accumulated here - the inverse of the normal fix: known
        # robot pose + camera detection -> new marker's map pose,
        # instead of known marker pose + detection -> robot's map
        # pose. Lets you survey a new marker just by driving past it
        # near a known one, instead of measuring it by hand.
        # =====================================================

        self.is_localized = False
        self.last_map_x = 0.0
        self.last_map_y = 0.0
        self.last_map_yaw = 0.0

        self.survey_samples = {}  # marker_id -> list of (x, y, yaw)

        self.declare_parameter('survey_max_samples', 50)
        self.survey_max_samples = int(
            self.get_parameter('survey_max_samples').value
        )

        self.survey_report_interval = 2.0  # seconds
        self.last_survey_report = time.monotonic()

        # =====================================================
        # SUBSCRIBERS
        # =====================================================

        self.create_subscription(
            Int32, '/aruco_id', self.aruco_id_callback, 10
        )
        self.create_subscription(
            PoseStamped, '/aruco_pose_camera', self.aruco_pose_callback, 10
        )
        self.create_subscription(
            Float32, '/aruco_distance', self.aruco_distance_callback, 10
        )
        self.create_subscription(
            Odometry, '/tb3_1/odom', self.odom_callback, 10
        )

        # =====================================================
        # PUBLISHER
        # =====================================================

        self.pose_pub = self.create_publisher(Pose2D, '/robot_pose_map', 10)
        self.localized_pub = self.create_publisher(Bool, '/localized', 10)

        # =====================================================
        # PUBLISHERS - RVIZ VISUALIZATION (DEMO)
        #
        # These are for visually demonstrating localization, not
        # used by any other node. Everything is published directly
        # in a "map" frame - no TF broadcaster is needed for this:
        # RViz needs no transform lookup when a message's frame_id
        # already matches its Fixed Frame, so just typing "map"
        # into RViz's Fixed Frame field (it won't be in the
        # dropdown, since nothing broadcasts it) is enough.
        # =====================================================

        self.viz_localized_pub = self.create_publisher(
            Odometry, '/viz/localized_pose', 10
        )
        self.viz_odom_only_pub = self.create_publisher(
            Odometry, '/viz/odom_only_pose', 10
        )
        self.viz_markers_pub = self.create_publisher(
            MarkerArray, '/viz/known_markers', 10
        )
        self.known_markers_viz = self.build_known_markers_viz()
        self.create_timer(2.0, self.publish_known_markers_viz)

        self.create_timer(1.0 / 10.0, self.update)

        self.get_logger().info(
            f'RobotLocalizer started - {len(self.marker_map)} known markers '
            f'from {self.aruco_locations_csv}, max_fix_distance='
            f'{self.max_fix_distance}m'
        )

        self.get_logger().info(
            f'Mode: {"EKF" if self.use_ekf else "hard snap"} | floor markers'
            f' | camera pitch={math.degrees(self.camera_pitch):.1f}deg '
            f'forward_offset={self.camera_forward_offset:.3f}m'
        )

        if self.camera_forward_offset == 0.0 and self.camera_left_offset == 0.0:
            self.get_logger().warn(
                'camera_forward_offset/camera_left_offset are still 0.0 - '
                'the camera mount offset has not been calibrated, fixes '
                'will be off by that amount'
            )

    # =========================================================
    # MARKER MAP LOADING
    # =========================================================

    def load_marker_locations(self, path):
        """Load marker_id -> (x, y, yaw) from the CSV map file."""

        markers = {}

        try:
            with open(path, newline='') as f:
                for row in csv.DictReader(f):
                    markers[int(row['marker_id'])] = (
                        float(row['x']),
                        float(row['y']),
                        float(row['yaw']),
                    )
        except (OSError, KeyError, ValueError) as exc:
            self.get_logger().error(f'Failed to load {path}: {exc}')

        return markers

    # =========================================================
    # RVIZ VISUALIZATION (DEMO)
    # =========================================================

    def build_known_markers_viz(self):
        """MarkerArray of every known marker's map position - a
        sphere plus a text label, so the audience can see the
        robot's estimated pose move relative to real landmarks."""

        array = MarkerArray()

        for marker_id, (x, y, _yaw) in self.marker_map.items():
            sphere = Marker()
            sphere.header.frame_id = 'map'
            sphere.ns = 'known_markers'
            sphere.id = marker_id * 2
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x = x
            sphere.pose.position.y = y
            sphere.pose.position.z = 0.05
            sphere.pose.orientation.w = 1.0
            sphere.scale = Vector3(x=0.06, y=0.06, z=0.06)
            sphere.color = ColorRGBA(r=0.1, g=0.4, b=1.0, a=1.0)
            array.markers.append(sphere)

            label = Marker()
            label.header.frame_id = 'map'
            label.ns = 'known_marker_labels'
            label.id = marker_id * 2 + 1
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = x
            label.pose.position.y = y
            label.pose.position.z = 0.18
            label.pose.orientation.w = 1.0
            label.scale = Vector3(x=0.0, y=0.0, z=0.09)
            label.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            label.text = f'id {marker_id}'
            array.markers.append(label)

        return array

    def publish_known_markers_viz(self):
        now = self.get_clock().now().to_msg()

        for m in self.known_markers_viz.markers:
            m.header.stamp = now

        self.viz_markers_pub.publish(self.known_markers_viz)

    # =========================================================
    # CALLBACKS
    # =========================================================

    def aruco_id_callback(self, msg):
        self.latest_marker_id = int(msg.data)

    def aruco_pose_callback(self, msg):
        self.latest_marker_pose = msg.pose
        self.latest_marker_stamp = self.get_clock().now()
        self.detection_seq += 1

    def aruco_distance_callback(self, msg):
        self.latest_marker_distance = float(msg.data)

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        yaw = quaternion_to_yaw(msg.pose.pose.orientation)
        now = time.monotonic()

        if not self.odom_received:
            # Until a marker is seen, "map" is assumed equal to odom.
            self.ekf.set_state(x, y, yaw)
        elif self.use_ekf:
            # EKF prediction from the odometry increment (paper Eq. 6-8):
            # distance moved along the heading, and heading change.
            dth = normalize_angle(yaw - self.odom_yaw)
            mid = self.odom_yaw + dth / 2.0
            s = (
                (x - self.odom_x) * math.cos(mid)
                + (y - self.odom_y) * math.sin(mid)
            )
            self.ekf.predict(s, dth, now - self.last_odom_time)

        self.odom_x = x
        self.odom_y = y
        self.odom_yaw = yaw
        self.last_odom_time = now
        self.odom_received = True

    # =========================================================
    # FIX COMPUTATION
    # =========================================================

    def get_fresh_detection(self):
        """Return (seq, marker_id, marker, distance) for the current
        detection if it's usable (recent, in range, odom available, marker
        actually lying flat), else None.

        marker is a marker_geometry.MarkerInBase: where the marker sits
        and which way it points relative to the robot's base_footprint on
        the ground plane. It's independent of whether the marker's map
        position is known, which is what lets try_update_fix() (known
        marker -> robot's map pose) and try_survey_unknown_marker()
        (known robot pose -> new marker's map pose) share it."""

        if self.latest_marker_id is None or self.latest_marker_pose is None:
            return None
        if (
            self.latest_marker_distance is None
            or self.latest_marker_distance > self.max_fix_distance
        ):
            return None
        if self.latest_marker_stamp is None:
            return None

        age = (
            self.get_clock().now() - self.latest_marker_stamp
        ).nanoseconds / 1e9

        if age > self.max_fix_age:
            return None  # stale detection

        if not self.odom_received:
            return None

        p = self.latest_marker_pose.position
        q = self.latest_marker_pose.orientation

        marker = marker_in_base_frame(
            (p.x, p.y, p.z),
            (q.x, q.y, q.z, q.w),
            self.camera_forward_offset,
            self.camera_left_offset,
            self.camera_yaw_offset,
            self.camera_pitch,
        )

        self.latest_diag = marker

        # A floor marker's normal must point up. If not, the pose
        # estimate has flipped (planar ambiguity) - don't trust it.
        if marker.normal_up < self.min_normal_up:
            self.get_logger().warn(
                f'Ignoring marker {self.latest_marker_id}: its normal is '
                f'not pointing up (normal_up={marker.normal_up:.2f} < '
                f'{self.min_normal_up:.2f}) - unreliable pose',
                throttle_duration_sec=2.0
            )
            return None

        marker = marker._replace(
            heading=normalize_angle(
                marker.heading + self.marker_yaw_correction
            )
        )

        return (
            self.detection_seq,
            self.latest_marker_id,
            marker,
            self.latest_marker_distance,
        )

    def diag_string(self):
        """Calibration hints from the latest detection, for the logs."""

        d = self.latest_diag

        if d is None:
            return ''

        return (
            f'[marker_hdg_in_robot={math.degrees(d.heading):.1f}deg '
            f'cam_h={d.cam_height:.3f}m (want {self.camera_height:.3f}) '
            f'pitch_implied={math.degrees(d.implied_pitch):.1f}deg '
            f'(using {math.degrees(self.camera_pitch):.1f})]'
        )

    def try_update_fix(self):
        """If a fresh, reliable detection of a KNOWN marker is
        available, use it to correct the pose estimate."""

        detection = self.get_fresh_detection()

        if detection is None:
            return

        seq, marker_id, marker, distance = detection

        if marker_id not in self.marker_map:
            return

        if seq == self.last_fix_seq:
            return  # this detection was already used

        self.last_fix_seq = seq

        # The robot's map pose implied by this one detection.
        fix_x, fix_y, fix_yaw = robot_pose_from_marker(
            self.marker_map[marker_id],
            marker.forward,
            marker.left,
            marker.heading,
        )

        # Hard sanity bound: reject a fix that lands outside the
        # physically possible track extent, before it can reach the
        # filter at all.
        if (
            fix_x < self.fix_bounds_x_min or fix_x > self.fix_bounds_x_max
            or fix_y < self.fix_bounds_y_min or fix_y > self.fix_bounds_y_max
        ):
            self.get_logger().warn(
                f'REJECTED implausible fix from marker '
                f'{marker_id}: ({fix_x:.3f}, {fix_y:.3f}) '
                f'is outside configured track bounds'
            )
            return

        if self.use_ekf:
            accepted, d2, status = self.ekf.update(
                (fix_x, fix_y, fix_yaw), distance
            )

            if not accepted:
                self.get_logger().warn(
                    f'EKF rejected fix from marker {marker_id}: '
                    f'z=({fix_x:.3f}, {fix_y:.3f}, '
                    f'{math.degrees(fix_yaw):.1f}deg) is statistically '
                    f'inconsistent with the estimate (d2={d2:.1f} > '
                    f'{self.ekf.gate:.1f})',
                    throttle_duration_sec=1.0
                )
                return

            self.have_fix = True
            self.last_fix_wall_time = time.monotonic()

            ex, ey, eth = self.ekf.x
            self.get_logger().info(
                f'LOCALIZATION FIX from marker {marker_id}: '
                f'map=({ex:.3f}, {ey:.3f}, {math.degrees(eth):.1f}deg) '
                f'z=({fix_x:.3f}, {fix_y:.3f}, '
                f'{math.degrees(fix_yaw):.1f}deg) '
                f'std=({100.0 * self.ekf.std_xy():.1f}cm, '
                f'{math.degrees(self.ekf.std_theta()):.1f}deg) '
                f'd2={d2:.1f} {status} {self.diag_string()}'
            )
            return

        # ---- hard-snap mode: recompute the offset so that
        # odom (+) offset == fix, now.
        self.offset_yaw = normalize_angle(fix_yaw - self.odom_yaw)

        co = math.cos(self.odom_yaw)
        so = math.sin(self.odom_yaw)
        dx = fix_x - self.odom_x
        dy = fix_y - self.odom_y
        self.offset_x = co * dx + so * dy
        self.offset_y = -so * dx + co * dy
        self.have_fix = True
        self.last_fix_wall_time = time.monotonic()

        self.get_logger().info(
            f'LOCALIZATION FIX from marker {marker_id}: '
            f'map=({fix_x:.3f}, {fix_y:.3f}, {math.degrees(fix_yaw):.1f}deg) '
            f'odom=({self.odom_x:.3f}, {self.odom_y:.3f}, '
            f'{math.degrees(self.odom_yaw):.1f}deg) {self.diag_string()}'
        )

    # =========================================================
    # AUTO-SURVEYING NEW (UNKNOWN) MARKERS
    # =========================================================

    def try_survey_unknown_marker(self):
        """If currently localized and looking at a marker NOT already
        in marker_map, compute where that marker must be in the map
        frame given our own trusted position, and accumulate the
        sample for later averaging/reporting."""

        if not self.is_localized:
            return  # our own position isn't trustworthy right now

        detection = self.get_fresh_detection()

        if detection is None:
            return

        seq, marker_id, marker, _distance = detection

        if marker_id in self.marker_map:
            return  # already known, nothing to survey

        if seq == self.last_survey_seq:
            return  # this detection was already sampled

        self.last_survey_seq = seq

        marker_map_x, marker_map_y, marker_map_yaw = marker_pose_from_robot(
            (self.last_map_x, self.last_map_y, self.last_map_yaw),
            marker.forward,
            marker.left,
            marker.heading,
        )

        samples = self.survey_samples.setdefault(marker_id, [])
        samples.append((marker_map_x, marker_map_y, marker_map_yaw))

        if len(samples) > self.survey_max_samples:
            samples.pop(0)

    def report_survey_progress(self):
        """Periodically print the averaged position of every
        in-progress unknown-marker survey, ready to paste into
        aruco_locations.csv once it looks stable."""

        for marker_id, samples in self.survey_samples.items():
            xs = [s[0] for s in samples]
            ys = [s[1] for s in samples]
            sin_sum = sum(math.sin(s[2]) for s in samples)
            cos_sum = sum(math.cos(s[2]) for s in samples)

            mean_x = sum(xs) / len(xs)
            mean_y = sum(ys) / len(ys)
            mean_yaw = math.atan2(sin_sum, cos_sum)

            self.get_logger().info(
                f'SURVEYING marker {marker_id} ({len(samples)} samples): '
                f'{marker_id},{mean_x:.3f},{mean_y:.3f},{mean_yaw:.4f}  '
                f'<- paste into aruco_locations.csv once stable '
                f'({math.degrees(mean_yaw):.1f}deg)'
            )

    # =========================================================
    # MAIN LOOP (10 Hz)
    # =========================================================

    def update(self):
        self.try_update_fix()

        if not self.odom_received:
            return

        if self.use_ekf:
            map_x, map_y, map_yaw = (
                float(self.ekf.x[0]),
                float(self.ekf.x[1]),
                float(self.ekf.x[2]),
            )
        else:
            # map_pose = odom_pose (+) offset
            co = math.cos(self.odom_yaw)
            so = math.sin(self.odom_yaw)

            map_x = self.odom_x + co * self.offset_x - so * self.offset_y
            map_y = self.odom_y + so * self.offset_x + co * self.offset_y
            map_yaw = normalize_angle(self.odom_yaw + self.offset_yaw)

        msg = Pose2D()
        msg.x = map_x
        msg.y = map_y
        msg.theta = map_yaw
        self.pose_pub.publish(msg)

        # /localized: the explicit, glanceable answer to "can I trust
        # this pose right now?"
        #   EKF mode:  True while the filter has had at least one marker
        #              fix AND its own uncertainty is still small
        #              (ekf_localized_std_*) - so it stays True for a few
        #              metres of driving after a fix, then drops as the
        #              odometry drift uncertainty accumulates.
        #   snap mode: True only for localized_timeout seconds after the
        #              last marker fix.
        now = time.monotonic()

        if self.last_fix_wall_time is None:
            fix_age = None
        else:
            fix_age = now - self.last_fix_wall_time

        if self.use_ekf:
            is_localized = (
                self.have_fix
                and self.ekf.std_xy() <= self.ekf_localized_std_xy
                and self.ekf.std_theta() <= self.ekf_localized_std_theta
            )
        else:
            is_localized = (
                fix_age is not None and fix_age <= self.localized_timeout
            )

        localized_msg = Bool()
        localized_msg.data = is_localized
        self.localized_pub.publish(localized_msg)

        # Make this cycle's trusted position available to
        # try_survey_unknown_marker(), then try it - only does
        # anything while is_localized is True and a currently-seen
        # marker isn't already in marker_map.
        self.is_localized = is_localized
        self.last_map_x = map_x
        self.last_map_y = map_y
        self.last_map_yaw = map_yaw
        self.try_survey_unknown_marker()

        if now - self.last_survey_report >= self.survey_report_interval:
            self.last_survey_report = now
            self.report_survey_progress()

        # RViz demo: the corrected estimate (green, should track the
        # real track/markers) vs. raw odometry with zero correction
        # applied (red, will visibly wander off as it drifts) - both
        # plotted in the same "map" frame for direct comparison.
        stamp = self.get_clock().now().to_msg()

        localized_odom = Odometry()
        localized_odom.header.stamp = stamp
        localized_odom.header.frame_id = 'map'
        localized_odom.child_frame_id = 'base_footprint'
        localized_odom.pose.pose.position.x = map_x
        localized_odom.pose.pose.position.y = map_y
        (qx, qy, qz, qw) = yaw_to_quaternion(map_yaw)
        localized_odom.pose.pose.orientation.x = qx
        localized_odom.pose.pose.orientation.y = qy
        localized_odom.pose.pose.orientation.z = qz
        localized_odom.pose.pose.orientation.w = qw

        if self.use_ekf:
            # Fill the x/y/yaw block of the 6x6 covariance so RViz's
            # Covariance option draws the filter's uncertainty (it
            # visibly grows while driving and shrinks at each marker).
            P = self.ekf.P
            cov = [0.0] * 36
            for i, ri in enumerate((0, 1, 5)):
                for j, rj in enumerate((0, 1, 5)):
                    cov[6 * ri + rj] = float(P[i, j])
            localized_odom.pose.covariance = cov

        self.viz_localized_pub.publish(localized_odom)

        odom_only = Odometry()
        odom_only.header.stamp = stamp
        odom_only.header.frame_id = 'map'
        odom_only.child_frame_id = 'base_footprint'
        odom_only.pose.pose.position.x = self.odom_x
        odom_only.pose.pose.position.y = self.odom_y
        (qx, qy, qz, qw) = yaw_to_quaternion(self.odom_yaw)
        odom_only.pose.pose.orientation.x = qx
        odom_only.pose.pose.orientation.y = qy
        odom_only.pose.pose.orientation.z = qz
        odom_only.pose.pose.orientation.w = qw
        self.viz_odom_only_pub.publish(odom_only)

        # Heartbeat: makes it visible that the node is alive and
        # publishing even when no marker is currently in range to
        # log a fresh LOCALIZATION FIX (that's expected/silent
        # otherwise, and easy to mistake for the node being stuck).
        if now - self.last_status_log >= self.status_log_interval:
            self.last_status_log = now

            fix_age_str = 'never' if fix_age is None else f'{fix_age:.1f}s ago'

            std_str = ''
            if self.use_ekf:
                std_str = (
                    f' std=({100.0 * self.ekf.std_xy():.1f}cm, '
                    f'{math.degrees(self.ekf.std_theta()):.1f}deg)'
                )

            self.get_logger().info(
                f'STATUS: map=({map_x:.3f}, {map_y:.3f}, '
                f'{math.degrees(map_yaw):.1f}deg) '
                f'localized={is_localized} last_fix={fix_age_str}{std_str}'
            )


def main(args=None):

    rclpy.init(args=args)

    node = RobotLocalizer()

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
