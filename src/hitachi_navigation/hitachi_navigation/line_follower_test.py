#!/usr/bin/env python3

import math
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float32, Int32


def quaternion_to_yaw(q):
    """Yaw angle (rad) from a geometry_msgs Quaternion."""
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    )


def normalize_angle(angle):
    """Wrap an angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class LineFollowerCornerOut(Node):
    
    def __init__(self):
        super().__init__('line_follower_node')

        # =====================================================
        # PARAMETERS - SPEEDS
        # =====================================================

        self.declare_parameter('normal_speed', 0.10)  # Normal forward speed
        self.declare_parameter('medium_speed', 0.08)  # Correcting speed
        self.declare_parameter('marker_approach_speed', 0.055)

        self.normal_speed = float(self.get_parameter('normal_speed').value)
        self.medium_speed = float(self.get_parameter('medium_speed').value)
        self.marker_approach_speed = float(self.get_parameter('marker_approach_speed').value)

        # =====================================================
        # PARAMETERS - PI CONTROLLER (SMOOTH TUNING)
        # =====================================================

        self.declare_parameter('kp', 0.50)  # REDUCED from 0.80 (less aggressive, smoother)
        self.declare_parameter('ki', 0.08)  # INCREASED from 0.05 (better correction)
        self.declare_parameter('max_angular', 0.50)  # REDUCED from 0.70 (smoother turns)
        self.declare_parameter('deadband', 0.015)  # REDUCED from 0.025 (better precision)
        self.declare_parameter('integral_limit', 0.30)  # REDUCED from 0.50 (prevents overshoot)

        self.kp = float(self.get_parameter('kp').value)
        self.ki = float(self.get_parameter('ki').value)
        self.max_angular = float(self.get_parameter('max_angular').value)
        self.deadband = float(self.get_parameter('deadband').value)
        self.integral_limit = float(self.get_parameter('integral_limit').value)

        # Max change in angular.z allowed per control tick (~33ms @ 30Hz).
        # Without this, a noisy line_error or a mode switch (e.g. into/out
        # of a corner) can command a large angular step in one tick, which
        # reads as a jerk/oscillation on the real robot even though the PI
        # math itself is stable. This caps every published angular command,
        # regardless of source (line following, marker approach, recovery,
        # corner rotation), so transitions ramp instead of snapping.
        self.declare_parameter('max_angular_step', 0.08)  # rad/s per tick
        self.max_angular_step = float(self.get_parameter('max_angular_step').value)
        self.last_angular_cmd = 0.0

        # =====================================================
        # PARAMETERS - LINE OUT-OF-FRAME CORNER DETECTION
        # =====================================================

        # Go completely straight when approaching corner
        # (Set to False if you want slight steering corrections)
        self.declare_parameter('go_straight_to_corner', True)  # Recommended: True
        self.go_straight_to_corner = bool(self.get_parameter('go_straight_to_corner').value)

        # Timeout before triggering rotation when line disappears
        self.declare_parameter('line_loss_timeout', 0.3)  # seconds (0.1-0.5)
        self.line_loss_timeout = float(self.get_parameter('line_loss_timeout').value)

        # Corner forward offset - move this distance before rotating.
        #
        # This is NOT "how far away the corner is" - the vision system
        # already tells us that: detect_line_gone() fires once the tape
        # has vacated the WHOLE ROI (near band included, the closest
        # band to the robot), which is the moment the tape's trailing
        # edge has receded to less than the near band's ground distance
        # from the camera. corner_forward_offset only needs to cover the
        # small, FIXED mechanical gap between that near-band ground
        # point and the robot's actual turning center (wheel axis) - a
        # one-time physical measurement, not a guessed "distance to the
        # corner" to iterate on per track:
        #   1. Run the corner approach and watch /hardware/yellow_line_debug
        #      (or the near/far detected topics) for the exact moment the
        #      tape leaves the ROI.
        #   2. Mark the robot's position on the floor at that instant.
        #   3. Measure from that mark to the true tape corner with a
        #      tape measure - that measured distance is this value.
        # Re-measure only if the camera mount (height/tilt/position) or
        # ROI (yellow_line_detector.py: roi_top/roi_height) changes -
        # the value should NOT need retuning per corner shape/track.
        self.declare_parameter('corner_forward_offset', 0.12)  # meters (measured camera-near-band-to-wheelbase gap)
        self.declare_parameter('corner_forward_speed', 0.05)  # m/s (slow forward)

        self.corner_forward_offset = float(self.get_parameter('corner_forward_offset').value)
        self.corner_forward_speed = float(self.get_parameter('corner_forward_speed').value)

        # Rotation settings - odometry feedback controls when to stop,
        # this speed is just how fast it turns while doing so.
        self.declare_parameter('corner_rotation_speed', 0.25)  # rad/s (very slow)
        self.declare_parameter('corner_rotation_target_deg', 90.0)
        self.declare_parameter('corner_rotation_max_duration', 6.0)  # safety cap (seconds)

        self.corner_rotation_speed = float(self.get_parameter('corner_rotation_speed').value)
        self.corner_rotation_target = math.radians(
            float(self.get_parameter('corner_rotation_target_deg').value)
        )
        self.corner_rotation_max_duration = float(
            self.get_parameter('corner_rotation_max_duration').value
        )

        # Turn direction for every corner (sign matches angular.z:
        # + = left/CCW, - = right/CW). This track always turns right.
        self.declare_parameter('default_corner_turn_direction', -1.0)
        self.default_corner_turn_direction = float(
            self.get_parameter('default_corner_turn_direction').value
        )

        # After the blind rotation completes, ease back onto the line
        # gently for this long instead of handing off straight to
        # full-speed PI, so a slightly imperfect landing doesn't cause
        # a hard snap-correction toward the centroid.
        self.declare_parameter('corner_settle_duration', 0.6)  # seconds
        self.declare_parameter('corner_settle_speed', 0.05)  # m/s

        self.corner_settle_duration = float(self.get_parameter('corner_settle_duration').value)
        self.corner_settle_speed = float(self.get_parameter('corner_settle_speed').value)
        self.corner_settle_until = None

        # =====================================================
        # PARAMETERS - GENERAL
        # =====================================================

        self.declare_parameter('recovery_time', 1.0)
        self.declare_parameter('recovery_angular', 0.40)
        self.declare_parameter('recovery_linear', 0.02)

        self.recovery_time = float(self.get_parameter('recovery_time').value)
        self.recovery_angular = float(self.get_parameter('recovery_angular').value)
        self.recovery_linear = float(self.get_parameter('recovery_linear').value)

        self.declare_parameter('detector_timeout', 0.6)
        self.declare_parameter('stop_duration', 4.0)  # ArUco marker stop
        self.declare_parameter('marker_timeout', 0.40)

        self.detector_timeout = float(self.get_parameter('detector_timeout').value)
        self.stop_duration = float(self.get_parameter('stop_duration').value)
        self.marker_timeout = float(self.get_parameter('marker_timeout').value)

        # =====================================================
        # LINE STATE
        # =====================================================

        self.line_detected = False
        self.line_error = 0.0

        self.integral_error = 0.0

        now = time.monotonic()
        self.last_detection_time = now
        self.last_line_seen_time = now
        self.last_control_time = now
        self.last_turn_direction = 0.0

        # =====================================================
        # CORNER ROTATION STATE
        # =====================================================

        self.rotating_at_corner = False
        self.rotation_start_time = None
        self.line_loss_start_time = None

        self.corner_phase = None  # 'forward' | 'rotate' | None
        self.corner_start_x = None
        self.corner_start_y = None
        self.corner_start_yaw = None
        self.corner_turn_direction = None
        self.line_width = 0.0
        self.corner_width_detection_enabled = False
        self.corner_width_threshold = 250.0  # pixels - NOT VALIDATED
        self.corner_width_debounce = 0.15  # seconds
        self.width_spike_start_time = None
        self.line_near_error = 0.0
        self.line_far_error = 0.0
        self.line_near_detected = True
        self.line_far_detected = True
        self.corner_bend_detection_enabled = False
        self.corner_bend_threshold = 0.30
        self.corner_bend_debounce = 0.15  # seconds
        self.bend_spike_start_time = None

        # Diagnostic: print width/mode/error periodically so real
        # straight vs. corner width ranges can be read off a normal run.
        self.width_diagnostic_log_interval = 1.0  # seconds
        self.last_width_diagnostic_log = 0.0

        # =====================================================
        # ODOMETRY STATE
        # =====================================================

        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_yaw = 0.0
        self.odom_received = False

        # =====================================================
        # ARUCO STATE
        # =====================================================

        self.aruco_visible = False
        self.current_aruco_id = -1
        self.last_aruco_visible_time = 0.0

        self.stop_until = None
        self.stopping_marker_id = None
        self.inspected_ids = set()
        
        # Move to center phase
        # marker_close_distance mirrors aruco_detector.py's 'close_distance'
        # param (0.2m) - the distance at which /aruco_close_id fires and
        # triggers this forward move. Kept here only for the startup log.
        self.declare_parameter('marker_close_distance', 0.20)
        self.declare_parameter('marker_center_distance', 0.335)  # Move forward to reach center
        self.declare_parameter('marker_center_speed', 0.051)  # Slow speed (same as approach)

        # How long to suppress corner detection after resuming from a
        # marker stop, since the resume-time misalignment error can
        # otherwise be mistaken for a corner bend.
        self.declare_parameter('marker_resume_cooldown', 1.0)  # seconds

        self.moving_to_marker_center = False
        self.marker_center_start_time = None
        self.marker_close_distance = float(self.get_parameter('marker_close_distance').value)
        self.marker_center_distance = float(self.get_parameter('marker_center_distance').value)
        self.marker_center_speed = float(self.get_parameter('marker_center_speed').value)
        self.marker_resume_cooldown = float(self.get_parameter('marker_resume_cooldown').value)
        self.corner_detection_resume_until = 0.0

        # =====================================================
        # SUBSCRIBERS - LINE DETECTION
        # =====================================================

        self.create_subscription(
            Bool,
            '/hardware/line_detected',
            self.line_detected_callback,
            10
        )

        self.create_subscription(
            Float32,
            '/hardware/line_error',
            self.line_error_callback,
            10
        )

        self.create_subscription(
            Float32,
            '/hardware/line_width',
            self.line_width_callback,
            10
        )

        self.create_subscription(
            Float32,
            '/hardware/line_near_error',
            self.line_near_error_callback,
            10
        )

        self.create_subscription(
            Float32,
            '/hardware/line_far_error',
            self.line_far_error_callback,
            10
        )

        self.create_subscription(
            Bool,
            '/hardware/line_near_detected',
            self.line_near_detected_callback,
            10
        )

        self.create_subscription(
            Bool,
            '/hardware/line_far_detected',
            self.line_far_detected_callback,
            10
        )

        # =====================================================
        # SUBSCRIBERS - ARUCO MARKERS
        # =====================================================

        self.create_subscription(
            Bool,
            '/aruco_visible',
            self.aruco_visible_callback,
            10
        )

        self.create_subscription(
            Int32,
            '/aruco_id',
            self.aruco_id_callback,
            10
        )

        self.create_subscription(
            Int32,
            '/aruco_close_id',
            self.aruco_close_callback,
            10
        )

        # =====================================================
        # SUBSCRIBER - ODOMETRY (for corner forward-offset/rotation)
        # =====================================================

        self.create_subscription(
            Odometry,
            '/tb3_1/odom',
            self.odom_callback,
            10
        )

        # =====================================================
        # PUBLISHER - MOTOR COMMANDS
        # =====================================================

        self.cmd_pub = self.create_publisher(
            Twist,
            '/tb3_1/cmd_vel',
            10
        )

        # 30 Hz control loop
        self.timer = self.create_timer(
            1.0 / 30.0,
            self.control_loop
        )

        self.current_mode = None

        # =====================================================
        # STARTUP LOGGING
        # =====================================================

        self.get_logger().info(
            '======================================================'
        )
        self.get_logger().info(
            'LineFollowerNode - SMOOTH Line Following + Features'
        )
        self.get_logger().info(
            '======================================================'
        )
        self.get_logger().info(
            'PI Tuning: Kp=0.50, Ki=0.08 (SMOOTH)'
        )
        self.get_logger().info(
            'Max angular: 0.50 rad/s (smooth turns)'
        )
        self.get_logger().info(
            f'Corner: Forward {self.corner_forward_offset}m (odom) → Rotate 90deg at {self.corner_rotation_speed} rad/s (odom yaw feedback)'
        )
        self.get_logger().info(
            f'ArUco: Detect at {self.marker_close_distance}m → Move {self.marker_center_distance}m closer → Stop 4.0s'
        )
        self.get_logger().info(
            'Works for ALL markers and corners'
        )
        self.get_logger().info(
            '======================================================'
        )

    # =========================================================
    # CALLBACKS - LINE DETECTION
    # =========================================================

    def line_detected_callback(self, msg):
        """Called whenever line detection status changes."""
        self.line_detected = bool(msg.data)
        now = time.monotonic()
        self.last_detection_time = now

        if self.line_detected:
            # Line is visible again
            self.last_line_seen_time = now
            # Reset line loss timer
            self.line_loss_start_time = None

    def line_error_callback(self, msg):
        """Called with line centering error."""
        self.line_error = float(msg.data)

        # Remember turn direction for recovery
        if abs(self.line_error) > 0.05:
            if self.line_error > 0.0:
                self.last_turn_direction = -1.0
            else:
                self.last_turn_direction = 1.0

    def line_width_callback(self, msg):
        """Called with detected line width (pixels) - used to spot an
        L-junction corner even when the line never fully leaves frame."""
        self.line_width = float(msg.data)

    def line_near_error_callback(self, msg):
        self.line_near_error = float(msg.data)

    def line_far_error_callback(self, msg):
        self.line_far_error = float(msg.data)

    def line_near_detected_callback(self, msg):
        self.line_near_detected = bool(msg.data)

    def line_far_detected_callback(self, msg):
        self.line_far_detected = bool(msg.data)

    # =========================================================
    # CALLBACKS - ARUCO MARKERS
    # =========================================================

    def aruco_visible_callback(self, msg):
        """Called when ArUco marker visibility changes."""
        self.aruco_visible = bool(msg.data)

        if self.aruco_visible:
            self.last_aruco_visible_time = time.monotonic()

    def aruco_id_callback(self, msg):
        """Called with current ArUco marker ID."""
        self.current_aruco_id = int(msg.data)

    # =========================================================
    # CALLBACK - ODOMETRY
    # =========================================================

    def odom_callback(self, msg):
        """Track current position/heading for the corner maneuver."""
        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y
        self.odom_yaw = quaternion_to_yaw(msg.pose.pose.orientation)
        self.odom_received = True

    def aruco_close_callback(self, msg):
        """Called when ArUco marker gets close enough (0.2m)."""
        marker_id = int(msg.data)

        # Ignore invalid IDs
        if marker_id < 0:
            return

        # Already moving to a marker center or stopped
        if self.moving_to_marker_center or self.stop_until is not None:
            return

        # =====================================================
        # TRIGGER MOVE TO MARKER CENTER (ALL MARKERS)
        # Move forward 0.2m to reach marker center, then stop
        # =====================================================

        self.stopping_marker_id = marker_id
        self.moving_to_marker_center = True
        self.marker_center_start_time = time.monotonic()
        self.integral_error = 0.0

        self.get_logger().warn(
            f'MARKER {marker_id} CLOSE - MOVING TO CENTER '
            f'(+{self.marker_center_distance:.3f}m)'
        )

    # =========================================================
    # MARKER CENTER APPROACH
    # =========================================================

    def handle_marker_center_approach(self, error, dt):
        """Move forward to reach marker center (0.2m forward)."""
        
        # Time needed to move 0.2m at marker_center_speed
        # distance = speed * time, so time = distance / speed
        center_approach_time = self.marker_center_distance / self.marker_center_speed
        
        now = time.monotonic()
        elapsed = now - self.marker_center_start_time

        if elapsed < center_approach_time:
            # Still moving to center
            self.log_mode(f'MARKER {self.stopping_marker_id}: MOVING TO CENTER')
            
            # Move forward slowly with gentle steering
            linear = self.marker_center_speed
            angular = self.calculate_pi(
                error,
                dt,
                kp_override=self.kp * 0.3,  # Very gentle steering
                max_angular_override=0.10   # Limited turn
            )
            
            self.publish_velocity(linear, angular)
            return

        # Reached marker center - now stop for inspection
        self.log_mode(f'MARKER {self.stopping_marker_id}: AT CENTER - STOPPING')
        
        self.moving_to_marker_center = False
        self.marker_center_start_time = None
        
        # Set stop period for inspection
        self.stop_until = now + self.stop_duration
        self.integral_error = 0.0
        
        self.get_logger().info(
            f'MARKER {self.stopping_marker_id} CENTER REACHED - INSPECTING FOR {self.stop_duration} SECONDS'
        )

    # =========================================================
    # MOTOR CONTROL
    # =========================================================

    def publish_velocity(self, linear, angular):
        """Publish velocity command to robot, slew-limiting angular.z so
        every mode (following, marker approach, recovery, corner rotation)
        ramps into a new turn rate instead of snapping to it in one tick."""
        delta = angular - self.last_angular_cmd

        if delta > self.max_angular_step:
            angular = self.last_angular_cmd + self.max_angular_step
        elif delta < -self.max_angular_step:
            angular = self.last_angular_cmd - self.max_angular_step

        self.last_angular_cmd = angular

        msg = Twist()
        msg.linear.x = float(linear)
        msg.angular.z = float(angular)
        self.cmd_pub.publish(msg)

    def stop_robot(self):
        """Stop all motion immediately - bypasses the angular slew limit
        (used for safety cutoffs like watchdog/corner timeouts, where a
        multi-tick ramp-down to zero is not appropriate)."""
        self.last_angular_cmd = 0.0
        self.cmd_pub.publish(Twist())

    # =========================================================
    # MODE LOGGING
    # =========================================================

    def log_mode(self, mode):
        """Log mode change (only logs when mode actually changes)."""
        if mode == self.current_mode:
            return

        self.current_mode = mode
        self.get_logger().info(f'MODE -> {mode}')

    # =========================================================
    # CORNER DETECTION
    # =========================================================

    def detect_line_gone(self):
        """
        Detect if line has gone OUT OF FRAME (corner reached).
        
        Returns True when:
        - Line was being followed
        - Line has now disappeared from camera
        - Line has been gone for line_loss_timeout
        """

        # Line is still visible - no corner
        if self.line_detected:
            self.line_loss_start_time = None
            return False

        # Line just disappeared - start timeout
        now = time.monotonic()

        if self.line_loss_start_time is None:
            self.line_loss_start_time = now
            return False

        # Check if line has been invisible long enough
        elapsed = now - self.line_loss_start_time

        if elapsed > self.line_loss_timeout:
            return True  # LINE OUT OF FRAME - CORNER REACHED!

        return False

    def detect_corner_width_spike(self):
        """
        Detect a 90-degree corner via line width, for corners that never
        fully leave the ROI (the L-junction stays visible somewhere
        across the frame width even with a narrow, close-in ROI depth).

        Returns True when line_width has stayed above
        corner_width_threshold for corner_width_debounce seconds.
        """

        if self.line_width < self.corner_width_threshold:
            self.width_spike_start_time = None
            return False

        now = time.monotonic()

        if self.width_spike_start_time is None:
            self.width_spike_start_time = now
            return False

        elapsed = now - self.width_spike_start_time

        if elapsed > self.corner_width_debounce:
            return True  # LINE WIDTH SPIKED - CORNER REACHED!

        return False

    def detect_corner_bend(self):
        """
        Detect a 90-degree corner via near-vs-far ROI position, for
        corners that never fully leave the ROI and don't reliably
        spike raw width either (a steep diagonal single line inflates
        width the same way a corner does). A slant/curve shifts the
        near/far centroid difference gradually and by a bounded
        amount; a true right-angle corner jumps sharply.

        Returns True when |far_error - near_error| has stayed above
        corner_bend_threshold for corner_bend_debounce seconds.
        """

        bend = abs(self.line_far_error - self.line_near_error)

        if bend < self.corner_bend_threshold:
            self.bend_spike_start_time = None
            return False

        now = time.monotonic()

        if self.bend_spike_start_time is None:
            self.bend_spike_start_time = now
            return False

        elapsed = now - self.bend_spike_start_time

        if elapsed > self.corner_bend_debounce:
            return True  # LINE BENT SHARPLY - CORNER REACHED!

        return False

    # =========================================================
    # CORNER ROTATION
    # =========================================================

    def corner_rotation_out_of_frame(self, dt):
        """
        Execute blind maneuver when line goes out of frame (90-degree corner).

        Procedure, driven by /tb3_1/odom feedback rather than time:
        1. Move forward corner_forward_offset meters (measured via odom).
        2. Rotate until yaw has turned corner_rotation_target radians (90deg).
        """

        now = time.monotonic()

        if self.rotation_start_time is None:
            # Just started corner handling
            self.rotation_start_time = now
            self.corner_phase = 'forward'
            self.corner_start_x = self.odom_x
            self.corner_start_y = self.odom_y
            self.log_mode('CORNER: MOVING FORWARD TO EDGE')

        # Safety cap in case odom stalls/stops publishing.
        if now - self.rotation_start_time > (
            self.corner_forward_offset / self.corner_forward_speed
            + self.corner_rotation_max_duration
        ):
            self.log_mode('CORNER: TIMEOUT - ABORTING ROTATION')
            self.stop_robot()
            self.rotating_at_corner = False
            self.rotation_start_time = None
            self.corner_phase = None
            self.line_loss_start_time = None
            self.integral_error = 0.0
            self.corner_settle_until = time.monotonic() + self.corner_settle_duration
            return

        # =====================================================
        # PHASE 1: MOVE FORWARD OFFSET
        # =====================================================

        if self.corner_phase == 'forward':
            traveled = math.hypot(
                self.odom_x - self.corner_start_x,
                self.odom_y - self.corner_start_y
            )

            if traveled < self.corner_forward_offset:
                self.log_mode(f'CORNER: MOVING FORWARD ({self.corner_forward_offset}m offset)')
                self.publish_velocity(self.corner_forward_speed, 0.0)
                return

            # Forward offset complete - set up rotation phase.
            # Fixed direction: every corner on this closed loop turns the
            # same way, so use the known direction rather than inferring
            # it from last_turn_direction (which reflects momentary
            # steering noise, not the track's actual geometry).
            self.corner_phase = 'rotate'
            self.corner_start_yaw = self.odom_yaw
            self.corner_turn_direction = self.default_corner_turn_direction

            self.get_logger().info(
                f'CORNER: forward offset done - '
                f'traveled={traveled:.3f}m (target={self.corner_forward_offset:.3f}m), '
                f'pivot=({self.odom_x:.3f}, {self.odom_y:.3f})'
            )

        # =====================================================
        # PHASE 2: ROTATE UNTIL YAW HAS TURNED 90 DEGREES
        # =====================================================

        yaw_turned = abs(normalize_angle(self.odom_yaw - self.corner_start_yaw))

        if yaw_turned < self.corner_rotation_target:
            angular = self.corner_rotation_speed * self.corner_turn_direction
            self.log_mode('CORNER: ROTATING SMOOTHLY')
            self.publish_velocity(0.0, angular)
            return

        # =====================================================
        # CORNER COMPLETE
        # =====================================================
        self.log_mode('ROTATION COMPLETE - SETTLING ONTO LINE')
        self.stop_robot()

        self.get_logger().info(
            f'CORNER: rotation done - '
            f'turned={math.degrees(yaw_turned):.1f}deg '
            f'(target={math.degrees(self.corner_rotation_target):.1f}deg), '
            f'final=({self.odom_x:.3f}, {self.odom_y:.3f}, '
            f'yaw={math.degrees(self.odom_yaw):.1f}deg)'
        )

        # Reset rotation state and ease back onto the line gently for a
        # short window instead of handing off straight to full-speed PI.
        self.rotating_at_corner = False
        self.rotation_start_time = None
        self.corner_phase = None
        self.line_loss_start_time = None
        self.integral_error = 0.0
        self.corner_settle_until = time.monotonic() + self.corner_settle_duration

    # =========================================================
    # PI CONTROLLER
    # =========================================================

    def calculate_pi(
        self,
        error,
        dt,
        kp_override=None,
        max_angular_override=None
    ):
        """
        Calculate PI controller output for steering.
        
        Args:
            error: Line centering error (-1.0 to +1.0)
            dt: Delta time since last call (seconds)
            kp_override: Optional override for Kp gain
            max_angular_override: Optional override for max angular velocity
        
        Returns:
            angular: Angular velocity command (rad/s)
        """

        kp = kp_override if kp_override is not None else self.kp
        max_ang = (
            max_angular_override
            if max_angular_override is not None
            else self.max_angular
        )

        # Deadband: ignore very small errors
        if abs(error) < self.deadband:
            error = 0.0

        # Integrate error for I term
        self.integral_error += error * dt

        # Limit integral to prevent windup
        self.integral_error = max(
            -self.integral_limit,
            min(self.integral_limit, self.integral_error)
        )

        # Calculate PI output
        output = (
            kp * error
            +
            self.ki * self.integral_error
        )

        # Convert to angular velocity (negative for standard orientation)
        angular = -output

        # Limit maximum angular velocity
        angular = max(-max_ang, min(max_ang, angular))

        return angular

    def _interp_speed(self, abs_error, breakpoints):
        """Piecewise-linear speed from (abs_error, speed) breakpoints.

        Replaces hard if/elif speed tiers, which stepped linear speed by
        a fixed jump the instant abs_error crossed a threshold (e.g.
        0.10 -> 0.08 m/s at error=0.08) - visible as a speed pulse on
        the real robot even for a tiny, noise-sized change in error.
        This ramps smoothly between the same breakpoints instead.
        """
        for i in range(len(breakpoints) - 1):
            x0, y0 = breakpoints[i]
            x1, y1 = breakpoints[i + 1]
            if x0 <= abs_error <= x1:
                if x1 == x0:
                    return y1
                t = (abs_error - x0) / (x1 - x0)
                return y0 + t * (y1 - y0)
        return breakpoints[-1][1]

    # =========================================================
    # LINE FOLLOWING
    # =========================================================

    def follow_line(self, dt, marker_visible):
        """
        Main line following behavior. Only called while the line is
        currently detected - corner handling (line lost) is decided
        in control_loop() instead, since it needs to run while the
        line is NOT detected.
        """

        error = self.line_error
        abs_error = abs(error)

        # =====================================================
        # DIAGNOSTIC: LOG WIDTH TO FIND A REAL CORNER THRESHOLD
        # =====================================================

        now = time.monotonic()

        if now - self.last_width_diagnostic_log >= self.width_diagnostic_log_interval:
            self.last_width_diagnostic_log = now
            bend = abs(self.line_far_error - self.line_near_error)
            self.get_logger().info(
                f'WIDTH DIAG: mode={self.current_mode} '
                f'width={self.line_width:.0f}px error={error:+.3f} '
                f'near={self.line_near_error:+.3f} far={self.line_far_error:+.3f} '
                f'bend={bend:.3f}'
            )

        # =====================================================
        # MARKER APPROACH MODE
        # =====================================================

        if marker_visible:

            self.log_mode(
                f'ARUCO APPROACH ID {self.current_aruco_id}'
            )

            # Smooth approach with gentle steering
            linear = self.marker_approach_speed

            angular = self.calculate_pi(
                error,
                dt,
                kp_override=self.kp * 0.4,  # Even gentler (was 0.6)
                max_angular_override=0.15   # Limited steering (was 0.25)
            )

            self.publish_velocity(linear, angular)

            return

        # =====================================================
        # CORNER DETECTION COOLDOWN (just resumed from a marker stop)
        # =====================================================

        in_corner_cooldown = now < self.corner_detection_resume_until

        # =====================================================
        # CHECK FOR CORNER VIA LINE WIDTH SPIKE
        # (line is still "detected" but an L-junction has widened it)
        # =====================================================

        if (
            not in_corner_cooldown
            and self.corner_width_detection_enabled
            and self.detect_corner_width_spike()
        ):

            self.get_logger().info(
                f'CORNER: width spike triggered - width={self.line_width:.0f}px '
                f'(threshold={self.corner_width_threshold:.0f}px)'
            )

            self.rotating_at_corner = True
            self.width_spike_start_time = None
            self.corner_rotation_out_of_frame(dt)
            return

        # =====================================================
        # CHECK FOR CORNER VIA NEAR/FAR BEND
        # (line still "detected" but has bent sharply within the ROI)
        # =====================================================

        if (
            not in_corner_cooldown
            and self.corner_bend_detection_enabled
            and self.detect_corner_bend()
        ):

            bend = abs(self.line_far_error - self.line_near_error)
            self.get_logger().info(
                f'CORNER: bend triggered - bend={bend:.3f} '
                f'(threshold={self.corner_bend_threshold:.3f}) '
                f'near={self.line_near_error:+.3f} far={self.line_far_error:+.3f}'
            )

            self.rotating_at_corner = True
            self.bend_spike_start_time = None
            self.corner_rotation_out_of_frame(dt)
            return

        # =====================================================
        # CORNER/CHAMFER APPROACH - FAR BAND LOST, NEAR STILL TRACKING
        #
        # The far (look-ahead) band losing the line while the near
        # (closest-to-robot) band still holds it means the tape is about
        # to leave the ROI entirely - a sharp 90-degree corner or a
        # chamfered diagonal exit. self.line_error is a centroid over
        # whatever the whole ROI still sees, so it swings as the far
        # segment shrinks away; steering off the near band alone instead
        # avoids that swing (this is the main source of the reported
        # oscillation/jerk right at chamfered and sharp-90 corners), and
        # crawling here gives the vision system more frames to confirm
        # the tape has actually left the near band too before any
        # forward-offset/rotation decision is made in control_loop().
        # =====================================================

        if not self.line_far_detected and self.line_near_detected:

            self.log_mode('APPROACHING CORNER - NEAR BAND STEERING')

            angular = self.calculate_pi(
                self.line_near_error,
                dt,
                kp_override=self.kp * 0.5,
                max_angular_override=0.25
            )

            self.publish_velocity(self.corner_forward_speed, angular)

            return

        # =====================================================
        # SMOOTH LINE FOLLOWING
        # =====================================================

        # Use smooth PI controller with adaptive speed
        angular = self.calculate_pi(error, dt)

        # Continuous speed ramp based on error magnitude - see
        # _interp_speed() for why this replaced discrete speed tiers.
        linear = self._interp_speed(
            abs_error,
            [
                (0.00, self.normal_speed),
                (0.08, self.normal_speed),
                (0.15, self.medium_speed),
                (0.25, 0.06),
                (1.00, 0.04),
            ]
        )

        if abs_error < 0.08:
            self.log_mode('FOLLOWING')
        elif abs_error < 0.15:
            self.log_mode('CORRECTING SMALL')
        elif abs_error < 0.25:
            self.log_mode('CORRECTING')
        else:
            self.log_mode('CORRECTING LARGE')

        self.publish_velocity(linear, angular)

    # =========================================================
    # LINE RECOVERY
    # =========================================================

    def recover_line(self):
        """
        Attempt to find line after it's lost.
        Searches by turning in the last known direction.
        """

        elapsed = (
            time.monotonic()
            - self.last_line_seen_time
        )

        if elapsed <= self.recovery_time:

            self.log_mode('LINE RECOVERY')

            angular = (
                self.last_turn_direction
                * self.recovery_angular
            )

            self.publish_velocity(
                self.recovery_linear,
                angular
            )

        else:

            self.log_mode(
                'LINE LOST -> STOP'
            )

            self.stop_robot()

    # =========================================================
    # MAIN CONTROL LOOP (30 Hz)
    # =========================================================

    def control_loop(self):
        """30 Hz control loop - main decision making."""

        now = time.monotonic()

        dt = now - self.last_control_time

        self.last_control_time = now

        # Clamp dt to reasonable range
        dt = max(0.001, min(0.10, dt))

        # =====================================================
        # PRIORITY 1: MOVING TO MARKER CENTER
        # =====================================================

        if self.moving_to_marker_center:
            self.handle_marker_center_approach(self.line_error, dt)
            return

        # =====================================================
        # PRIORITY 2: ARUCO MARKER INSPECTION STOP
        # =====================================================

        if self.stop_until is not None:

            if now < self.stop_until:

                self.log_mode(
                    f'STOPPED AT ARUCO {self.stopping_marker_id}'
                )

                self.stop_robot()

                return

            # Stop period finished
            finished_id = self.stopping_marker_id

            self.stop_until = None
            self.stopping_marker_id = None
            self.integral_error = 0.0

            # Resuming after a stop can leave the robot briefly
            # misaligned with the line (large error), which spikes the
            # near/far bend reading enough to look like a corner even
            # though there isn't one. Suppress corner detection for a
            # short window while the PI controller re-centers normally.
            self.corner_detection_resume_until = (
                now + self.marker_resume_cooldown
            )

            self.get_logger().info(
                f'ARUCO {finished_id} inspection complete - RESUMING'
            )

        # =====================================================
        # CHECK IF MARKER IS CURRENTLY VISIBLE
        # =====================================================

        marker_visible_now = (

            self.aruco_visible

            and

            (
                now
                - self.last_aruco_visible_time
                < self.marker_timeout
            )
        )

        # =====================================================
        # DETECTOR WATCHDOG
        # =====================================================

        if (
            now - self.last_detection_time
            > self.detector_timeout
        ):

            self.log_mode(
                'LINE DETECTOR TIMEOUT'
            )

            self.stop_robot()

            return

        # =====================================================
        # DURING CORNER ROTATION
        # =====================================================

        if self.rotating_at_corner:

            self.corner_rotation_out_of_frame(dt)

            # Only resume when rotation is actually complete
            # (when corner_rotation_out_of_frame stops calling itself)
            # Don't interrupt rotation just because line reappeared

            return

        # =====================================================
        # SETTLING ONTO LINE AFTER A CORNER
        # =====================================================

        if self.corner_settle_until is not None:

            if now >= self.corner_settle_until:

                self.corner_settle_until = None

            elif self.line_detected:

                self.last_line_seen_time = now

                self.log_mode('CORNER: SETTLING ONTO LINE')

                angular = self.calculate_pi(
                    self.line_error,
                    dt,
                    kp_override=self.kp * 0.3,
                    max_angular_override=0.15
                )

                self.publish_velocity(self.corner_settle_speed, angular)

                return

            else:

                self.log_mode('CORNER: SETTLING - SEARCHING FOR LINE')

                self.publish_velocity(self.corner_settle_speed, 0.0)

                return

        # =====================================================
        # LINE VISIBLE -> FOLLOW IT
        # =====================================================

        if self.line_detected:

            self.last_line_seen_time = now

            self.follow_line(dt, marker_visible_now)

            return

        # =====================================================
        # LINE NOT DETECTED - CHECK IF THIS IS A CORNER
        # (line gone out of frame at a 90-degree corner)
        #
        # detect_line_gone() is always called (not short-circuited by the
        # guards below) so its line_loss_start_time debounce state stays
        # accurate every tick, regardless of cooldown/marker state - the
        # coasting branch further down depends on that being current.
        # =====================================================

        line_gone = self.detect_line_gone()

        if (
            not marker_visible_now
            and now >= self.corner_detection_resume_until
            and line_gone
        ):

            self.rotating_at_corner = True

            self.corner_rotation_out_of_frame(dt)

            return

        # =====================================================
        # MARKER VISIBLE BUT LINE LOST
        # =====================================================

        if marker_visible_now:

            self.log_mode(
                'MARKER VISIBLE - LINE LOST'
            )

            # Approach marker slowly and straight
            self.publish_velocity(0.035, 0.0)

            return

        # =====================================================
        # LINE MOMENTARILY LOST - COAST (don't snap to recovery yet)
        #
        # Any brief detection flicker (common on chamfered diagonals,
        # and right at a 90-degree corner) used to fall straight through
        # to recover_line()'s aggressive slow-crawl-and-sharp-turn search
        # the instant line_detected went False, even though it usually
        # reappears within line_loss_timeout with no real corner or loss
        # involved. That produced a visible stutter every time detection
        # blipped. While still inside that debounce window, hold a
        # steady straight coast instead - if it was just a blip, the
        # line reappears and follow_line() resumes normally next tick;
        # if it's a real corner, the branch above fires once the
        # timeout elapses; only a loss that outlasts the timeout (e.g.
        # suppressed by corner cooldown, or the robot genuinely drove
        # off the line) falls through to the real search below.
        # =====================================================

        if (
            self.line_loss_start_time is not None
            and (now - self.line_loss_start_time) <= self.line_loss_timeout
        ):

            self.log_mode('LINE MOMENTARILY LOST - COASTING')

            self.publish_velocity(self.corner_forward_speed, 0.0)

            return

        # =====================================================
        # LINE COMPLETELY LOST
        # =====================================================

        self.integral_error = 0.0

        self.recover_line()


# =============================================================
# MAIN ENTRY POINT
# =============================================================

def main(args=None):

    rclpy.init(args=args)

    node = LineFollowerCornerOut()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.get_logger().info('Stopping robot')

        for _ in range(10):

            node.stop_robot()

            time.sleep(0.02)

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()