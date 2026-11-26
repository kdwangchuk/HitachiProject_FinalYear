"""
Geometry for ArUco markers lying FLAT ON THE FLOOR, seen by a robot-mounted
camera that may be tilted down.

Conventions
-----------
Camera (optical) frame, as produced by cv2.aruco / aruco_detector.py:
    X right, Y down, Z forward (out of the lens).
Robot frame:
    x forward, y left, z up.
Camera pitch is the angle the optical axis is tilted DOWN from horizontal
(positive = looking down). Camera roll is assumed to be zero.

Marker frame (OpenCV): origin at the marker centre, +x to the right along
the marker's top edge, +y up the marker, +z out of the marker's face
towards the viewer. A marker lying face-up on the floor therefore has its
+z axis pointing straight UP.

Marker heading (the "yaw" column of aruco_locations.csv) is the direction
of the marker's +x axis, seen from above, counter-clockwise from the map
+x axis. (In the aruco_detector debug image this is the RED axis.)
For a marker laid with its pattern upright as the robot approaches, the
marker's +y points along the direction of travel, so
    marker heading = heading of travel - 90 degrees.
"""

import math
from collections import namedtuple


def normalize_angle(angle):
    """Wrap an angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def quaternion_to_rotation_matrix(x, y, z, w):
    """3x3 rotation matrix (nested lists) from a unit quaternion.
    For aruco_detector.py's pose the columns are the marker's axes
    expressed in the camera frame."""
    return [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ]


def camera_to_robot(v, pitch):
    """Rotate a vector from the camera optical frame into the robot
    frame (x forward, y left, z up) for a camera tilted down by pitch."""
    vx, vy, vz = v
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    return (
        vz * cp - vy * sp,
        -vx,
        -vz * sp - vy * cp,
    )


def robot_to_camera(v, pitch):
    """Inverse of camera_to_robot (used by the tests)."""
    rx, ry, rz = v
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    return (
        -ry,
        -sp * rx - cp * rz,
        cp * rx - sp * rz,
    )


MarkerInBase = namedtuple(
    'MarkerInBase',
    [
        'forward',       # marker centre, metres ahead of the robot base
        'left',          # marker centre, metres left of the robot base
        'heading',       # marker +x axis direction relative to robot heading (rad)
        'normal_up',     # z of the marker normal in the robot frame (1.0 = flat on floor)
        'cam_height',    # implied camera height above the marker plane (m)
        'implied_pitch', # camera pitch implied by the marker normal (rad)
    ]
)


def marker_in_base_frame(
    position,
    quaternion,
    camera_forward_offset,
    camera_left_offset,
    camera_yaw_offset,
    camera_pitch,
):
    """
    position   (x, y, z) of the marker centre in the camera optical frame
    quaternion (x, y, z, w) of the marker orientation in the same frame

    Returns a MarkerInBase: where the marker is, and which way it points,
    relative to the robot's base_footprint on the ground plane.
    """
    qx, qy, qz, qw = quaternion
    R = quaternion_to_rotation_matrix(qx, qy, qz, qw)

    x_axis_cam = (R[0][0], R[1][0], R[2][0])
    normal_cam = (R[0][2], R[1][2], R[2][2])

    p = camera_to_robot(position, camera_pitch)
    x_axis = camera_to_robot(x_axis_cam, camera_pitch)
    normal = camera_to_robot(normal_cam, camera_pitch)

    cf = math.cos(camera_yaw_offset)
    sf = math.sin(camera_yaw_offset)

    forward = camera_forward_offset + cf * p[0] - sf * p[1]
    left = camera_left_offset + sf * p[0] + cf * p[1]

    heading = normalize_angle(
        math.atan2(x_axis[1], x_axis[0]) + camera_yaw_offset
    )

    # Camera pitch that would make the marker normal point straight up
    # (camera frame: up = (0, -cos(pitch), -sin(pitch))).
    implied_pitch = math.atan2(-normal_cam[2], -normal_cam[1])

    return MarkerInBase(
        forward=forward,
        left=left,
        heading=heading,
        normal_up=normal[2],
        cam_height=-p[2],
        implied_pitch=implied_pitch,
    )


def robot_pose_from_marker(marker_pose, forward, left, heading):
    """Robot (x, y, theta) in the map, given the marker's known map pose
    (x, y, yaw) and where the marker appears relative to the robot."""
    mx, my, myaw = marker_pose

    theta = normalize_angle(myaw - heading)
    c = math.cos(theta)
    s = math.sin(theta)

    return (
        mx - (c * forward - s * left),
        my - (s * forward + c * left),
        theta,
    )


def marker_pose_from_robot(robot_pose, forward, left, heading):
    """Marker (x, y, yaw) in the map, given the robot's map pose and
    where the marker appears relative to the robot (inverse of
    robot_pose_from_marker - used to survey unknown markers)."""
    rx, ry, rth = robot_pose

    c = math.cos(rth)
    s = math.sin(rth)

    return (
        rx + c * forward - s * left,
        ry + s * forward + c * left,
        normalize_angle(rth + heading),
    )
