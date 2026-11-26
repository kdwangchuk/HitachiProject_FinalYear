"""
Extended Kalman filter for a planar robot pose (x, y, theta).

Follows Bach, Khoi & Yi, "Application of QR Code for Localization and
Navigation of Indoor Mobile Robot", IEEE Access 2023:

  Prediction (Eq. 6-8)  encoder motion model
      x' = x + s * cos(theta + dtheta / 2)
      y' = y + s * sin(theta + dtheta / 2)
      theta' = theta + dtheta
      P' = A P A^T + Q,   A = [[1, 0, -s*sin(.)], [0, 1, s*cos(.)], [0, 0, 1]]

  Correction (Eq. 9-13) a marker gives a direct pose measurement, so H = I
      K = P (P + G)^-1
      x = x + K (z - x)
      P = (I - K) P

Differences from the paper, on purpose:
  * Q grows with distance travelled / angle turned / time, instead of a
    fixed constant, so uncertainty only builds up while the robot moves.
  * The measurement noise G grows with the marker's distance from the
    camera (a small marker is noisier the further away it is).
  * Innovation gating: a fix that is statistically implausible given the
    current uncertainty is rejected instead of being fused.
  * P is updated in Joseph form so it stays symmetric/positive-definite.

No ROS imports here so it can be unit-tested on its own.
"""

import math

import numpy as np


def normalize_angle(angle):
    """Wrap an angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


class PoseEKF:

    def __init__(
        self,
        q_xy_per_m=0.002,
        q_theta_per_rad=0.01,
        q_theta_per_m=0.001,
        q_per_sec=1.0e-5,
        r_xy=0.0025,
        r_theta=0.008,
        r_distance_ref=0.20,
        gate=16.27,
        max_rejects=20,
    ):
        self.q_xy_per_m = q_xy_per_m          # m^2 per metre travelled
        self.q_theta_per_rad = q_theta_per_rad  # rad^2 per radian turned
        self.q_theta_per_m = q_theta_per_m    # rad^2 per metre travelled
        self.q_per_sec = q_per_sec            # baseline growth per second
        self.r_xy = r_xy                      # m^2 at reference distance
        self.r_theta = r_theta                # rad^2 at reference distance
        self.r_distance_ref = r_distance_ref  # metres
        self.gate = gate                      # chi-square, 3 dof (0.1%)
        self.max_rejects = max_rejects

        self.x = np.zeros(3)
        self.P = np.diag([10.0, 10.0, 10.0])
        self.initialized = False  # True once a marker fix has been fused
        self.reject_count = 0

    # ---------------------------------------------------------

    def set_state(self, x, y, theta):
        """Set the pose without claiming it is a measurement (used to
        start the filter at the odometry pose before any marker fix)."""
        self.x = np.array([x, y, normalize_angle(theta)], dtype=float)

    def std_xy(self):
        return math.sqrt(max(0.0, self.P[0, 0] + self.P[1, 1]))

    def std_theta(self):
        return math.sqrt(max(0.0, self.P[2, 2]))

    # ---------------------------------------------------------

    def predict(self, s, dtheta, dt):
        """
        s      signed distance moved along the robot's heading (m)
        dtheta heading change (rad)
        dt     time since the previous prediction (s)
        """
        theta_mid = self.x[2] + dtheta / 2.0
        c = math.cos(theta_mid)
        sn = math.sin(theta_mid)

        self.x[0] += s * c
        self.x[1] += s * sn
        self.x[2] = normalize_angle(self.x[2] + dtheta)

        A = np.array([
            [1.0, 0.0, -s * sn],
            [0.0, 1.0, s * c],
            [0.0, 0.0, 1.0],
        ])

        q_xy = self.q_xy_per_m * abs(s) + self.q_per_sec * dt
        q_th = (
            self.q_theta_per_rad * abs(dtheta)
            + self.q_theta_per_m * abs(s)
            + self.q_per_sec * dt
        )
        Q = np.diag([q_xy, q_xy, q_th])

        self.P = A @ self.P @ A.T + Q

    # ---------------------------------------------------------

    def update(self, z, distance):
        """
        Fuse a marker-derived pose z = (x, y, theta) in the map frame.

        distance is the marker's distance from the camera (m), used to
        scale the measurement noise.

        Returns (accepted, d2, status) where status is one of
        'init', 'update', 'reinit', 'rejected'.
        """
        z = np.array(
            [z[0], z[1], normalize_angle(z[2])], dtype=float
        )

        scale = max(1.0, distance / self.r_distance_ref) ** 2
        R = np.diag([
            self.r_xy * scale,
            self.r_xy * scale,
            self.r_theta * scale,
        ])

        # First fix ever: nothing to compare against, adopt it.
        if not self.initialized:
            self.x = z
            self.P = R
            self.initialized = True
            self.reject_count = 0
            return True, 0.0, 'init'

        nu = z - self.x
        nu[2] = normalize_angle(nu[2])

        S = self.P + R
        S_inv = np.linalg.inv(S)
        d2 = float(nu @ S_inv @ nu)

        if d2 > self.gate:
            self.reject_count += 1

            # Rejected many times in a row: the filter itself is more
            # likely wrong than the measurements, so take the fix.
            if self.reject_count >= self.max_rejects:
                self.x = z
                self.P = R
                self.reject_count = 0
                return True, d2, 'reinit'

            return False, d2, 'rejected'

        self.reject_count = 0

        K = self.P @ S_inv
        self.x = self.x + K @ nu
        self.x[2] = normalize_angle(self.x[2])

        I_K = np.eye(3) - K
        self.P = I_K @ self.P @ I_K.T + K @ R @ K.T

        return True, d2, 'update'
