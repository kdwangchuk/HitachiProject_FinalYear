#!/usr/bin/env python3

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, Float32, Int32


class LineDetector(Node):
    """
    Improved yellow line detector - tracks line to frame edges.
    
    Detects yellow line and publishes:
    - Line detected (Bool)
    - Line error (Float32) - position relative to center
    - Line width (Float32)
    
    KEY: Continues detecting line even at frame edges/corners
    """

    def __init__(self):
        super().__init__('line_detector')

        # =====================================================
        # SETTINGS - COLOR DETECTION (HSV)
        # =====================================================
        
        # Yellow line HSV range - EXPANDED for edge cases
        self.lower_yellow = np.array([10, 80, 80])   # Lower threshold
        self.upper_yellow = np.array([40, 255, 255])  # Upper threshold

        # =====================================================
        # SETTINGS - REGION OF INTEREST
        # =====================================================
        
        # Narrow band close to the robot (not far look-ahead).
        # A wide ROI blends the current line segment with whatever
        # comes next, which both adds lag to steering and pulls the
        # centroid toward the next segment before a corner is reached.
        # A close, narrow ROI fixes centering/smoothness in general,
        # and also makes line-loss a reliable 90-degree-corner signal
        # (curves/slants keep some line inside this band at all times;
        # only a hard right-angle discontinuity empties it).
        # Tune these on hardware.
        self.roi_top = 300
        self.roi_height = 180

        # Y centroid offset (moves detection point down)
        self.y_offset = 0

        # =====================================================
        # SETTINGS - LINE DETECTION
        # =====================================================
        
        # Minimum area to be considered a line
        self.min_area = 50  # REDUCED (detect smaller line segments at edges)
        
        # Morphological operations
        self.kernel_size = 3  # SMALLER (preserve thin lines at edges)
        self.erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        self.dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # =====================================================
        # STATE
        # =====================================================
        
        self.image_width = 640
        self.image_height = 480
        self.last_valid_error = 0.0
        self.last_valid_near_error = 0.0
        self.last_valid_far_error = 0.0

        # Near/far sub-bands within the ROI, used to tell a 90-degree
        # corner (line position jumps sharply between near and far)
        # apart from a slant/curve (position shifts gradually) - raw
        # line width can't tell these apart since a steep diagonal
        # single line also produces a wide bounding box.
        self.near_band_height = 60  # bottom rows of ROI (closest to robot)
        self.far_band_height = 60  # top rows of ROI (farthest lookahead)

        # =====================================================
        # SUBSCRIBERS
        # =====================================================
        
        self.create_subscription(
            CompressedImage,
            '/webcam/image/compressed',
            self.image_callback,
            10
        )

        # =====================================================
        # PUBLISHERS
        # =====================================================
        
        self.line_detected_pub = self.create_publisher(
            Bool,
            '/hardware/line_detected',
            10
        )

        self.line_error_pub = self.create_publisher(
            Float32,
            '/hardware/line_error',
            10
        )

        self.line_width_pub = self.create_publisher(
            Float32,
            '/hardware/line_width',
            10
        )

        self.line_y_position_pub = self.create_publisher(
            Float32,
            '/hardware/line_y_position',  # NEW: Publish Y position for encoder-based navigation
            10
        )

        self.line_near_error_pub = self.create_publisher(
            Float32,
            '/hardware/line_near_error',
            10
        )

        self.line_far_error_pub = self.create_publisher(
            Float32,
            '/hardware/line_far_error',
            10
        )

        # Raw per-frame presence for each band (True only when THIS frame
        # has yellow pixels in that band). line_near_error/line_far_error
        # fall back to the last valid reading when a band is empty, which
        # hides the exact frame a band went empty - these publish the
        # unfiltered fact instead, so a consumer can tell "still tracking,
        # just noisy" apart from "this band is genuinely empty now".
        self.line_near_detected_pub = self.create_publisher(
            Bool,
            '/hardware/line_near_detected',
            10
        )

        self.line_far_detected_pub = self.create_publisher(
            Bool,
            '/hardware/line_far_detected',
            10
        )

        self.debug_pub = self.create_publisher(
            CompressedImage,
            '/hardware/yellow_line_debug/compressed',
            10
        )

        self.get_logger().info('========================================')
        self.get_logger().info('Improved Line Detector - Edge Tracking')
        self.get_logger().info('========================================')
        self.get_logger().info(f'HSV Range: [{self.lower_yellow} to {self.upper_yellow}]')
        self.get_logger().info(f'ROI: Y {self.roi_top} to {self.roi_top + self.roi_height}')
        self.get_logger().info(f'Y Offset: {self.y_offset} pixels')
        self.get_logger().info('Min Area: 50 (detects at edges)')
        self.get_logger().info('========================================')

    def image_callback(self, msg):
        """Process compressed image and detect yellow line"""

        self._process_image(msg)

    def _band_error(self, band_mask, image_center_x):
        """Normalized x-centroid error (-1..1) of a mask sub-band, or
        None if the band has no yellow pixels."""
        M = cv2.moments(band_mask, binaryImage=True)
        if M['m00'] <= 0:
            return None
        cx = M['m10'] / M['m00']
        error = (cx - image_center_x) / image_center_x
        return max(-1.0, min(1.0, error))

    def _process_image(self, msg):
        # Decompress image
        np_arr = np.frombuffer(msg.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if image is None:
            return

        self.image_height, self.image_width = image.shape[:2]

        # =====================================================
        # CONVERT TO HSV
        # =====================================================
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # =====================================================
        # APPLY ROI (FULL LOWER HALF - catches corners!)
        # =====================================================
        
        roi = hsv[self.roi_top:self.roi_top + self.roi_height, :]

        # =====================================================
        # DETECT YELLOW LINE (with expansion for edges)
        # =====================================================
        
        mask = cv2.inRange(roi, self.lower_yellow, self.upper_yellow)

        # Gentle morphological operations to preserve edges
        mask = cv2.erode(mask, self.erosion_kernel, iterations=1)
        mask = cv2.dilate(mask, self.dilation_kernel, iterations=1)

        # =====================================================
        # NEAR/FAR BAND ERROR (corner-vs-slant discrimination)
        # =====================================================

        image_center_x = self.image_width / 2.0

        near_mask = mask[self.roi_height - self.near_band_height:self.roi_height, :]
        far_mask = mask[0:self.far_band_height, :]

        near_error_raw = self._band_error(near_mask, image_center_x)
        near_detected = near_error_raw is not None
        if near_error_raw is None:
            near_error = self.last_valid_near_error
        else:
            near_error = near_error_raw
            self.last_valid_near_error = near_error

        far_error_raw = self._band_error(far_mask, image_center_x)
        far_detected = far_error_raw is not None
        if far_error_raw is None:
            far_error = self.last_valid_far_error
        else:
            far_error = far_error_raw
            self.last_valid_far_error = far_error

        self.line_near_error_pub.publish(Float32(data=near_error))
        self.line_far_error_pub.publish(Float32(data=far_error))
        self.line_near_detected_pub.publish(Bool(data=near_detected))
        self.line_far_detected_pub.publish(Bool(data=far_detected))

        # =====================================================
        # FIND LINE CONTOURS
        # =====================================================

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            # No line detected
            self.line_detected_pub.publish(Bool(data=False))
            self.line_error_pub.publish(Float32(data=self.last_valid_error))
            self.line_width_pub.publish(Float32(data=0.0))
            return

        # Find largest contour (the line)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        if area < self.min_area:
            # Too small to be a line
            self.line_detected_pub.publish(Bool(data=False))
            self.line_error_pub.publish(Float32(data=self.last_valid_error))
            self.line_width_pub.publish(Float32(data=0.0))
            return

        # =====================================================
        # CALCULATE LINE PROPERTIES
        # =====================================================
        
        # Get moments for centroid
        M = cv2.moments(largest_contour)
        
        if M['m00'] > 0:
            # Centroid in ROI coordinates
            cx_roi = M['m10'] / M['m00']
            cy_roi = M['m01'] / M['m00']
            
            # Convert to full image coordinates
            cx = cx_roi
            cy = self.roi_top + cy_roi + self.y_offset
        else:
            self.line_detected_pub.publish(Bool(data=False))
            return

        # =====================================================
        # CALCULATE LINE ERROR (position relative to center)
        # =====================================================
        
        # Center of image horizontally
        image_center_x = self.image_width / 2.0
        
        # Error: how far from center (-1 to +1)
        line_error = (cx - image_center_x) / (image_center_x)
        line_error = max(-1.0, min(1.0, line_error))  # Clamp to [-1, 1]
        
        # Store as last valid
        self.last_valid_error = line_error

        # =====================================================
        # CALCULATE LINE WIDTH
        # =====================================================
        
        x, y, w, h = cv2.boundingRect(largest_contour)
        line_width = float(w)

        # =====================================================
        # PUBLISH RESULTS
        # =====================================================
        
        self.line_detected_pub.publish(Bool(data=True))
        self.line_error_pub.publish(Float32(data=line_error))
        self.line_width_pub.publish(Float32(data=line_width))
        self.line_y_position_pub.publish(Float32(data=cy))  # NEW: Y position for encoder navigation

        # =====================================================
        # DEBUG IMAGE
        # =====================================================
        
        debug_image = image.copy()
        
        # Draw full ROI rectangle (shows detection area)
        cv2.rectangle(debug_image, (0, self.roi_top), 
                     (self.image_width, self.roi_top + self.roi_height),
                     (0, 255, 0), 2)
        
        # Draw detected line contour (thick for visibility)
        cv2.drawContours(debug_image, [largest_contour], 0, (0, 255, 255), 3)
        
        # Draw centroid (larger circle)
        cv2.circle(debug_image, (int(cx), int(cy)), 8, (0, 0, 255), -1)
        
        # Draw center line (reference)
        cv2.line(debug_image, (int(self.image_width/2), 0), 
                (int(self.image_width/2), self.image_height),
                (255, 0, 0), 2)
        
        # Draw bounding box around detected line
        cv2.rectangle(debug_image, (x, self.roi_top + y), 
                     (x + w, self.roi_top + y + h),
                     (255, 255, 0), 2)
        
        # Add text info (visible and clear)
        text = f"Error: {line_error:+.3f} | Width: {line_width:.0f} | Y: {int(cy)}"
        cv2.putText(debug_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8, (255, 255, 255), 2)
        
        text_config = f"Y Offset: {self.y_offset} | Area: {area:.0f}"
        cv2.putText(debug_image, text_config, (10, 65), cv2.FONT_HERSHEY_SIMPLEX,
                   0.8, (255, 255, 0), 2)
        
        text_roi = f"ROI: {self.roi_top} to {self.roi_top + self.roi_height}"
        cv2.putText(debug_image, text_roi, (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                   0.8, (0, 255, 255), 2)

        # Publish debug image
        _, buffer = cv2.imencode('.jpg', debug_image)
        debug_msg = CompressedImage()
        debug_msg.format = "jpeg"
        debug_msg.data = buffer.tobytes()
        self.debug_pub.publish(debug_msg)


def main(args=None):
    rclpy.init(args=args)
    node = LineDetector()

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