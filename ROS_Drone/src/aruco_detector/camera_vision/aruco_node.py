#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import cv2
import numpy as np


def rotation_matrix_to_quaternion(R: np.ndarray):
    """Robust conversion of 3x3 rotation matrix to quaternion (x, y, z, w)."""
    trace = np.trace(R)
    if trace > 0.0:
        S = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    else:
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S
    return float(qx), float(qy), float(qz), float(qw)


class ArucoDetector(Node):

    def __init__(self):
        super().__init__('aruco_detector')

        # --- CONFIGURATION ---
        self.target_id = 7
        self.marker_size = 0.10  # meters
        
        # RealSense Topics
        self.camera_topic = '/camera/color/image_raw'
        self.info_topic = '/camera/color/camera_info'

        self.get_logger().info(f"Starting ArUco detection with OpenCV {cv2.__version__}")

        # ArUco Setup
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        # Prepare 3D points for solvePnP (Counter-clockwise: TL, TR, BR, BL)
        # Z is 0 because the marker is flat
        ms = self.marker_size / 2.0
        self.marker_points = np.array([
            [-ms,  ms, 0],
            [ ms,  ms, 0],
            [ ms, -ms, 0],
            [-ms, -ms, 0]
        ], dtype=np.float32)

        self.bridge = CvBridge()
        self.frame_count = 0

        # --- SUBSCRIBERS ---
        self.subscription = self.create_subscription(
            Image,
            self.camera_topic,
            self.image_callback,
            10) # Reliable

        self.info_sub = self.create_subscription(
            CameraInfo,
            self.info_topic, 
            self.info_callback,
            10) # Reliable

        # --- PUBLISHERS ---
        self.debug_pub = self.create_publisher(Image, '/aruco/debug_image', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/platform/pose_cam', 10)
        self.mask_pub = self.create_publisher(Bool, '/platform/vision_available', 10)

        self.camera_matrix = None
        self.dist_coeffs = None
        self.last_pose_msg = None
        self.warned_no_intrinsics = False

    def info_callback(self, msg: CameraInfo) -> None:
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k, dtype=float).reshape((3, 3))
            self.dist_coeffs = np.array(msg.d, dtype=float)
            self.get_logger().info(f"Camera intrinsics received from {self.info_topic}")

    def image_callback(self, msg: Image) -> None:
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CV Bridge Error: {e}")
            return

        self.frame_count += 1
        if self.frame_count == 1:
            self.get_logger().info(">>> SUCCESS: First image received! <<<")
        elif self.frame_count % 30 == 0:
            self.get_logger().info(f"Video stream active - Frame #{self.frame_count}")

        corners, ids, rejected = self.detector.detectMarkers(cv_image)
        vision_available = False

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)
            ids = ids.flatten()

            if self.target_id in ids:
                idx = int(np.where(ids == self.target_id)[0][0])
                vision_available = True

                if self.camera_matrix is not None and self.dist_coeffs is not None:
                    # --- NEW: Replace estimatePoseSingleMarkers with solvePnP ---
                    success, rvec, tvec = cv2.solvePnP(
                        self.marker_points, 
                        corners[idx], 
                        self.camera_matrix, 
                        self.dist_coeffs,
                        flags=cv2.SOLVEPNP_IPPE_SQUARE
                    )
                    
                    if success:
                        # Draw Axis
                        try:
                            cv2.drawFrameAxes(cv_image, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.05)
                        except AttributeError:
                            # Fallback if drawFrameAxes is missing
                            pass 

                        x, y, z = tvec.flatten()
                        self.get_logger().info(f"TARGET FOUND: x={x:.2f}m, y={y:.2f}m, z={z:.2f}m")

                        # Publish Pose
                        pose_msg = PoseStamped()
                        pose_msg.header = msg.header
                        pose_msg.pose.position.x = float(x)
                        pose_msg.pose.position.y = float(y)
                        pose_msg.pose.position.z = float(z)

                        # Convert rotation vector to quaternion
                        R, _ = cv2.Rodrigues(rvec)
                        qx, qy, qz, qw = rotation_matrix_to_quaternion(R)
                        pose_msg.pose.orientation.x = qx
                        pose_msg.pose.orientation.y = qy
                        pose_msg.pose.orientation.z = qz
                        pose_msg.pose.orientation.w = qw

                        self.pose_pub.publish(pose_msg)
                else:
                    if not self.warned_no_intrinsics:
                        self.get_logger().warn(f"ID {self.target_id} detected but waiting for camera intrinsics...")
                        self.warned_no_intrinsics = True

        # Publish Mask and Debug Image
        mask_msg = Bool()
        mask_msg.data = vision_available
        self.mask_pub.publish(mask_msg)
        self.debug_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8'))


def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
