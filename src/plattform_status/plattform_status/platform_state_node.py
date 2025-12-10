#!/usr/bin/env python3
import math
import collections

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float32MultiArray


class PlatformStateNode(Node):
    """
    Node that:
      - subscribes to platform pose from ArUco (/platform/pose_cam, in camera frame)
      - subscribes to vision availability mask (/platform/vision_available)
      - subscribes to IMU data (/mavros/imu/data by default)
      - subscribes to UAV odometry (/mavros/local_position/odom by default, in world/map frame)
      - transforms platform pose to world frame
      - keeps a history of world-frame position and velocity
      - computes a "wind" proxy from IMU
      - publishes a GRU input feature vector on /gru_input_vector
    """

    def __init__(self):
        super().__init__('platform_state_node')

        # --- Parameters ---
        self.history_len = self.declare_parameter(
            'history_len', 20
        ).get_parameter_value().integer_value
        self.dt_default = self.declare_parameter(
            'dt_default', 0.05
        ).get_parameter_value().double_value

        # Topics (defaults chosen for typical setup with MAVROS)
        self.pose_topic = self.declare_parameter(
            'pose_topic', '/platform/pose_cam'
        ).get_parameter_value().string_value
        self.mask_topic = self.declare_parameter(
            'mask_topic', '/platform/vision_available'
        ).get_parameter_value().string_value
        self.imu_topic = self.declare_parameter(
            'imu_topic', '/mavros/imu/data'
        ).get_parameter_value().string_value
        self.odom_topic = self.declare_parameter(
            'odom_topic', '/mavros/local_position/odom'
        ).get_parameter_value().string_value

        # Kept for compatibility; no longer used to compensate velocities
        self.use_uav_comp = self.declare_parameter(
            'use_uav_compensation', False
        ).get_parameter_value().bool_value

        self.max_vel = self.declare_parameter(
            'max_rel_velocity', 10.0
        ).get_parameter_value().double_value
        self.w_alpha = self.declare_parameter(
            'w_lowpass_alpha', 0.1
        ).get_parameter_value().double_value

        # Camera extrinsics: camera frame relative to UAV frame
        self.cam_tx = self.declare_parameter('cam_tx', 0.0).get_parameter_value().double_value
        self.cam_ty = self.declare_parameter('cam_ty', 0.0).get_parameter_value().double_value
        self.cam_tz = self.declare_parameter('cam_tz', 0.0).get_parameter_value().double_value
        self.cam_roll = self.declare_parameter('cam_roll', 0.0).get_parameter_value().double_value
        self.cam_pitch = self.declare_parameter('cam_pitch', 0.0).get_parameter_value().double_value
        self.cam_yaw = self.declare_parameter('cam_yaw', 0.0).get_parameter_value().double_value

        self.t_uav_cam = np.array([self.cam_tx, self.cam_ty, self.cam_tz], dtype=float)
        self.R_uav_cam = self.rpy_to_rot(self.cam_roll, self.cam_pitch, self.cam_yaw)

        # --- History buffers (world-frame) ---
        self.x_hist = collections.deque(maxlen=self.history_len)
        self.y_hist = collections.deque(maxlen=self.history_len)
        self.vx_hist = collections.deque(maxlen=self.history_len)
        self.vy_hist = collections.deque(maxlen=self.history_len)

        # Last platform position in world frame and time
        self.last_pos_world = None  # np.array([x, y])
        self.last_pose_time = None  # float seconds

        # Current UAV pose in world frame
        self.have_odom = False
        self.uav_pos_world = np.zeros(3, dtype=float)    # [x, y, z]
        self.R_world_uav = np.eye(3, dtype=float)        # rotation matrix

        # Scalar features
        self.w_t = 0.0
        self.m_t = 0.0

        # Optional: UAV velocity (not used in world-frame platform velocity)
        self.vx_uav = 0.0
        self.vy_uav = 0.0

        # --- Subscriptions ---
        self.create_subscription(
            PoseStamped,
            self.pose_topic,
            self.pose_cb,
            qos_profile_sensor_data
        )
        self.create_subscription(
            Bool,
            self.mask_topic,
            self.mask_cb,
            10  # Bool is fine with default QoS
        )
        self.create_subscription(
            Imu,
            self.imu_topic,
            self.imu_cb,
            qos_profile_sensor_data
        )
        self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_cb,
            qos_profile_sensor_data
        )

        # --- Publisher ---
        self.gru_pub = self.create_publisher(
            Float32MultiArray,
            '/gru_input_vector',
            10
        )

        # --- Timer ---
        self.timer = self.create_timer(self.dt_default, self.timer_cb)

        self._buffer_ready_logged = False
        self._warned_no_odom = False

        self.get_logger().info(
            f"PlatformStateNode (world frame) started with history_len={self.history_len}, "
            f"dt={self.dt_default}\n"
            f"  pose_topic: {self.pose_topic}\n"
            f"  mask_topic: {self.mask_topic}\n"
            f"  imu_topic: {self.imu_topic}\n"
            f"  odom_topic: {self.odom_topic}"
        )

    # --------- Helper functions for rotations ---------

    @staticmethod
    def rpy_to_rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
        cr = math.cos(roll)
        sr = math.sin(roll)
        cp = math.cos(pitch)
        sp = math.sin(pitch)
        cy = math.cos(yaw)
        sy = math.sin(yaw)

        # R = Rz(yaw) * Ry(pitch) * Rx(roll)
        Rz = np.array([[cy, -sy, 0.0],
                       [sy,  cy, 0.0],
                       [0.0, 0.0, 1.0]])
        Ry = np.array([[cp, 0.0, sp],
                       [0.0, 1.0, 0.0],
                       [-sp, 0.0, cp]])
        Rx = np.array([[1.0, 0.0, 0.0],
                       [0.0, cr, -sr],
                       [0.0, sr,  cr]])
        return Rz @ Ry @ Rx

    @staticmethod
    def quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
        # Normalize quaternion
        norm = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
        if norm < 1e-8 or not np.isfinite(norm):
            return np.eye(3, dtype=float)
        qx /= norm
        qy /= norm
        qz /= norm
        qw /= norm

        xx = qx * qx
        yy = qy * qy
        zz = qz * qz
        xy = qx * qy
        xz = qx * qz
        yz = qy * qz
        wx = qw * qx
        wy = qw * qy
        wz = qw * qz

        R = np.array(
            [
                [1.0 - 2.0 * (yy + zz),     2.0 * (xy - wz),         2.0 * (xz + wy)],
                [2.0 * (xy + wz),           1.0 - 2.0 * (xx + zz),   2.0 * (yz - wx)],
                [2.0 * (xz - wy),           2.0 * (yz + wx),         1.0 - 2.0 * (xx + yy)],
            ],
            dtype=float,
        )
        return R

    # --------- Callbacks ---------

    def odom_cb(self, msg: Odometry) -> None:
        """Store UAV pose in world frame (position + orientation) and its velocity."""
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        pz = msg.pose.pose.position.z

        ox = msg.pose.pose.orientation.x
        oy = msg.pose.pose.orientation.y
        oz = msg.pose.pose.orientation.z
        ow = msg.pose.pose.orientation.w

        if np.isfinite(px) and np.isfinite(py) and np.isfinite(pz):
            self.uav_pos_world = np.array([px, py, pz], dtype=float)

        self.R_world_uav = self.quat_to_rot(ox, oy, oz, ow)

        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.vx_uav = vx if np.isfinite(vx) else 0.0
        self.vy_uav = vy if np.isfinite(vy) else 0.0

        self.have_odom = True

    def pose_cb(self, msg: PoseStamped) -> None:
        """
        Platform pose from ArUco in camera frame -> transform into world frame.
        Then compute world-frame velocities and update histories.
        """
        if not self.have_odom:
            if not self._warned_no_odom:
                self.get_logger().warn(
                    "No odom received yet; cannot transform platform pose to world frame."
                )
                self._warned_no_odom = True
            return

        xc = msg.pose.position.x
        yc = msg.pose.position.y
        zc = msg.pose.position.z

        if not (np.isfinite(xc) and np.isfinite(yc) and np.isfinite(zc)):
            return

        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # camera -> UAV -> world
        p_cam = np.array([xc, yc, zc], dtype=float)
        p_uav = self.R_uav_cam @ p_cam + self.t_uav_cam
        p_world = self.R_world_uav @ p_uav + self.uav_pos_world

        xw = float(p_world[0])
        yw = float(p_world[1])

        # finite-difference velocities in world frame
        if self.last_pos_world is not None and self.last_pose_time is not None:
            dt = t - self.last_pose_time
            if dt > 1e-4:
                vx_meas = (xw - self.last_pos_world[0]) / dt
                vy_meas = (yw - self.last_pos_world[1]) / dt
            else:
                vx_meas = 0.0
                vy_meas = 0.0
        else:
            vx_meas = 0.0
            vy_meas = 0.0

        vx_meas = float(np.clip(vx_meas, -self.max_vel, self.max_vel))
        vy_meas = float(np.clip(vy_meas, -self.max_vel, self.max_vel))

        if not np.isfinite(vx_meas) or not np.isfinite(vy_meas):
            vx_meas = 0.0
            vy_meas = 0.0

        self.last_pos_world = np.array([xw, yw], dtype=float)
        self.last_pose_time = t

        vx_rel = vx_meas
        vy_rel = vy_meas

        self.x_hist.append(xw)
        self.y_hist.append(yw)
        self.vx_hist.append(vx_rel)
        self.vy_hist.append(vy_rel)

    def mask_cb(self, msg: Bool) -> None:
        self.m_t = 1.0 if msg.data else 0.0

    def imu_cb(self, msg: Imu) -> None:
        ax = msg.linear_acceleration.x
        ay = msg.linear_acceleration.y
        az = msg.linear_acceleration.z

        if not (np.isfinite(ax) and np.isfinite(ay) and np.isfinite(az)):
            return

        acc_norm = float(np.sqrt(ax * ax + ay * ay + az * az))
        g = 9.81
        w_raw = abs(acc_norm - g)
        self.w_t = (1.0 - self.w_alpha) * self.w_t + self.w_alpha * w_raw

    def timer_cb(self) -> None:
        """Periodically build GRU input vector from world-frame history and publish it."""
        if len(self.x_hist) < self.history_len:
            return

        if not self._buffer_ready_logged:
            self.get_logger().info("GRU input buffer filled; starting to publish.")
            self._buffer_ready_logged = True

        x_arr = np.asarray(self.x_hist, dtype=np.float32)
        y_arr = np.asarray(self.y_hist, dtype=np.float32)
        vx_arr = np.asarray(self.vx_hist, dtype=np.float32)
        vy_arr = np.asarray(self.vy_hist, dtype=np.float32)

        feat = np.concatenate(
            [
                x_arr,
                y_arr,
                vx_arr,
                vy_arr,
                np.array([self.w_t, self.m_t], dtype=np.float32),
            ]
        )

        msg = Float32MultiArray()
        msg.data = feat.tolist()
        self.gru_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PlatformStateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
