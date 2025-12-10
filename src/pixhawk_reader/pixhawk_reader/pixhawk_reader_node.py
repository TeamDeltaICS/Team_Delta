#!/usr/bin/env python3
# Example MAVROS start:
# ros2 run mavros mavros_node --ros-args \
#   -p fcu_url:="serial:///dev/ttyACM1:57600"

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Imu, NavSatFix
from nav_msgs.msg import Odometry

# --- AJOUT : Import du service MAVROS ---
from mavros_msgs.srv import StreamRate


class PixhawkReader(Node):
    def __init__(self):
        super().__init__('pixhawk_reader')

        # --- Parameters (topics + log throttling) ---
        self.imu_topic = self.declare_parameter(
            'imu_topic', '/mavros/imu/data'
        ).get_parameter_value().string_value

        self.gps_topic = self.declare_parameter(
            'gps_topic', '/mavros/global_position/global'
        ).get_parameter_value().string_value

        self.odom_topic = self.declare_parameter(
            'odom_topic', '/mavros/local_position/odom'
        ).get_parameter_value().string_value

        self.imu_log_every = self.declare_parameter(
            'imu_log_every', 100
        ).get_parameter_value().integer_value

        self.gps_log_every = self.declare_parameter(
            'gps_log_every', 10
        ).get_parameter_value().integer_value

        self.odom_log_every = self.declare_parameter(
            'odom_log_every', 20
        ).get_parameter_value().integer_value

        # --- SUBSCRIBERS ---
        # IMU data
        self.imu_sub = self.create_subscription(
            Imu,
            self.imu_topic,
            self.imu_callback,
            qos_profile_sensor_data,
        )

        # GPS data
        self.gps_sub = self.create_subscription(
            NavSatFix,
            self.gps_topic,
            self.gps_callback,
            qos_profile_sensor_data,
        )

        # ODOMETRY (local position + orientation in world frame)
        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            qos_profile_sensor_data,
        )

        # Counters to throttle log output
        self.imu_count = 0
        self.gps_count = 0
        self.odom_count = 0

        self.get_logger().info(
            'PixhawkReader node started. Waiting for MAVROS data...\n'
            f'  IMU topic:   {self.imu_topic}\n'
            f'  GPS topic:   {self.gps_topic}\n'
            f'  Odom topic:  {self.odom_topic}'
        )

        # --- AJOUT : Lancement automatique du Stream Rate ---
        # On attend 5 secondes pour laisser le temps à MAVROS de démarrer
        self.init_timer = self.create_timer(5.0, self.configure_mavros_stream)


    # --- AUTOMATIC STREAM CONFIGURATION ---

    def configure_mavros_stream(self):
        """Demande au Pixhawk d'envoyer toutes les données (StreamRate)"""
        # On annule le timer pour ne lancer cette commande qu'une seule fois
        self.init_timer.cancel()

        # Création du client pour appeler le service
        client = self.create_client(StreamRate, '/mavros/set_stream_rate')

        # Vérification si le service est disponible (si MAVROS tourne)
        if not client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("⚠️ MAVROS service '/mavros/set_stream_rate' introuvable. "
                                   "Vérifiez que mavros_node tourne. Le flux n'a pas été activé.")
            return

        # Préparation de la requête
        request = StreamRate.Request()
        request.stream_id = 0          # 0 = MAV_DATA_STREAM_ALL (Tout envoyer)
        request.message_rate = 20      # Fréquence en Hz (20 Hz)
        request.on_off = True          # Activer

        # Appel asynchrone pour ne pas bloquer le node
        self.get_logger().info("📡 Envoi de la commande 'Set Stream Rate' au Pixhawk...")
        future = client.call_async(request)
        future.add_done_callback(self.stream_response_callback)

    def stream_response_callback(self, future):
        """Callback appelé quand le Pixhawk a répondu à la commande"""
        try:
            future.result() # Si pas d'exception, c'est que ça a marché
            self.get_logger().info("✅ Flux de données (IMU/GPS/Odom) activé avec succès à 20Hz !")
        except Exception as e:
            self.get_logger().error(f"❌ Échec de l'activation du flux : {e}")


    # --- CALLBACKS CAPTEURS ---

    def imu_callback(self, msg: Imu) -> None:
        self.imu_count += 1
        if self.imu_count % self.imu_log_every == 0:
            self.get_logger().info(
                f'IMU Accel [m/s^2]: '
                f'x={msg.linear_acceleration.x:.2f}, '
                f'y={msg.linear_acceleration.y:.2f}, '
                f'z={msg.linear_acceleration.z:.2f}'
            )

    def gps_callback(self, msg: NavSatFix) -> None:
        self.gps_count += 1
        if self.gps_count % self.gps_log_every != 0:
            return

        status = "No Fix"
        if msg.status.status >= 0:
            status = "Fix OK"

        self.get_logger().info(
            f'GPS [{status}]: '
            f'Lat={msg.latitude:.7f}, Lon={msg.longitude:.7f}, Alt={msg.altitude:.2f} m'
        )

    def odom_callback(self, msg: Odometry) -> None:
        """
        Odometry from PX4 EKF via MAVROS:
        - pose.pose.position: local position in world (ENU) frame
        - pose.pose.orientation: fused attitude (quaternion)
        """
        self.odom_count += 1
        if self.odom_count % self.odom_log_every != 0:
            return

        # Position
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        pz = msg.pose.pose.position.z

        # Orientation (quaternion -> yaw for readability)
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w

        yaw = self.quat_to_yaw(qx, qy, qz, qw)

        self.get_logger().info(
            f'ODOM: pos=({px:.2f}, {py:.2f}, {pz:.2f}) m, '
            f'yaw={math.degrees(yaw):.1f} deg'
        )

    @staticmethod
    def quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
        """Convert quaternion to yaw angle (rad) in ENU frame."""
        # Normalize defensively
        norm = math.sqrt(x * x + y * y + z * z + w * w)
        if norm > 1e-8:
            x /= norm
            y /= norm
            z /= norm
            w /= norm

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)


def main(args=None):
    rclpy.init(args=args)
    node = PixhawkReader()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()