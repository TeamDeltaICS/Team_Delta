import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np

class DepthMonitor(Node):

    def __init__(self):
        super().__init__('depth_monitor')

        # --- CONFIGURATION ---
        # Le topic standard de la profondeur sur RealSense
        # Parfois : /camera/aligned_depth_to_color/image_raw (pour aligner avec la couleur)
        # Ou : /camera/depth/image_rect_raw (profondeur brute)
        self.depth_topic = '/camera/depth/image_rect_raw' 
        
        # --- QoS (Indispensable pour RealSense) ---
        # La RealSense publie souvent en "Best Effort", il faut donc s'adapter
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image,
            self.depth_topic,
            self.depth_callback,
            qos_profile)

        self.get_logger().info("Depth Monitor démarré. En attente de données...")

    def depth_callback(self, msg):
        try:
            # 1. Conversion ROS -> OpenCV
            # IMPORTANT : L'encodage est 'passthrough' ou '16UC1' 
            # (Entier non signé 16 bits = valeur en millimètres)
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f"Erreur conversion: {e}")
            return

        # 2. Récupérer la distance au centre de l'image
        height, width = depth_image.shape
        center_x, center_y = width // 2, height // 2
        
        # Pour éviter le bruit d'un seul pixel mort, on fait la moyenne d'un petit carré de 10x10 au centre
        # On ignore les zéros (qui signifient "pas de mesure")
        roi = depth_image[center_y-5:center_y+5, center_x-5:center_x+5]
        
        # On filtre les valeurs > 0
        valid_pixels = roi[roi > 0]
        
        if len(valid_pixels) == 0:
            # Si le capteur est trop près ou trop loin, il renvoie 0 partout
            self.get_logger().warn("Zone aveugle ou hors de portée !")
            return

        # Moyenne en millimètres
        dist_mm = np.mean(valid_pixels)
        dist_m = dist_mm / 1000.0  # Conversion en mètres

        # 3. Logique de décision (Fuzzy Logic simplifiée)
        self.decision_logic(dist_m)

    def decision_logic(self, distance):
        # On formate le message pour l'affichage
        msg_prefix = f"[Distance: {distance:.2f}m]"

        if distance > 2.0:
            self.get_logger().info(f"{msg_prefix} LOIN -> On peut accélérer / Descendre vite")
        
        elif 1.0 < distance <= 2.0:
            self.get_logger().warn(f"{msg_prefix} MOYEN -> On commence à ralentir")
        
        elif 0.3 < distance <= 1.0:
            self.get_logger().error(f"{msg_prefix} PROCHE -> Atterrissage de précision / Danger !")
        
        else:
            self.get_logger().error(f"{msg_prefix} TRÈS PROCHE -> Arrêt moteur ou Touchdown !")

def main(args=None):
    rclpy.init(args=args)
    node = DepthMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()