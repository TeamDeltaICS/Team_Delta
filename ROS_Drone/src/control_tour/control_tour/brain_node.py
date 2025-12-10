"""#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import math
import onnxruntime as ort # Moteur d'inférence pour le GRU

from geometry_msgs.msg import PoseStamped, Vector3, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from mavros_msgs.msg import State
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# ==========================================
# 1. MOTEUR DE LOGIQUE FLOUE (FUZZY LOGIC)
# ==========================================
class FuzzyMamdaniEngine:
    def __init__(self):
        # Scores de sortie (Actions)
        self.OUT_ABORT = 1.5
        self.OUT_HOLD = 4.0
        self.OUT_APPROACH = 7.0
        self.OUT_DESCEND = 9.5

    def triangle(self, x, a, b, c):
        return max(0.0, min((x - a)/(b - a + 1e-6), (c - x)/(c - b + 1e-6)))

    def trapezoid(self, x, a, b, c, d):
        return max(0.0, min(min((x - a)/(b - a + 1e-6), 1.0), (d - x)/(d - c + 1e-6)))

    def compute(self, pvis, d_norm):
        # Fuzzification
        mu_p_low  = self.trapezoid(pvis, -0.1, 0.0, 0.3, 0.5)
        mu_p_med  = self.triangle(pvis, 0.3, 0.5, 0.8)
        mu_p_high = self.trapezoid(pvis, 0.6, 0.8, 1.0, 1.1)
        
        mu_d_low  = self.trapezoid(d_norm, -0.1, 0.0, 0.2, 0.4)
        mu_d_med  = self.triangle(d_norm, 0.2, 0.5, 0.8)
        mu_d_high = self.trapezoid(d_norm, 0.6, 0.8, 1.0, 1.1)

        # Inférence (Règles)
        r_abort    = max(min(mu_p_low, mu_d_low), min(mu_p_med, mu_d_low), min(mu_p_high, mu_d_low))
        r_hold     = max(min(mu_p_low, mu_d_med), min(mu_p_low, mu_d_high))
        r_approach = max(min(mu_p_med, mu_d_med), min(mu_p_med, mu_d_high), min(mu_p_high, mu_d_med))
        r_descend  = min(mu_p_high, mu_d_high)

        # Dé-fuzzification (Barycentre)
        numerator = (r_abort * self.OUT_ABORT) + (r_hold * self.OUT_HOLD) + \
                    (r_approach * self.OUT_APPROACH) + (r_descend * self.OUT_DESCEND)
        denominator = r_abort + r_hold + r_approach + r_descend + 1e-6
        
        score = numerator / denominator
        
        # Décision finale
        if score < 3.0: return "ABORT"
        elif score < 5.5: return "HOLD"
        elif score < 8.5: return "APPROACH"
        else: return "DESCEND"


# ==========================================
# 2. NOEUD PRINCIPAL (TOUR DE CONTRÔLE)
# ==========================================
class ControlTower(Node):
    def __init__(self):
        super().__init__('control_tower_node')

        # --- CONFIGURATION IA ---
        # Chemin absolu vers le modèle (à adapter selon votre dossier)
        self.onnx_path = "/home/t1204/Team_Delta/models/gru_paper.onnx"
        self.use_onnx = False
        
        try:
            self.ort_session = ort.InferenceSession(self.onnx_path)
            self.get_logger().info(f"✅ Modèle GRU chargé : {self.onnx_path}")
            self.use_onnx = True
        except Exception as e:
            self.get_logger().warn(f"⚠️ Erreur chargement ONNX : {e}. Le drone volera sans prédiction GRU.")

        self.fuzzy_engine = FuzzyMamdaniEngine()

        # --- CONFIGURATION ROS ---
        qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)

        # Subscribers
        self.sub_odom = self.create_subscription(Odometry, '/mavros/local_position/odom', self.odom_cb, qos_profile)
        self.sub_state = self.create_subscription(State, '/mavros/state', self.state_cb, qos_profile)
        self.sub_aruco = self.create_subscription(Vector3, '/drone/target_position', self.aruco_cb, 10)

        # Publishers
        self.pub_setpoint = self.create_publisher(PoseStamped, '/mavros/setpoint_position/local', 10)
        self.pub_status = self.create_publisher(String, '/control_tour/status', 10)

        # Variables d'état
        self.current_state = State()
        self.uav_pos = np.zeros(3)
        self.aruco_rel = None
        self.last_aruco_time = self.get_clock().now()

        # Filtre de Kalman (Estimation mouvement plateforme)
        self.setup_kalman()
        
        # Buffer pour le GRU (Historique)
        self.history_len = 20
        self.input_dim = 6
        self.history_buffer = np.zeros((self.history_len, self.input_dim), dtype=np.float32)

        # Boucle de contrôle (20 Hz)
        self.dt = 0.05 
        self.timer = self.create_timer(self.dt, self.control_loop)
        self.get_logger().info("🚀 Tour de Contrôle prête !")

    def setup_kalman(self):
        # Initialisation Kalman 9 états (Pos, Vel, Acc)
        self.xk = np.zeros((9, 1))
        self.Pk = np.eye(9) * 0.1
        self.F = np.eye(9) # Matrice de transition
        # ... (Configuration simplifiée de la matrice F pour modèle vitesse constante) ...
        dt = 0.05
        self.F[0,1]=dt; self.F[3,4]=dt; self.F[6,7]=dt 
        self.H = np.zeros((3, 9)); self.H[0,0]=1; self.H[1,3]=1; self.H[2,6]=1
        self.Q = np.eye(9) * 0.001
        self.R = np.eye(3) * 0.05

    def state_cb(self, msg): self.current_state = msg
    def odom_cb(self, msg):
        self.uav_pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])
    
    def aruco_cb(self, msg):
        self.aruco_rel = np.array([msg.x, msg.y, msg.z])
        self.last_aruco_time = self.get_clock().now()

    def sigmoid(self, x): return 1 / (1 + np.exp(-x))

    def control_loop(self):
        # Sécurité : Si le drone n'est pas en vol OFFBOARD, on ne calcule rien
        if self.current_state.mode != "OFFBOARD" or not self.current_state.armed:
            return

        # 1. VÉRIFICATION VISION
        # On considère la vision perdue si pas de message depuis 0.5 sec
        time_since_last = (self.get_clock().now() - self.last_aruco_time).nanoseconds / 1e9
        has_vision = (self.aruco_rel is not None) and (time_since_last < 0.5)

        # 2. ESTIMATION PLATEFORME (KALMAN)
        # Prédiction Kalman
        self.xk = np.dot(self.F, self.xk)
        self.Pk = np.dot(np.dot(self.F, self.Pk), self.F.T) + self.Q

        # Mise à jour Kalman si on voit l'ArUco
        if has_vision:
            # Position Absolue Plateforme = Position Drone + ArUco Relatif
            z_meas = (self.uav_pos + self.aruco_rel).reshape(3,1)
            
            y_res = z_meas - np.dot(self.H, self.xk)
            S = np.dot(np.dot(self.H, self.Pk), self.H.T) + self.R
            K = np.dot(np.dot(self.Pk, self.H.T), np.linalg.inv(S))
            self.xk = self.xk + np.dot(K, y_res)
            self.Pk = np.dot((np.eye(9) - np.dot(K, self.H)), self.Pk)

        # 3. GRU (IA)
        # Préparation des entrées pour le réseau
        gru_input = np.array([
            self.xk[0,0], self.xk[3,0], # Pos X, Y
            self.xk[1,0], self.xk[4,0], # Vel X, Y
            0.0,                        # Vent (inconnu)
            1.0 if has_vision else 0.0  # Masque Vision
        ], dtype=np.float32)
        
        # Mise à jour buffer historique
        self.history_buffer = np.roll(self.history_buffer, -1, axis=0)
        self.history_buffer[-1, :] = gru_input

        # Inférence
        dx, dy, pvis = 0.0, 0.0, 0.0
        if self.use_onnx:
            try:
                # Format (Batch, Seq, Feat)
                inp = self.history_buffer.reshape(1, self.history_len, self.input_dim)
                outs = self.ort_session.run(None, {self.ort_session.get_inputs()[0].name: inp})
                res = outs[0].flatten()
                dx, dy = float(res[0]), float(res[1])
                pvis = self.sigmoid(float(res[2]))
            except Exception:
                pass # On continue sans correction si erreur

        # 4. PRÉDICTION FINALE (Où sera la plateforme dans 0.1s ?)
        # Position Future = Position Kalman Actuelle + Vitesse * dt + Correction GRU
        pred_x = self.xk[0,0] + self.xk[1,0]*0.1 + dx
        pred_y = self.xk[3,0] + self.xk[4,0]*0.1 + dy
        pred_z = self.xk[6,0] # Z ne bouge pas trop

        # 5. DÉCISION (FUZZY LOGIC)
        dist_z = abs(self.uav_pos[2] - pred_z)
        d_norm = min(max((dist_z - 0.2)/1.5, 0.0), 1.0)
        
        mode = self.fuzzy_engine.compute(pvis, d_norm)

        # 6. ACTION (Génération commande)
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        if mode == "DESCEND":
            msg.pose.position.x = pred_x
            msg.pose.position.y = pred_y
            msg.pose.position.z = pred_z - 0.2 # On se pose (sous le niveau détecté)
            
        elif mode == "APPROACH":
            msg.pose.position.x = pred_x
            msg.pose.position.y = pred_y
            msg.pose.position.z = pred_z + 1.0 # 1m au dessus
            
        elif mode == "HOLD":
            # On reste là où on est
            msg.pose.position.x = self.uav_pos[0]
            msg.pose.position.y = self.uav_pos[1]
            msg.pose.position.z = self.uav_pos[2]
            
        else: # ABORT
            # On monte
            msg.pose.position.x = self.uav_pos[0]
            msg.pose.position.y = self.uav_pos[1]
            msg.pose.position.z = self.uav_pos[2] + 1.5

        self.pub_setpoint.publish(msg)
        self.pub_status.publish(String(data=f"Mode:{mode} | Pvis:{pvis:.2f} | Dist:{dist_z:.2f}"))

def main(args=None):
    rclpy.init(args=args)
    node = ControlTower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

"""



# ============================================================================ #
#                       Original Code - Full Flight Commands                   #
# ============================================================================ #

"""
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import math

# Gestion ONNX
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Float32MultiArray, Bool
from mavros_msgs.msg import State
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# ==========================================
# 1. MOTEUR DE LOGIQUE FLOUE (MAMDANI)
# ==========================================
class FuzzyMamdaniEngine:
    def __init__(self):
        self.OUT_ABORT = 1.5
        self.OUT_HOLD = 4.0
        self.OUT_APPROACH = 7.0
        self.OUT_DESCEND = 9.5

    def triangle(self, x, a, b, c):
        return max(0.0, min((x - a)/(b - a + 1e-6), (c - x)/(c - b + 1e-6)))

    def trapezoid(self, x, a, b, c, d):
        return max(0.0, min(min((x - a)/(b - a + 1e-6), 1.0), (d - x)/(d - c + 1e-6)))

    def compute(self, pvis, d_norm):
        # Fuzzification
        mu_p_low  = self.trapezoid(pvis, -0.1, 0.0, 0.3, 0.5)
        mu_p_med  = self.triangle(pvis, 0.3, 0.5, 0.8)
        mu_p_high = self.trapezoid(pvis, 0.6, 0.8, 1.0, 1.1)
        
        mu_d_low  = self.trapezoid(d_norm, -0.1, 0.0, 0.2, 0.4)
        mu_d_med  = self.triangle(d_norm, 0.2, 0.5, 0.8)
        mu_d_high = self.trapezoid(d_norm, 0.6, 0.8, 1.0, 1.1)

        # Inférence
        r_abort    = max(min(mu_p_low, mu_d_low), min(mu_p_med, mu_d_low), min(mu_p_high, mu_d_low))
        r_hold     = max(min(mu_p_low, mu_d_med), min(mu_p_low, mu_d_high))
        r_approach = max(min(mu_p_med, mu_d_med), min(mu_p_med, mu_d_high), min(mu_p_high, mu_d_med))
        r_descend  = min(mu_p_high, mu_d_high)

        # Dé-fuzzification
        numerator = (r_abort * self.OUT_ABORT) + (r_hold * self.OUT_HOLD) + \
                    (r_approach * self.OUT_APPROACH) + (r_descend * self.OUT_DESCEND)
        denominator = r_abort + r_hold + r_approach + r_descend + 1e-6
        
        score = numerator / denominator
        
        if score < 3.0: return "ABORT"
        elif score < 5.5: return "HOLD"
        elif score < 8.5: return "APPROACH"
        else: return "DESCEND"


# ==========================================
# 2. NOEUD PRINCIPAL (CERVEAU)
# ==========================================
class ControlTower(Node):
    def __init__(self):
        super().__init__('control_tower_node')

        # --- 1. CONFIGURATION ---
        self.onnx_path = "/home/t1204/Team_Delta/models/gru_paper.onnx"
        self.use_onnx = False
        
        if ONNX_AVAILABLE:
            try:
                self.ort_session = ort.InferenceSession(self.onnx_path)
                self.get_logger().info(f"✅ Modèle GRU chargé : {self.onnx_path}")
                self.use_onnx = True
            except Exception as e:
                self.get_logger().warn(f"⚠️ Erreur chargement ONNX : {e}")
        else:
            self.get_logger().warn("⚠️ Module onnxruntime manquant.")

        self.fuzzy_engine = FuzzyMamdaniEngine()
        
        # Paramètres d'historique (Doivent matcher platform_state_node)
        self.history_len = 20 
        self.input_dim = 6    

        # --- 2. SUBSCRIPTIONS ---
        qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)

        # État du drone (MAVROS)
        self.sub_state = self.create_subscription(State, '/mavros/state', self.state_cb, qos_profile)
        
        # Position du drone (MAVROS) - Pour connaître notre propre altitude Z
        self.sub_odom = self.create_subscription(Odometry, '/mavros/local_position/odom', self.odom_cb, qos_profile)

        # Position cible (ARUCO) - Pour savoir où est la plateforme actuellement
        # Note : On écoute maintenant PoseStamped venant de aruco_node
        self.sub_aruco_pose = self.create_subscription(PoseStamped, '/platform/pose_cam', self.aruco_pose_cb, 10)

        # Vecteur préparé pour le GRU (Platform State Node)
        self.sub_gru_input = self.create_subscription(Float32MultiArray, '/gru_input_vector', self.gru_input_cb, 10)

        # --- 3. PUBLISHERS ---
        self.pub_setpoint = self.create_publisher(PoseStamped, '/mavros/setpoint_position/local', 10)
        self.pub_status = self.create_publisher(String, '/control_tour/status', 10)

        # --- 4. VARIABLES ---
        self.current_state = State()
        self.uav_pos = np.zeros(3) # [x, y, z] du drone
        self.gru_vector = None     # Dernier vecteur reçu
        
        self.aruco_pos_cam = None  # Dernière position vue de l'ArUco
        self.last_aruco_time = self.get_clock().now()

        # Boucle de contrôle à 20 Hz
        self.timer = self.create_timer(0.05, self.control_loop)
        self.get_logger().info("🧠 Cerveau prêt et en attente des données...")

    # --- CALLBACKS ---
    def state_cb(self, msg): 
        self.current_state = msg

    def odom_cb(self, msg):
        self.uav_pos[0] = msg.pose.pose.position.x
        self.uav_pos[1] = msg.pose.pose.position.y
        self.uav_pos[2] = msg.pose.pose.position.z

    def aruco_pose_cb(self, msg):
        # ArUco position relative à la caméra
        self.aruco_pos_cam = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        self.last_aruco_time = self.get_clock().now()

    def gru_input_cb(self, msg):
        # Stocke le vecteur reçu de platform_state_node
        # msg.data est une liste plate (flattened)
        self.gru_vector = np.array(msg.data, dtype=np.float32)

    def sigmoid(self, x): 
        return 1 / (1 + np.exp(-x))

    # --- BOUCLE PRINCIPALE ---
    def control_loop(self):
        # 0. SÉCURITÉ : Si pas Offboard ou pas Armé, on ne fait rien
        if self.current_state.mode != "OFFBOARD" or not self.current_state.armed:
            return

        # 1. INFÉRENCE IA (GRU)
        dx, dy, pvis = 0.0, 0.0, 0.0
        
        if self.use_onnx and self.gru_vector is not None:
            try:
                # Le platform_state_node envoie un vecteur concaténé : [x_hist, y_hist, vx_hist, vy_hist, w, m]
                # Il faut le remettre en forme pour le modèle ONNX : (Batch, Seq, Feat) -> (1, 20, 6)
                
                # Reconstruction du tenseur d'entrée
                # Attention : Cette reconstruction dépend de comment platform_state_node a aplati les données.
                # Supposons ici que platform_state_node a envoyé 4 tableaux de 20 + 2 scalaires.
                
                raw_data = self.gru_vector
                
                # Extraction des blocs (20 éléments chacun)
                N = self.history_len
                x_hist  = raw_data[0:N]
                y_hist  = raw_data[N:2*N]
                vx_hist = raw_data[2*N:3*N]
                vy_hist = raw_data[3*N:4*N]
                w_t     = raw_data[4*N]   # Wind
                m_t     = raw_data[4*N+1] # Mask
                
                # Construction de la matrice (20, 6)
                input_matrix = np.zeros((1, N, 6), dtype=np.float32)
                for i in range(N):
                    input_matrix[0, i, 0] = x_hist[i]
                    input_matrix[0, i, 1] = y_hist[i]
                    input_matrix[0, i, 2] = vx_hist[i]
                    input_matrix[0, i, 3] = vy_hist[i]
                    input_matrix[0, i, 4] = w_t # Suppose constant sur la fenêtre ou dernière valeur
                    input_matrix[0, i, 5] = m_t # Idem

                # Exécution ONNX
                inputs = {self.ort_session.get_inputs()[0].name: input_matrix}
                outputs = self.ort_session.run(None, inputs)
                
                # Résultats
                res = outputs[0].flatten()
                dx, dy = float(res[0]), float(res[1])
                pvis = self.sigmoid(float(res[2]))
                
            except Exception as e:
                # self.get_logger().warn(f"Erreur inférence : {e}")
                pass

        # 2. LOGIQUE FLOUE (DÉCISION)
        
        # Calcul de la distance Z (Altitude) par rapport à la plateforme
        # Si on voit l'ArUco, on utilise la distance réelle mesurée par la caméra
        time_since_vision = (self.get_clock().now() - self.last_aruco_time).nanoseconds / 1e9
        is_vision_fresh = time_since_vision < 0.5

        dist_z = 2.0 # Valeur par défaut (haut)
        
        if is_vision_fresh and self.aruco_pos_cam is not None:
            # En caméra frame, Z est souvent la profondeur (distance avant), 
            # mais selon l'orientation (downward facing), Z peut être l'altitude.
            # Avec realsense downward : Z caméra = Distance verticale
            dist_z = abs(self.aruco_pos_cam[2])
        else:
            # Si on ne voit rien, on utilise la confiance du réseau (pvis) 
            # ou on suppose qu'on est loin
            pass

        # Normalisation pour le moteur flou
        d_norm = min(max((dist_z - 0.2)/1.5, 0.0), 1.0)
        
        # Décision
        mode = self.fuzzy_engine.compute(pvis, d_norm)

        # 3. GÉNÉRATION COMMANDE (SETPOINT)
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        # Cible (X,Y) = Dernière position connue (via history) + Correction GRU (dx, dy)
        # On prend la dernière valeur du buffer d'historique comme "position actuelle estimée de la plateforme"
        target_x = self.uav_pos[0] # Par défaut, sur place
        target_y = self.uav_pos[1]
        
        if self.gru_vector is not None and len(self.gru_vector) > 0:
             # Dernière position connue du bateau (dernier élément de l'historique)
             # Index: Le vecteur est [X_hist(20), Y_hist(20)...]
             # Le dernier X est à l'index 19
             # Le dernier Y est à l'index 39
             target_x = self.gru_vector[19] + dx
             target_y = self.gru_vector[39] + dy

        # Application du Mode de vol
        if mode == "DESCEND":
            msg.pose.position.x = target_x
            msg.pose.position.y = target_y
            msg.pose.position.z = self.uav_pos[2] - 0.2 # On descend doucement
            
        elif mode == "APPROACH":
            msg.pose.position.x = target_x
            msg.pose.position.y = target_y
            msg.pose.position.z = max(self.uav_pos[2], 1.0) # On maintient min 1m ou altitude actuelle
            
        elif mode == "HOLD":
            # Stationnaire à la position actuelle du DRONE (pas du bateau)
            msg.pose.position.x = self.uav_pos[0]
            msg.pose.position.y = self.uav_pos[1]
            msg.pose.position.z = self.uav_pos[2]
            
        else: # ABORT
            # On remonte d'urgence
            msg.pose.position.x = self.uav_pos[0]
            msg.pose.position.y = self.uav_pos[1]
            msg.pose.position.z = self.uav_pos[2] + 1.5

        # Envoi
        self.pub_setpoint.publish(msg)
        
        # Debug
        status_msg = f"Mode:{mode} | Pvis:{pvis:.2f} | Dist:{dist_z:.2f}m"
        self.pub_status.publish(String(data=status_msg))

def main(args=None):
    rclpy.init(args=args)
    node = ControlTower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
"""

# ============================================================================ #
#                       Code for testing - No flight commands                  #
# ============================================================================ #

#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import math

# ONNX Management
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Float32MultiArray
from mavros_msgs.msg import State
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# ==========================================
# 1. FUZZY LOGIC ENGINE (MAMDANI)
# ==========================================
class FuzzyMamdaniEngine:
    def __init__(self):
        # Output Scores (Actions)
        self.OUT_ABORT = 1.5
        self.OUT_HOLD = 4.0
        self.OUT_APPROACH = 7.0
        self.OUT_DESCEND = 9.5

    def triangle(self, x, a, b, c):
        return max(0.0, min((x - a)/(b - a + 1e-6), (c - x)/(c - b + 1e-6)))

    def trapezoid(self, x, a, b, c, d):
        return max(0.0, min(min((x - a)/(b - a + 1e-6), 1.0), (d - x)/(d - c + 1e-6)))

    def compute(self, pvis, d_norm):
        # Fuzzification
        mu_p_low  = self.trapezoid(pvis, -0.1, 0.0, 0.3, 0.5)
        mu_p_med  = self.triangle(pvis, 0.3, 0.5, 0.8)
        mu_p_high = self.trapezoid(pvis, 0.6, 0.8, 1.0, 1.1)
        
        mu_d_low  = self.trapezoid(d_norm, -0.1, 0.0, 0.2, 0.4)
        mu_d_med  = self.triangle(d_norm, 0.2, 0.5, 0.8)
        mu_d_high = self.trapezoid(d_norm, 0.6, 0.8, 1.0, 1.1)

        # Inference (Rules)
        r_abort    = max(min(mu_p_low, mu_d_low), min(mu_p_med, mu_d_low), min(mu_p_high, mu_d_low))
        r_hold     = max(min(mu_p_low, mu_d_med), min(mu_p_low, mu_d_high))
        r_approach = max(min(mu_p_med, mu_d_med), min(mu_p_med, mu_d_high), min(mu_p_high, mu_d_med))
        r_descend  = min(mu_p_high, mu_d_high)

        # De-fuzzification (Centroid)
        numerator = (r_abort * self.OUT_ABORT) + (r_hold * self.OUT_HOLD) + \
                    (r_approach * self.OUT_APPROACH) + (r_descend * self.OUT_DESCEND)
        denominator = r_abort + r_hold + r_approach + r_descend + 1e-6
        
        score = numerator / denominator
        
        # Final Decision
        if score < 3.0: return "ABORT"
        elif score < 5.5: return "HOLD"
        elif score < 8.5: return "APPROACH"
        else: return "DESCEND"


# ==========================================
# 2. MAIN NODE (PASSIVE OBSERVER)
# ==========================================
class ControlTower(Node):
    def __init__(self):
        super().__init__('control_tower_node')

        self.get_logger().warn("⚠️ OBSERVER MODE: No commands will be sent to the drone.")
        self.get_logger().info("ℹ️ Move the drone by hand to test state changes.")

        # --- AI CONFIGURATION ---
        self.onnx_path = "/home/t1204/Team_Delta/src/control_tour/models/gru_paper.onnx"
        self.use_onnx = False
        
        if ONNX_AVAILABLE:
            try:
                self.ort_session = ort.InferenceSession(self.onnx_path)
                self.get_logger().info(f"✅ GRU Model loaded: {self.onnx_path}")
                self.use_onnx = True
            except Exception as e:
                self.get_logger().warn(f"⚠️ ONNX Load Error: {e}")
        else:
            self.get_logger().warn("⚠️ onnxruntime module missing.")

        self.fuzzy_engine = FuzzyMamdaniEngine()
        self.history_len = 20 
        self.input_dim = 6    

        # --- SUBSCRIPTIONS ---
        qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)

        # Listen to everything
        self.sub_state = self.create_subscription(State, '/mavros/state', self.state_cb, qos_profile)
        self.sub_odom = self.create_subscription(Odometry, '/mavros/local_position/odom', self.odom_cb, qos_profile)
        self.sub_aruco_pose = self.create_subscription(PoseStamped, '/platform/pose_cam', self.aruco_pose_cb, 10)
        self.sub_gru_input = self.create_subscription(Float32MultiArray, '/gru_input_vector', self.gru_input_cb, 10)

        # --- PUBLISHERS (Status only) ---
        # NOTE: self.pub_setpoint has been REMOVED to guarantee no movement commands
        self.pub_status = self.create_publisher(String, '/control_tour/status', 10)

        # NEW: mode only, as its own topic
        self.pub_mode = self.create_publisher(
            String, '/control_tour/mode', 10
        )
        # NEW: GRU state (Active / Inactive / Error)
        self.pub_gru_status = self.create_publisher(
            String, '/control_tour/gru_status', 10
        )
        # NEW: numeric vector with key values (easy to use from Python/MATLAB)
        # data layout:
        # [0]  = pvis
        # [1]  = dist_z
        # [2]  = d_norm
        # [3]  = uav_pos_x
        # [4]  = uav_pos_y
        # [5]  = uav_pos_z
        # [6]  = target_x
        # [7]  = target_y
        # [8]  = time_since_vision
        self.pub_decision_vec = self.create_publisher(
            Float32MultiArray, '/control_tour/decision_vector', 10
        )


        # --- STATE VARIABLES ---
        self.current_state = State()
        self.uav_pos = np.zeros(3)
        self.gru_vector = None     
        self.aruco_pos_cam = None 
        self.last_aruco_time = self.get_clock().now()

        # Timer
        self.timer = self.create_timer(0.05, self.control_loop)
        
    # --- CALLBACKS ---
    def state_cb(self, msg): self.current_state = msg
    def odom_cb(self, msg):
        self.uav_pos[0] = msg.pose.pose.position.x
        self.uav_pos[1] = msg.pose.pose.position.y
        self.uav_pos[2] = msg.pose.pose.position.z

    def aruco_pose_cb(self, msg):
        self.aruco_pos_cam = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        self.last_aruco_time = self.get_clock().now()

    def gru_input_cb(self, msg):
        self.gru_vector = np.array(msg.data, dtype=np.float32)

    def sigmoid(self, x): return 1 / (1 + np.exp(-x))

    # --- MAIN LOOP (READ ONLY) ---
    def control_loop(self):
        
        # NOTE: No "if armed" check, so we can test on the bench.

        # 1. AI INFERENCE (GRU)
        dx, dy, pvis = 0.0, 0.0, 0.0
        gru_status = "Inactive"

        if self.use_onnx and self.gru_vector is not None:
            try:
                # Reconstruct data
                raw_data = self.gru_vector
                N = self.history_len
                
                # Simplified extraction for inference
                x_hist  = raw_data[0:N]
                y_hist  = raw_data[N:2*N]
                vx_hist = raw_data[2*N:3*N]
                vy_hist = raw_data[3*N:4*N]
                w_t     = raw_data[4*N]
                m_t     = raw_data[4*N+1]
                
                input_matrix = np.zeros((1, N, 6), dtype=np.float32)
                for i in range(N):
                    input_matrix[0, i, 0] = x_hist[i]
                    input_matrix[0, i, 1] = y_hist[i]
                    input_matrix[0, i, 2] = vx_hist[i]
                    input_matrix[0, i, 3] = vy_hist[i]
                    input_matrix[0, i, 4] = w_t
                    input_matrix[0, i, 5] = m_t

                inputs = {self.ort_session.get_inputs()[0].name: input_matrix}
                outputs = self.ort_session.run(None, inputs)
                res = outputs[0].flatten()
                dx, dy = float(res[0]), float(res[1])
                pvis = self.sigmoid(float(res[2]))
                gru_status = "Active"
            except Exception:
                gru_status = "Error"

        # 2. FUZZY LOGIC
        # Calculate distance
        time_since_vision = (self.get_clock().now() - self.last_aruco_time).nanoseconds / 1e9
        is_vision_fresh = time_since_vision < 0.5
        
        dist_z = 0.0
        source_dist = "Unknown"

        if is_vision_fresh and self.aruco_pos_cam is not None:
            # Use real distance seen by camera
            dist_z = abs(self.aruco_pos_cam[2])
            source_dist = "Camera"
        else:
            # If no vision, use drone altitude (assuming target is at Z=0)
            # Just for simulation/testing purposes
            dist_z = max(self.uav_pos[2], 0.0) 
            source_dist = "Baro/GPS"

        # Fuzzy Calculation
        d_norm = min(max((dist_z - 0.2)/1.5, 0.0), 1.0)
        mode = self.fuzzy_engine.compute(pvis, d_norm)

        # 3. VIRTUAL TARGET CALCULATION (What the drone WOULD target)
        target_x = self.uav_pos[0]
        target_y = self.uav_pos[1]
        
        if self.gru_vector is not None and len(self.gru_vector) > 0:
             # Simulate correction
             target_x = self.gru_vector[19] + dx
             target_y = self.gru_vector[39] + dy

        # 4. DISPLAY DECISIONS
        
        # Color codes for terminal
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        RED = '\033[91m'
        RESET = '\033[0m'
        
        color = RESET
        if mode == "DESCEND": color = GREEN
        elif mode == "APPROACH": color = GREEN
        elif mode == "HOLD": color = YELLOW
        elif mode == "ABORT": color = RED

        # Construct log message
        log_msg = (
            f"DECISION: {color}{mode:<8}{RESET} | "
            f"AI Conf: {pvis:.2f} ({gru_status}) | "
            f"Height: {dist_z:.2f}m ({source_dist}) | "
            f"Target: X={target_x:.1f} Y={target_y:.1f}"
        )
        
        self.get_logger().info(log_msg)
        
        # Publish to status topic for RQT
        self.pub_status.publish(String(data=log_msg))

        # 5. NEW: PUBLISH STRUCTURED VALUES FOR RECORDING
        # Mode as its own topic
        self.pub_mode.publish(String(data=mode))

        # GRU status as its own topic
        self.pub_gru_status.publish(String(data=gru_status))

        # Numeric vector with all key values
        vec = Float32MultiArray()
        vec.data = [
            float(pvis),
            float(dist_z),
            float(d_norm),
            float(self.uav_pos[0]),
            float(self.uav_pos[1]),
            float(self.uav_pos[2]),
            float(target_x),
            float(target_y),
            float(time_since_vision),
        ]
        self.pub_decision_vec.publish(vec)

def main(args=None):
    rclpy.init(args=args)
    node = ControlTower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()