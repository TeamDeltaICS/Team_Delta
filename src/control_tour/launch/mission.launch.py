import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    
    # --- CONFIGURATION ---
    fcu_url = 'serial:///dev/ttyACM1:57600'
    
    # --- COMMANDE POUR NOUVEAU TERMINAL ---
    # Cette commande ouvre une fenêtre, donne un titre, et exécute le node
    # L'option '--' signifie "la fin des options du terminal, voici la commande à lancer"
    new_terminal_prefix = 'gnome-terminal -- ' 
    # Si vous voulez nommer la fenêtre (plus propre) :
    # new_terminal_prefix = 'gnome-terminal --title="MON TITRE" -- '

    # --- 1. DRIVER CAMÉRA (REALSENSE) ---
    # On laisse la caméra dans le terminal principal (celui où vous tapez la commande)
    # car c'est un fichier launch inclus, c'est plus complexe à détacher.
    realsense_dir = get_package_share_directory('realsense2_camera')
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(realsense_dir, 'launch', 'rs_launch.py')),
        launch_arguments={
            'align_depth.enable': 'true',
            'enable_gyro': 'true',
            'enable_accel': 'true'
        }.items()
    )

    # --- 1. BRAIN (Same terminal) ---
    brain_node = Node(
        package='control_tour',
        executable='brain_node',
        name='control_tower',
        output='screen',
        #prefix='gnome-terminal --title="BRAIN (AI)" --',
    )
 
    # --- 2. MAVROS (In a new terminal) ---
    mavros_node = Node(
        package='mavros',
        executable='mavros_node',
        output='screen',
        prefix='gnome-terminal --title="MAVROS (Pixhawk)" --', 
        parameters=[{
            'fcu_url': fcu_url,
            'system_id': 255,
            'component_id': 191,
            'target_system_id': 1,
            'target_component_id': 1,
        }]
    )

    # --- 3. VISION (In a new terminal) ---
    aruco_node = Node(
        package='camera_vision',
        executable='aruco_node',
        name='aruco_detector',
        output='screen',
        prefix='gnome-terminal --title="VISION (ArUco)" --',
    )
   
    # --- 4. LOGGER PIXHAWK (In a new terminal) ---
    pixhawk_reader_node = Node(
        package='pixhawk_reader',
        executable='pixhawk_reader_node',
        name='pixhawk_logger',
        output='screen',
        prefix='gnome-terminal --title="IMU - GPS" --',
    )

    # --- 5. PLATFORM (In a new terminal) ---
    platform_node = Node(
       package='plattform_status',
       executable='platform_state_node',
       name='boat_monitor',
       output='screen',
       prefix='gnome-terminal --title="PLATFORM" --',
    )


    return LaunchDescription([
        realsense_launch,
        
        # On lance MAVROS dans sa fenêtre
        mavros_node,
        
        # On lance les autres après les délais
        TimerAction(period=5.0, actions=[aruco_node, pixhawk_reader_node]),
        TimerAction(period=6.0, actions=[brain_node, platform_node]),
    ])