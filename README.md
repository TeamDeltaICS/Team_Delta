# Team Delta - Drone Project

This repository contains all resources for the Team Delta drone project. It is divided into two main parts: simulation (MATLAB/CoppeliaSim) and embedded code (ROS).

---

## 📂 Repository Structure

*   **`Matlab_Simulation/`**: Contains the simulation environment.
    *   `*.ttt`: Scenes for the CoppeliaSim simulator.
    *   `*.m`: Control scripts and algorithms using MATLAB.
*   **`ROS_Drone/`**: Contains the source code for the real drone.
    *   `src/`: Source code of the ROS packages (to be compiled on the target machine).

---

## 🎮 Simulation (MATLAB & CoppeliaSim)

To run the simulation on a PC:

1.  Open **CoppeliaSim**.
2.  Load the scene found in the `Matlab_Simulation` folder (`.ttt` file).
3.  Open **MATLAB**.
4.  Navigate to the `Matlab_Simulation` folder.
5.  Run the main script (`.m`) to initiate the connection with the simulator.

---

## 🚀 Installation & Usage (ROS Part)

The `ROS_Drone` folder contains only the source code (`src`). To use it on a new machine (or on the robot), you must recreate the workspace and compile the code.

### Prerequisites
*   Ubuntu (version compatible with your ROS distribution)
*   ROS 2 (with `colcon` installed)

### Steps to install on a new machine

1.  **Clone this repository** (if you haven't already):
    ```bash
    git clone https://github.com/TeamDeltaCS/Team_Delta.git
    ```

2.  **Create a Workspace**:
    Open a terminal and create a folder for your workspace:
    ```bash
    mkdir -p ~/team_delta_ws/src
    ```

3.  **Copy the source code**:
    Copy the contents of the `ROS_Drone/src` folder from this repository into the `src` folder of your new workspace:
    ```bash
    # Example copy command (adjust the path to the cloned folder)
    cp -r ~/path/to/Team_Delta/ROS_Drone/src/* ~/team_delta_ws/src/
    ```

4.  **Install dependencies**:
    This step ensures all necessary libraries are installed on your system:
    ```bash
    cd ~/team_delta_ws
    rosdep install --from-paths src --ignore-src -r -y
    ```

5.  **Build the project**:
    Generate the install and build files:
    ```bash
    colcon build
    ```

6.  **Source the environment**:
    Before running any ROS commands, you must source the workspace:
    ```bash
    source install/setup.bash
    ```
    *(Tip: Add this line to your `.bashrc` file so you don't have to type it every time).*

---

## 👥 Authors
*   **Team Delta**
*   Sergi Sanmartin
*   Gaëtan Soudée
*   Marcel Naderer
