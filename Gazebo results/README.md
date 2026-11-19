## Prerequisites

The following software environment is required to run the code on Linux:

- **Python:** 3.6.9
- **Operating System:** Ubuntu 18.04
- **ROS:** ROS Melodic (Full Desktop installation, including Gazebo)

Please clone the [Gazebo Models repository](https://github.com/osrf/gazebo_models) and download the models into the following directory:
~/.gazebo/models

## Running the Gazebo Simulation

To launch the static simulation environment, follow these steps:

**Terminal 1:** Launch the Autolabor robot's Gazebo simulation environment.

roslaunch autolabor_pro1_world.launch


**Terminal 2:** Run the robot navigation testing script.

python SORLPER_test.py


## Running the Dynamic Environment

To test the robot's navigation performance in a dynamic environment, launch the following Gazebo world instead:

roslaunch autolabor_pro1_dynamic_world.launch


After the dynamic environment has been launched, run the navigation testing script in another terminal:

python SORLPER_test.py
