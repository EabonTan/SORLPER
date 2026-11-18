#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, Path
# from tf import TransformListener
import math
# from gazebo_msgs.srv import SetModelState
# from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Pose, Quaternion


class DQN(nn.Module):
    def __init__(self, state_size, action_size, device):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        return self.fc(x)


class Dn1Robot:
    def __init__(self):
        # Initialize ROS
        rospy.init_node('dn1_robot', anonymous=True)
        # Publisher and Subscriber
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.laser_sub = rospy.Subscriber('/scan', LaserScan, self.callback_laser_scan, queue_size=1)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.callback_odometry, queue_size=1)  # Now referencing a defined method
        
        # Path Publisher
        self.path_pub = rospy.Publisher('/robot_trajectory', Path, queue_size=10)
        self.path = Path()
        self.path.header.frame_id = "map"  # Set the frame of reference for the path


        # Initialize parameters
        self.agent_position = np.array([0, 0])  # Initial agent position
        self.friend_position = np.array([4, 4])  # Target position
        self.d_em = 0.2  # Collision distance
        self.d_fri = 0.5 # Arrival distance

        self.current_angle = 0.0  # Initialize current angle
        
        # Define control parameters
        # self.control_rate = rospy.Rate(10)  # 10 Hz
        model_path = "model/Q_net.pth" 
        self.weighting_decay = 10
        self.device = torch.device("cpu")
        self.q_net = DQN(6, 36, device=torch.device("cpu")).to(self.device)
        self.q_net.load_state_dict(torch.load(model_path))  # Load the trained model
        self.q_net.eval()  # Set the network to evaluation mode
     
    def callback_laser_scan(self, scan_data):
        laser_ranges = scan_data.ranges
        angle_min = scan_data.angle_min
        angle_increment = scan_data.angle_increment
        laser_points = []
        for i, distance in enumerate(laser_ranges):
            if distance < scan_data.range_max:
                angle = angle_min + i * angle_increment
                x = distance * math.cos(angle)
                y = distance * math.sin(angle)
                laser_points.append((x, y))

        d_e, d_index = min(laser_ranges), laser_ranges.index(min(laser_ranges))
        enemy_angle = angle_min + d_index * angle_increment
        enemy = (math.cos(enemy_angle) * d_e, math.sin(enemy_angle) * d_e)
        #if enemy_angle > -math.pi/2 and enemy_angle < math.pi:
        #    enemy_angle =  enemy_angle + math.pi/2
        #elif enemy_angle >= -math.pi and enemy_angle < math.pi/2:
        #   enemy_angle =  enemy_angle + 5*math.pi/2
        return -enemy_angle, enemy

    # The missing callback function for the Odometry data
    def callback_odometry(self, odom_data):
        # Process odometry data to update the robot's position and current angle
        self.agent_position = (odom_data.pose.pose.position.x, odom_data.pose.pose.position.y)
        quaternion = (
            odom_data.pose.pose.orientation.x,
            odom_data.pose.pose.orientation.y,
            odom_data.pose.pose.orientation.z,
            odom_data.pose.pose.orientation.w
        )
        euler = self.quaternion_to_euler(quaternion)
        self.current_angle = euler[2]  

    def quaternion_to_euler(self, quat):
        # Convert quaternion to Euler angles
        x, y, z, w = quat
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        theta_z = math.atan2(t3, t4)

        return roll_x, pitch_y, theta_z 

    def update_path(self, pose):
        # Create a PoseStamped message and append it to the Path
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = rospy.Time.now()
        pose_stamped.header.frame_id = "map"
        pose_stamped.pose = pose
        self.path.poses.append(pose_stamped)
        self.path_pub.publish(self.path)

    def get_current_angle(self):
        return self.current_angle  
        
    def move_robot(self, odom_data, action, enemy_dis):
        cmd_msg = Twist()        
        target_angle = action * (360 / 36)  # Convert action to angle        
        target_angle_rad = math.radians(target_angle)  # Convert angle to rad
        if target_angle_rad < math.pi:
            target_angle_rad = target_angle_rad + self.current_angle
        else:
            target_angle_rad = target_angle_rad - 2 * math.pi + self.current_angle
        tolerance = math.radians(0.1)  
        current_angle_rad = self.current_angle
        print("the current_angle_rad", current_angle_rad)
        print("the target_angle_rad", target_angle_rad)

        
        while True:  
            self.get_current_angle()
            current_angle_rad = self.current_angle
            angle_diff_rad = (target_angle_rad - current_angle_rad + math.pi) % (2 * math.pi) - math.pi  
            # angle_diff_rad = target_angle_rad - current_angle_rad
            if abs(angle_diff_rad) < tolerance:  
                rospy.sleep(0.2)  # Check again after a brief pause to ensure it's stable
                if abs(angle_diff_rad) < tolerance:  # Double-check the angle difference
                    break  # Target angle reached, stop rotation  

            cmd_msg.angular.z = 0.2 if angle_diff_rad > 0 else -0.2  
            self.cmd_pub.publish(cmd_msg)  
            rospy.sleep(0.1)  

      
        cmd_msg.angular.z = 0.0  
        self.cmd_pub.publish(cmd_msg)
        rospy.sleep(1)  
        laser_scan_data = rospy.wait_for_message('/scan', LaserScan)
        enemy_angle, enemy_pos = self.callback_laser_scan(laser_scan_data)

        if enemy_dis >= 0.3:
            cmd_msg.linear.x = 0.2  
            start_time = rospy.Time.now()
            print("moving", cmd_msg.linear.x)
            while rospy.Time.now() - start_time < rospy.Duration(1):
                self.cmd_pub.publish(cmd_msg)
                # rospy.sleep(0.1)
            print("moving time_jiange", cmd_msg.linear.x)
        else:
            cmd_msg.linear.x = 0.0
            self.cmd_pub.publish(cmd_msg)
            rospy.sleep(1)
            if abs(enemy_angle) >= np.pi/4:
                cmd_msg.linear.x = 0.2
                start_time = rospy.Time.now()
                print("moving", cmd_msg.linear.x)
                while rospy.Time.now() - start_time < rospy.Duration(1):
                    self.cmd_pub.publish(cmd_msg)
        # self.run()        
        
        
        cmd_msg.linear.x = 0.0
        self.cmd_pub.publish(cmd_msg)
        print("the diff of target_angle_rad and current_angle_rad", angle_diff_rad)

    def run(self):
        step = 0
        while not rospy.is_shutdown():  
           
            odom_data = rospy.wait_for_message('/odom', Odometry)
            agent_pos = self.agent_position

          
            friend_pos = self.friend_position  
            laser_scan_data = rospy.wait_for_message('/scan_1', LaserScan)
            # DN input
            enemy_angle, enemy_pos = self.callback_laser_scan(laser_scan_data)
            enemy_pos_r = [0, 0]
            # obstacle zuobiao bianhuan
            enemy_pos_r[0] = enemy_pos[0] + agent_pos[0]
            enemy_pos_r[1] = enemy_pos[1] + agent_pos[1]

            enemy_dis = np.linalg.norm(np.array([enemy_pos[1], enemy_pos[0]]) - np.array([0, 0]))

            fri_dis = np.linalg.norm(friend_pos - agent_pos)
            fri_angle = math.atan2((friend_pos - agent_pos)[1], (friend_pos - agent_pos)[0])
            fri_angle = fri_angle - self.current_angle
            print(fri_dis, enemy_dis, fri_angle, enemy_angle)
            print("enemy_pos corrdinate is :{}", enemy_pos)
            print("enemy_pos_r corrdinate is :{}", enemy_pos_r)
            print("robot corrdinate is :{}", agent_pos)

            
            state = [np.array([np.cos(fri_angle), np.sin(fri_angle), np.cos(enemy_angle), np.sin(enemy_angle),
                          fri_dis / (fri_dis + enemy_dis), enemy_dis / (fri_dis + enemy_dis)])]
            state_tensor = torch.tensor(state, dtype=torch.float).to(self.device)
            action = self.q_net(state_tensor).argmax().item()  # Choose the best action
            cmd_msg = Twist() 
            if fri_dis < self.d_fri:
                rospy.loginfo("Reached the target!step:{}".format(step))
                cmd_msg.linear.x = 0.0
                self.cmd_pub.publish(cmd_msg)
                break
            elif enemy_dis <= self.d_em:
                rospy.loginfo("Touch the obstacles!!!step:{}".format(step))
                cmd_msg.linear.x = 0.0
                self.cmd_pub.publish(cmd_msg)
                break
           
            self.move_robot(odom_data, action, enemy_dis)  #
            step += 1

            rospy.sleep(0.1)  


if __name__ == '__main__':
    try:
        robot = Dn1Robot()
        robot.run()
    except rospy.ROSInterruptException:
        pass
