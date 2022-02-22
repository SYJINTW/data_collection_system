import os
import subprocess
import csv
import numpy as np
import pandas as pd

# ============================================================
#        |  X|  Y|  Z| Yaw| Pitch| Roll|
# UE     |  a|  b|  c|   d|     e|    f|
# Airsim |  a|  b| -c|  -d|     e|    f|
# MIV    |  a| -b|  c|   d|    -e|    f|
# ============================================================

# Basic setting
# ============================================================
camera_positions = ['round']
scenes = ['test']
# ============================================================

class Camera_pose:
    name = ""
    position = [0, 0, 0] # X, Y, Z
    rotation = [0, 0, 0] # Yaw, Pitch, Roll
    
    def __init__(self, _array=['',0,0,0,0,0,0]):
        self.name = _array[0]
        self.position = [float(x) for x in _array[1:4]]
        self.rotation = [float(x) for x in _array[4:7]]
        # print(self.position)
        # print(self.rotation)
    

def ue_to_airsim(poses, filename='test'):
    airsim_poses = []
    for pose in poses:
        pose_tmp = [pose.name, 
                    pose.position[0], pose.position[1], -pose.position[2],
                    -pose.rotation[0], pose.rotation[1], pose.rotation[2]]
        airsim_poses.append(pose_tmp)
    df = pd.DataFrame(airsim_poses)
    df.to_csv(f'./pose_traces/{filename}_airsim.csv', header=['Name', 'X', 'Y', 'Z', 'Yaw', 'Pitch', 'Roll'], index=False)


def ue_to_miv(poses, filename='test'):
    airsim_poses = []
    for pose in poses:
        pose_tmp = [pose.name, 
                    pose.position[0], -pose.position[1], pose.position[2],
                    pose.rotation[0], -pose.rotation[1], pose.rotation[2]]
        airsim_poses.append(pose_tmp)
    df = pd.DataFrame(airsim_poses)
    df.to_csv(f'./pose_traces/{filename}_miv.csv', header=['Name', 'X', 'Y', 'Z', 'Yaw', 'Pitch', 'Roll'], index=False)


def import_cameras_pose(csvfile_PATH):
    cameras_pose = []
    with open(csvfile_PATH, 'r') as csv_f:
        rows = csv.reader(csv_f)
        next(rows) # skip header
        for row in rows:
            cameras_pose.append(Camera_pose(row))
    return cameras_pose

def main():
    for camera_position in camera_positions:
        for scene in scenes:
            poses_ue = import_cameras_pose(f'./pose_traces/{camera_position}_{scene}.csv')
            ue_to_airsim(poses_ue, f'{camera_position}_{scene}')
            ue_to_miv(poses_ue, f'{camera_position}_{scene}')


if __name__ == '__main__':
    main()
