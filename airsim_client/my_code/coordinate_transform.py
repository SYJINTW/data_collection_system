import os
import subprocess
import csv
import numpy as np
import pandas as pd
import multiprocessing as mp

# ============================================================
#        |  X|  Y|  Z| Yaw| Pitch| Roll|
# UE     |  a|  b|  c|   d|     e|    f|
# Airsim |  a|  b| -c|  -d|     e|    f|
# MIV    |  a| -b|  c|   d|    -e|    f|
# ============================================================

# Basic setting
# ============================================================
camera_positions = ['round']
scenes = ['bunny']
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
    df.to_csv(f'./pose_traces/{filename}/{filename}_airsim.csv', header=['Name', 'X', 'Y', 'Z', 'Yaw', 'Pitch', 'Roll'], index=False)


def ue_to_miv(poses, filename='test'):
    airsim_poses = []
    for pose in poses:
        pose_tmp = [pose.name, 
                    pose.position[0], -pose.position[1], pose.position[2],
                    pose.rotation[0], -pose.rotation[1], pose.rotation[2]]
        airsim_poses.append(pose_tmp)
    df = pd.DataFrame(airsim_poses)
    df.to_csv(f'./pose_traces/{filename}/{filename}_miv.csv', header=['Name', 'X', 'Y', 'Z', 'Yaw', 'Pitch', 'Roll'], index=False)

def seperate_pose(filename):
    df = pd.read_csv(f'./pose_traces/{filename}/{filename}_miv.csv')
    df = df.drop(columns=['Name'])
    all_cameras = df.values
    # print(all_cameras)
    for i in range(len(all_cameras)):
        tmp = []
        tmp.append(all_cameras[i])
        df_output = pd.DataFrame(tmp)
        df_output.to_csv(f'./pose_traces/{filename}/pose_{i}_miv.csv', header=['X', 'Y', 'Z', 'Yaw', 'Pitch', 'Roll'], index=False)

def import_cameras_pose(csvfile_PATH):
    cameras_pose = []
    with open(csvfile_PATH, 'r') as csv_f:
        rows = csv.reader(csv_f)
        next(rows) # skip header
        for row in rows:
            cameras_pose.append(Camera_pose(row))
    return cameras_pose

def change_main(filename):
    poses_ue = import_cameras_pose(f'./pose_traces/{filename}.csv')
    if not os.path.isdir(f'./pose_traces/{filename}'):
        os.makedirs(f'./pose_traces/{filename}')
    
    ue_to_airsim(poses_ue, filename)
    ue_to_miv(poses_ue, filename)
    seperate_pose(filename)

def main():
    for camera_position in camera_positions:
        for scene in scenes:
            filename = f'{camera_position}_{scene}'
            change_main(filename)
            # poses_ue = import_cameras_pose(f'./pose_traces/{filename}.csv')
            # if not os.path.isdir(f'./pose_traces/{filename}'):
            #     os.makedirs(f'./pose_traces/{filename}')
            # ue_to_airsim(poses_ue, filename)
            # ue_to_miv(poses_ue, filename)
            # seperate_pose(filename)


if __name__ == '__main__':
    main()
