import os
from pathlib import Path
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
    coordinate = ""
    
    def __init__(self, _array=['',0,0,0,0,0,0],_coordinate=""):
        self.name = _array[0]
        self.position = [float(x) for x in _array[1:4]]
        self.rotation = [float(x) for x in _array[4:7]]
        self.coordinate = _coordinate
        # print(self.position)
        # print(self.rotation)
    

def ue_to_airsim(poses: list, filedir: Path, filename: str):
    airsim_poses = []
    for pose in poses:
        pose_tmp = [pose.name, 
                    pose.position[0], pose.position[1], -pose.position[2],
                    -pose.rotation[0], pose.rotation[1], pose.rotation[2]]
        airsim_poses.append(pose_tmp)
    df = pd.DataFrame(airsim_poses)
    df.to_csv(f'{filedir}/{filename}.csv', header=['Name', 'X', 'Y', 'Z', 'Yaw', 'Pitch', 'Roll'], index=False)
    return airsim_poses

def ue_to_miv(poses: list, filedir: Path, filename: str):
    miv_poses = []
    for pose in poses:
        pose_tmp = [pose.name, 
                    pose.position[0], -pose.position[1], pose.position[2],
                    pose.rotation[0], -pose.rotation[1], pose.rotation[2]]
        miv_poses.append(pose_tmp)
    df = pd.DataFrame(miv_poses)
    df.to_csv(f'{filedir}/{filename}.csv', header=['Name', 'X', 'Y', 'Z', 'Yaw', 'Pitch', 'Roll'], index=False)
    return miv_poses

def seperate_pose(filedir: Path, filename: str):
    df = pd.read_csv(f'{filedir}/{filename}_miv.csv')
    df = df.drop(columns=['Name'])
    all_cameras = df.values
    # print(all_cameras)
    for i in range(len(all_cameras)):
        tmp = []
        tmp.append(all_cameras[i])
        df_output = pd.DataFrame(tmp)
        df_output.to_csv(f'{filedir}/pose_{i}_miv.csv', header=['X', 'Y', 'Z', 'Yaw', 'Pitch', 'Roll'], index=False)

def import_cameras_pose(csvfile_PATH):
    cameras_pose = []
    with open(csvfile_PATH, 'r') as csv_f:
        rows = csv.reader(csv_f)
        next(rows) # skip header
        for row in rows:
            cameras_pose.append(Camera_pose(row,'ue'))
    return cameras_pose

def import_camera_pose_arr(cam_arr: list):
    '''
    the cam_arr should be like [Name,X,Y,Z,Yaw,Pitch,Row] and in ue coordinate
    '''
    cameras_pose = []
    for row in cam_arr:
        cameras_pose.append(Camera_pose(row,'ue'))
    return cameras_pose

def generate_all_coordinate_data(pose_arr: list, filedir: Path, filename: str):
    print('generate_all_coordinate_data')
    poses_ue = import_camera_pose_arr(pose_arr)
    airsim_poses = ue_to_airsim(poses_ue, filedir, f'{filename}_airsim')
    miv_poses = ue_to_miv(poses_ue, filedir, f'{filename}_miv')
    seperate_pose(filedir, f'{filename}')
    return airsim_poses, miv_poses

def main():
    for camera_position in camera_positions:
        for scene in scenes:
            filename = f'{camera_position}_{scene}'
            # change_main(filename)
            # poses_ue = import_cameras_pose(f'./pose_traces/{filename}.csv')
            # if not os.path.isdir(f'./pose_traces/{filename}'):
            #     os.makedirs(f'./pose_traces/{filename}')
            # ue_to_airsim(poses_ue, filename)
            # ue_to_miv(poses_ue, filename)
            # seperate_pose(filename)


if __name__ == '__main__':
    main()
