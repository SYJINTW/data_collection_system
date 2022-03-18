import numpy as np
import pandas as pd
import csv
from pathlib import Path

# ============================================================

import CameraPose

# ============================================================

VIEW_START_POINT = [0,0,5] # meters

# ============================================================

def import_cameras_pose(csvfile_PATH):
    cameras_pose = []
    with open(csvfile_PATH, 'r') as csv_f:
        rows = csv.reader(csv_f) # [t,x,y,z,roll,pitch,yaw]
        next(rows) # skip header
        for row in rows:
            cameras_pose.append(CameraPose.CameraPose('',[row[1],row[2],row[3],row[6],row[5],row[4]],VIEW_START_POINT,'airsim'))
    return cameras_pose



def read_csv_to_CameraPose(csvfile_PATH: Path):
    '''

    '''





def split_poses(csvfile_PATH_list: list, num_of_group: int, num_of_frame: int, downsample_num: int):
    '''
    Args:
        csvfile_PATH_list: a list of csvfile_PATH (list of str)
        num_of_group: the number of groups should split
        num_of_frame: the number of frames in each group
    
    '''
    total_frames = num_of_group*num_of_frame

    all_poses = []
    for csvfile_PATH in csvfile_PATH_list:
        poses = import_cameras_pose(csvfile_PATH)[:total_frames]
        all_poses.append(poses)

    print('Number of poses:', len(all_poses))
    for group_idx in range(len(all_poses)):
        print(f'Number of pose traces in Group {group_idx}: {len(all_poses[group_idx])}')
        
def main():
    csvfile_PATH_list = [
                        './raw_poses/pose0.csv',
                        './raw_poses/pose1.csv',
                        './raw_poses/pose2.csv',
                        './raw_poses/pose3.csv',
                        './raw_poses/pose4.csv'
                        ]
    num_of_group = 2
    num_of_frame = 30
    downsample_num = 5

    split_poses(csvfile_PATH_list, num_of_group, num_of_frame, downsample_num)

if __name__ == '__main__':
    main()  

