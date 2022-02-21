import math
import os
import subprocess
import numpy as np
import pandas as pd

def three_dim_round(r=0):
    name_num = 0
    all_cameras = []
    # dim 1 => x,y
    for i in range(0,4):
        deg = i/4*360
        all_cameras.append([f'v{name_num}', -r*round(np.cos(np.deg2rad(deg)),2), r*round(np.sin(np.deg2rad(deg)),2), 0,deg,0,0])
        name_num = name_num + 1
    # dim 2 => y,z
    for i in range(0,4):
        deg = i/4*360
        all_cameras.append([f'v{name_num}', 0, r*round(np.cos(np.deg2rad(deg)),2), r*round(np.sin(np.deg2rad(deg)),2), 90, -deg, 90])
        name_num = name_num + 1
    # dim 3 => x,z
    for i in range(0,4):
        deg = i/4*360
        all_cameras.append([f'v{name_num}', -r*round(np.cos(np.deg2rad(deg)),2), 0, r*round(np.sin(np.deg2rad(deg)),2), 0, -deg, 90])
        name_num = name_num + 1
    return all_cameras


def three_round(r=0):
    name_num = 0
    all_cameras = []
    z_30 = r*round(np.sin(np.deg2rad(30)),2)
    for i in range(0,8):
        deg = i/8*360
        all_cameras.append([f'v{name_num}', -r*round(np.cos(np.deg2rad(deg)),2), r*round(np.sin(np.deg2rad(deg)),2), 0,-deg,0,0])
        name_num = name_num + 1
    # dim 1 => x,y
    for i in range(0,8):
        deg = i/8*360
        all_cameras.append([f'v{name_num}', -r*round(np.cos(np.deg2rad(deg)),2), r*round(np.sin(np.deg2rad(deg)),2), 0,-deg,0,0])
        name_num = name_num + 1

    return all_cameras


def main():
    all_cameras = three_dim_round(2)
    df = pd.DataFrame(all_cameras)
    df.to_csv('./pose_traces/round_test.csv', header=['Name', 'X', 'Y', 'Z', 'Yaw', 'Pitch', 'Roll'], index=False)

if __name__ == '__main__':
    main()
