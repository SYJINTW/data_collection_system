import pandas as pd
import numpy as np
import os

camera_positions = ['round']
scenes = ['test']

def seperate_pose(filename):
    df = pd.read_csv(f'./pose_traces/{filename}_miv.csv')
    df = df.drop(columns=['Name'])
    all_cameras = df.values
    print(all_cameras)
    for i in range(len(all_cameras)):
        tmp = []
        tmp.append(all_cameras[i])
        df_output = pd.DataFrame(tmp)
        df_output.to_csv(f'./pose_traces/{filename}/pose_{i}_miv.csv', header=['X', 'Y', 'Z', 'Yaw', 'Pitch', 'Roll'], index=False)

def main():
    for camera_position in camera_positions:
        for scene in scenes:
            filename = f'{camera_position}_{scene}'
            seperate_pose(filename)

if __name__ == '__main__':
    main()