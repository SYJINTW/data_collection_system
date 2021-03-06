from pathlib import Path
import open3d as o3d
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import csv
from scipy.spatial.transform import Rotation as R
from itertools import combinations
import time

import open3d_coverage
import Camera_pose

# from airsim_client.my_code.coordinate_transform import ue_to_airsim


# ============================================================

VIEW_START_POINT = [0,0,5] # meters
MAX_NUM_TRI = 8754 # get from obj file

# ============================================================

def import_cameras_pose(csvfile_PATH):
    cameras_pose = []
    with open(csvfile_PATH, 'r') as csv_f:
        rows = csv.reader(csv_f) # [t,x,y,z,roll,pitch,yaw]
        next(rows) # skip header
        for row in rows:
            cameras_pose.append(Camera_pose.Camera_pose('',[row[1],row[2],row[3],row[6],row[5],row[4]],VIEW_START_POINT,'airsim'))
    return cameras_pose

def airsim_to_camera(x,y,z,yaw,pitch,roll):
    '''
    in Airsim (front right down)
    '''
    # euler to vector
    eye = []
    center = []
    up = []

    shift_xyz = [x,y,z]
    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
    rotMat = r.as_matrix()
    
    eye = np.round_(np.array([shift_xyz[1],-shift_xyz[2],-shift_xyz[0]]), decimals=2)
    
    center_arr = np.dot(rotMat, np.array([1,0,0]).T)
    center = np.round_(np.array(eye + [center_arr[1],-center_arr[2],-center_arr[0]]), decimals=2)
    
    up_arr = np.dot(rotMat, np.array([0,0,1]).T)
    up = np.round_([up_arr[1],-up_arr[2],-up_arr[0]], decimals=2)

    return eye, center, up

def get_primitive_ids(scene, _eye, _center, _up):

    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        fov_deg=90,
        eye=_eye,
        center=_center,
        up=_up,
        width_px=1280,
        height_px=720,
    )

    viewport = scene.cast_rays(rays)
    viewport_datas = viewport['primitive_ids'].numpy()
    
    bin_arr = np.zeros(MAX_NUM_TRI+1)
    unique, counts = np.unique(viewport_datas, return_counts=True)
    for idx in range(unique.shape[0]):
        if unique[idx] == 4294967295:
            bin_arr[0] = counts[idx]
        else:
            bin_arr[unique[idx]] = counts[idx]
    
    return bin_arr

def choose_pose_traces(csvfile_PATH: Path, objfile_PATH: Path, downsample_num: int, threshold_coverage: float, num_for_group: int, dir_name: str):
    '''
    Args:
        csvfile_PATH
        objfile_PATH
        downsample_num
        max_coverage
        num_for_group
        save_filename

    Returns:
        the numpy array of pose traces [x,y,z,yaw,pitch,roll]
    '''

    # create necessary dir
    Path(f'./raw_data/{dir_name}/pose_traces').mkdir(parents=True, exist_ok=True) 

    # load pose_traces 
    camera_poses = import_cameras_pose(csvfile_PATH)
    
    # load obj scene
    mesh = o3d.io.read_triangle_mesh(objfile_PATH)
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)

    output_camera_poses = []
    output_poses = []
    ref_bin = []
    idx = 0
    group_idx = 0

    for camera_pose_idx in np.arange(0,len(camera_poses),downsample_num):
        camera_pose = camera_poses[camera_pose_idx]
        bin_arr = get_primitive_ids(scene, camera_pose.camera[0], camera_pose.camera[1], camera_pose.camera[2])

        if idx == 0 and group_idx == 0: # start the first group
            
            camera_pose.name = f'v{idx}'
            output_camera_poses.append(camera_pose)
            
            output_poses.append([
                                f'v{idx}',
                                camera_pose.ue[0],
                                camera_pose.ue[1],
                                camera_pose.ue[2],
                                camera_pose.ue[3],
                                camera_pose.ue[4],
                                camera_pose.ue[5],
                                ])
            ref_bin = bin_arr
            idx = idx + 1
        else:
            merge_bin = np.minimum(ref_bin, bin_arr)
            coverage = np.sum(merge_bin) / (1280*720)
            # print(coverage)
            if coverage <= threshold_coverage: # append this pose trace and set it as the new ref
                
                camera_pose.name = f'v{idx}'
                output_camera_poses.append(camera_pose)

                output_poses.append([
                                    f'v{idx}',
                                    camera_pose.ue[0],
                                    camera_pose.ue[1],
                                    camera_pose.ue[2],
                                    camera_pose.ue[3],
                                    camera_pose.ue[4],
                                    camera_pose.ue[5],
                                    ])
                ref_bin = bin_arr
                idx = idx + 1

                if idx == num_for_group: # the end of this group, use the last ref_bin as the new ref_bin
                    # print(f'Group: {group_idx}')
                    # print(output_poses)
                    df = pd.DataFrame(output_poses)
                    # save pose trace to csv file
                    df.to_csv(f'./raw_data/{dir_name}/pose_traces/{dir_name}_in{group_idx}.csv', header=['Name','X','Y','Z','Yaw','Pitch','Row'],index=False)
                    df.to_csv(f'./raw_data/{dir_name}/pose_traces/{dir_name}_out{group_idx}.csv', header=['Name','X','Y','Z','Yaw','Pitch','Row'],index=False)
                    yield group_idx, output_camera_poses
                    
                    output_camera_poses = []
                    output_poses = []
                    idx = 0
                    group_idx = group_idx + 1

def main():
    objfile_PATH = './obj_source/bunny_mesh_merge.OBJ'
    mesh = o3d.io.read_triangle_mesh(objfile_PATH)
    print(np.asarray(mesh.triangles).shape[0])
    # print("coordinate_transform")

if __name__ == '__main__':
    main()
    


