from email import header
from re import S
import open3d as o3d
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
from itertools import combinations
import time
from tomlkit import string
import multiprocessing as mp
from functools import partial

import coordinate_transform

# ============================================================

OPEN3D_RENDER = True
COVERAGE_TABLE = True

# ============================================================

FOV = 90 # degree
WIDTH = 1280 # pixel
HEIGHT = 720 # pixel
PIXEL = WIDTH * HEIGHT
OBJ_PATH = './obj_source/bunny_mesh_merge.OBJ'
MAX_NUM_TRI = 8754 # get from obj file
VIEW_START_POINT = [0,0,5] # meters
ORDER = 3

# ============================================================

tv_filename = 'tv_bunny'
sv_filename = 'sv_bunny'

# ============================================================

def get_primitive_ids(x,y,z,yaw,pitch,roll, mesh_obj_path,filename, pose_filename):
    
    shift_xyz = [x,y,z]
    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
    rotMat = r.as_matrix()
    
    _eye = np.round_(np.array([shift_xyz[1],-shift_xyz[2],-shift_xyz[0]]), decimals=2)
    
    center_arr = np.dot(rotMat, np.array([1,0,0]).T)
    _center = np.round_(np.array(_eye + [center_arr[1],-center_arr[2],-center_arr[0]]), decimals=2)
    
    up_arr = np.dot(rotMat, np.array([0,0,1]).T)
    _up = np.round_([up_arr[1],-up_arr[2],-up_arr[0]], decimals=2)


    mesh = o3d.io.read_triangle_mesh(mesh_obj_path)
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)

    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        fov_deg=FOV,
        eye=_eye,
        center=_center,
        up=_up,
        width_px=WIDTH,
        height_px=HEIGHT,
    )

    ans = scene.cast_rays(rays)

    num_arr = np.zeros(MAX_NUM_TRI+1)
    unique, counts = np.unique(ans['primitive_ids'].numpy(), return_counts=True)
    for idx in range(unique.shape[0]):
        if unique[idx] == 4294967295:
            num_arr[0] = counts[idx]
        else:
            num_arr[unique[idx]] = counts[idx]

    num_arr = num_arr.flatten()
    return num_arr

def cal_coverage_by_np(base_view, src_view):
    base_num_arr = np.zeros(MAX_NUM_TRI+1)
    base_unique, base_counts = np.unique(base_view, return_counts=True)
    for idx in range(base_unique.shape[0]):
        if base_unique[idx] == 4294967295:
            base_num_arr[0] = base_counts[idx]
        else:
            base_num_arr[base_unique[idx]] = base_counts[idx]
    
    src_num_arr = np.zeros(MAX_NUM_TRI+1)
    unique, counts = np.unique(src_view, return_counts=True)
    for idx in range(unique.shape[0]):
        if unique[idx] == 4294967295:
            src_num_arr[0] = counts[idx]
        else:
            src_num_arr[unique[idx]] = counts[idx]

    merge_min = np.minimum(base_num_arr, src_num_arr)
    coverage = np.sum(merge_min) / (WIDTH*HEIGHT)
    return coverage

def euler_to_camera(x,y,z,yaw,pitch,roll):
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

def get_coverage_data(mesh_obj_path, filename):
    ue_poses = pd.read_csv(f'./pose_traces/{filename}.csv').to_numpy()
    
    # maybe can add pool
    bin_arr = []
    for idx, ue_pose in enumerate(ue_poses):
        name = f'{filename}_{idx}'
        bin_arr.append(get_primitive_ids(ue_pose[1]*100-VIEW_START_POINT[0]*100,
                        ue_pose[2]*100-VIEW_START_POINT[1]*100,
                        (-ue_pose[3])*100-VIEW_START_POINT[2]*100,
                        (-ue_pose[4]),
                        ue_pose[5],
                        ue_pose[6],
                        mesh_obj_path, filename, name
                        ))
    return bin_arr

def coverage_table_generator(tv_filename, sv_filename, tv_bin_arr, sv_bin_arr, orders):
    '''
    generate the coverage table to a csv_file

    output_filename:
    {tv_filename}_{sv_filename}_merge.csv
    
    foramt:
    [targetView,camSet,coverage]
    '''
    # find tv_num
    tv_bin_arr = np.array(tv_bin_arr)
    tv_num = tv_bin_arr.shape[0]
    
    # find sv_num
    sv_bin_arr = np.array(sv_bin_arr)
    sv_num = sv_bin_arr.shape[0]

    coverages_dict_list = []
    # for order in range(1,orders+1):
    for tv_idx in range(tv_num):
        coverages_dict = {}
        for order in range(1,orders+1):
            # C(max_of_sv,order) -> choose 'order' of source views 
            combins = [c for c in combinations(range(sv_num), order)]
            for combin in combins:
                merge_bin = np.zeros(MAX_NUM_TRI+1)
                sv_bin = np.zeros(MAX_NUM_TRI+1)
                for sv_idx in combin:
                    sv_bin = np.maximum(sv_bin, sv_bin_arr[sv_idx])
                merge_bin = np.minimum(tv_bin_arr[tv_idx], sv_bin)
                coverage = np.sum(merge_bin) / PIXEL
                coverages_dict[combin] = coverage
        coverages_dict_list.append(coverages_dict)
    
    # pd.DataFrame(coverages).to_csv(f'./coverage_results/{tv_filename}_to_{sv_filename}/{tv_filename}_to_{sv_filename}_results.csv', header=header_arr, index=False)
    return coverages_dict_list

def computeQualityModel(tv_filename: str, sv_filename: str, order: int):

    tv_bin_arr = get_coverage_data(OBJ_PATH, tv_filename)
    sv_bin_arr = get_coverage_data(OBJ_PATH, sv_filename)

    return coverage_table_generator(tv_filename, sv_filename, tv_bin_arr, sv_bin_arr, order)
    
def main():
    start_time = time.time()
    coverages_dict_list = computeQualityModel(tv_filename, sv_filename, ORDER)
    end_time = time.time()
    print(end_time-start_time)
    pd.DataFrame(coverages_dict_list).to_csv('test.csv',index=False)
                        
if __name__ == '__main__':
    main()
    
