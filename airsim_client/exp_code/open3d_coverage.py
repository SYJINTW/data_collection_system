from pathlib import Path
from re import S
import open3d as o3d
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
from itertools import combinations
import time
import CameraPose


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

def get_primitive_ids(scene, _eye: list, _center: list, _up: list, save_dir_PATH: Path, filename: str, MAX_NUM_TRI: int):
    
    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        fov_deg=FOV,
        eye=_eye,
        center=_center,
        up=_up,
        width_px=WIDTH,
        height_px=HEIGHT,
    )

    ans = scene.cast_rays(rays)
    ans_datas = ans['primitive_ids'].numpy()
    
    # save primitive_ids datas
    df = pd.DataFrame(ans_datas)
    df.to_csv(f'{save_dir_PATH}/{filename}_primitive_ids.csv', index=False)
    
    num_arr = np.zeros(MAX_NUM_TRI+1)
    unique, counts = np.unique(ans_datas, return_counts=True)
    for idx in range(unique.shape[0]):
        if unique[idx] == 4294967295:
            num_arr[0] = counts[idx]
        else:
            num_arr[unique[idx]] = counts[idx]

    df_bin = pd.DataFrame(num_arr)
    df_bin.to_csv(f'{save_dir_PATH}/{filename}_bin.csv', index=False)
    
    return num_arr

def get_coverage_data(scene, poses: list, save_dir_PATH: Path, filename: str, MAX_NUM_TRI: int):
    num_arr = []
    for pose_idx in range(len(poses)):
        filename_with_pose = f'{filename}_pose{pose_idx}'
        pose = poses[pose_idx]
        primitive_ids_data = get_primitive_ids(scene, pose.camera[0], pose.camera[1], pose.camera[2], save_dir_PATH, filename_with_pose, MAX_NUM_TRI)
        num_arr.append(primitive_ids_data)
    return num_arr

def coverage_table_generator(tv_bin_arrs, sv_bin_arrs, orders: int, save_dir_PATH: Path, filename: str):
    '''
    generate the coverage table by binary arrays to a csv_file
    '''

    header_arr = ['targetView', 'camSet', 'coverage']

    # find tv_num
    tv_num = len(tv_bin_arrs)
    
    # find sv_num
    sv_num = len(sv_bin_arrs)

    coverages = []
    coverages_dict_list = []
    # for order in range(1,orders+1):
    for tv_idx in range(tv_num):
        tv_bin = tv_bin_arrs[tv_idx]
        coverages_dict = {}
        for order in range(1,orders+1):
            # C(max_of_sv,order) -> choose 'order' of source views 
            combins = [c for c in combinations(range(sv_num), order)]
            for combin in combins:
                merge_bin = np.zeros(MAX_NUM_TRI+1)
                sv_bin = np.zeros(MAX_NUM_TRI+1)
                for sv_idx in combin:
                    tmp_sv_bin = sv_bin_arrs[sv_idx]
                    sv_bin = np.maximum(sv_bin, tmp_sv_bin)
                merge_bin = np.minimum(tv_bin, sv_bin)
                coverage = np.sum(merge_bin) / PIXEL
                coverages.append([tv_idx, combin, coverage])
                coverages_dict[combin] = coverage
        coverages_dict_list.append(coverages_dict)
    
    pd.DataFrame(coverages).to_csv(f'{save_dir_PATH}/{filename}_results.csv', header=header_arr, index=False)
    return coverages_dict_list

def computeQualityModel(workdir_PATH: Path, tv_poses: list, sv_poses: list, order: int, group_idx: int, scene, MAX_NUM_TRI: int):
    
    save_dir_PATH = Path(f'{workdir_PATH}/coverage_data')
    save_dir_PATH.mkdir(parents=True, exist_ok=True)

    # get tv and sv binary array
    tv_bin_arrs = get_coverage_data(scene, tv_poses, save_dir_PATH, f'out{group_idx}', MAX_NUM_TRI)
    sv_bin_arrs = get_coverage_data(scene, sv_poses, save_dir_PATH, f'in{group_idx}', MAX_NUM_TRI)
    
    save_dir_PATH = Path(f'{workdir_PATH}/coverage_table')
    save_dir_PATH.mkdir(parents=True, exist_ok=True)
    
    return coverage_table_generator(tv_bin_arrs, sv_bin_arrs, order, save_dir_PATH, f'in_to_out{group_idx}')

def main():
    print("coverage")
                        
if __name__ == '__main__':
    main()
    
