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

import coordinate_transform

# ============================================================

OPEN3D_RENDER = True
COVERAGE_TABLE = True

# ============================================================

FOV = 90 # degree
WIDTH = 1280 # pixel
HEIGHT = 720 # pixel
PIXEL = WIDTH*HEIGHT
OBJ_PATH = './obj_source/bunny_mesh_merge.OBJ'
MAX_NUM_TRI = 8754 # get from obj file
VIEW_START_POINT = [0,0,5] # meters
ORDER = 3

# ============================================================

tv_filename = 'tv_bunny'
sv_filename = 'sv_bunny'

# ============================================================

def get_primitive_ids(mesh_obj_path, _eye, _center, _up, filename, pose_filename):
    mesh = o3d.io.read_triangle_mesh(mesh_obj_path)
    # print(mesh)
    # print(np.asarray(mesh.vertices))
    # print(np.asarray(mesh.triangles))
    # print(type(mesh))
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
    ans_datas = ans['primitive_ids'].numpy()
    
    df = pd.DataFrame(ans_datas)
    df.to_csv(f'./coverage_data/{filename}/{pose_filename}_primitive_ids.csv', index=False)
    
    num_arr = np.zeros(MAX_NUM_TRI+1)
    unique, counts = np.unique(ans_datas, return_counts=True)
    for idx in range(unique.shape[0]):
        if unique[idx] == 4294967295:
            num_arr[0] = counts[idx]
        else:
            num_arr[unique[idx]] = counts[idx]

    df_bin = pd.DataFrame(num_arr)
    df_bin.to_csv(f'./coverage_data/{filename}/{pose_filename}_bin.csv', index=False)

def cal_coverage_by_np(base_view, src_view):
    base_num_arr = np.zeros(MAX_NUM_TRI+1)
    base_unique, base_counts = np.unique(base_view, return_counts=True)
    for idx in range(base_unique.shape[0]):
        if base_unique[idx] == 4294967295:
            base_num_arr[0] = base_counts[idx]
        else:
            base_num_arr[base_unique[idx]] = base_counts[idx]
    # print('Num of unknown: ', base_num_arr[0])
    
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
    if not os.path.isdir('./coverage_data'):
        os.makedirs('./coverage_data')
    if not os.path.isdir(f'./coverage_data/{filename}'):
        os.makedirs(f'./coverage_data/{filename}')
    airsim_poses = pd.read_csv(f'./pose_traces/{filename}/{filename}_airsim.csv').to_numpy()
    
    for idx, airsim_pose in enumerate(airsim_poses):
        name = f'{filename}_{idx}'
        eye, center, up = euler_to_camera(airsim_pose[1]*100-VIEW_START_POINT[0]*100,
                                            airsim_pose[2]*100-VIEW_START_POINT[1]*100,
                                            airsim_pose[3]*100-+VIEW_START_POINT[2]*100,
                                            airsim_pose[4],
                                            airsim_pose[5],
                                            airsim_pose[6]
                                            )
        get_primitive_ids(mesh_obj_path, eye, center, up, filename, name)

def coverage_table_generator(tv_filename, sv_filename, orders):
    '''
    generate the coverage table to a csv_file

    output_filename:
    {tv_filename}_{sv_filename}_merge.csv
    
    foramt:
    [targetView,camSet,coverage]
    '''
    header_arr = ['targetView', 'camSet', 'coverage']

    if not os.path.isdir(f'./coverage_results'):
        os.makedirs(f'./coverage_results')
    if not os.path.isdir(f'./coverage_results/{tv_filename}_to_{sv_filename}'):
        os.makedirs(f'./coverage_results/{tv_filename}_to_{sv_filename}')

    # find tv_num
    tv_poses = pd.read_csv(f'./pose_traces/{tv_filename}/{tv_filename}_airsim.csv')
    tv_num = tv_poses.shape[0]
    
    # find sv_num
    sv_poses = pd.read_csv(f'./pose_traces/{sv_filename}/{sv_filename}_airsim.csv')
    sv_num = sv_poses.shape[0]

    coverages = []
    for order in range(1,orders+1):
        for tv_idx in range(tv_num):
            tv_bin = pd.read_csv(f'./coverage_data/{tv_filename}/{tv_filename}_{tv_idx}_bin.csv').to_numpy().flatten()
            # C(max_of_sv,order) -> choose 'order' of source views 
            combins = [c for c in combinations(range(sv_num), order)]
            for combin in combins:
                # cam_key = '('
                # for element in combin:
                #     cam_key = cam_key + f'{element},'
                # cam_key = cam_key + ')'
                
                merge_bin = np.zeros(MAX_NUM_TRI+1)
                sv_bin = np.zeros(MAX_NUM_TRI+1)
                for sv_idx in combin:
                    tmp_sv_bin = pd.read_csv(f'./coverage_data/{sv_filename}/{sv_filename}_{sv_idx}_bin.csv').to_numpy().flatten()
                    sv_bin = np.maximum(sv_bin, tmp_sv_bin)
                merge_bin = np.minimum(tv_bin, tmp_sv_bin)
                coverage = np.sum(merge_bin) / PIXEL
                coverages.append([tv_idx, combin, coverage])
    
    pd.DataFrame(coverages).to_csv(f'./coverage_results/{tv_filename}_to_{sv_filename}/{tv_filename}_to_{sv_filename}_results.csv', header=header_arr, index=False)
    return coverages

def computeQualityModel(tv_filename: str, sv_filename: str, order: int):

    if OPEN3D_RENDER:
        if not os.path.isdir('./coverage_data'):
            os.makedirs('./coverage_data')
        
        coordinate_transform.change_main(tv_filename)
        get_coverage_data(OBJ_PATH, tv_filename)

        coordinate_transform.change_main(sv_filename)
        get_coverage_data(OBJ_PATH, sv_filename)


    if COVERAGE_TABLE:
        return coverage_table_generator(tv_filename, sv_filename, order)
    
def qualityLoader(file_path: str, kmax: int =None, model: str ='coverage') -> list:
    '''
    file_path: path to a csv with format
        viewport,key,coverage,quaility1, qulaity2, ...
        int,'tuple',float,float,float
    kmax: only extract camSet with the number of its elements <= kmax
        None means all

    return a list of dict 
        [{
            (0): 0.5,
            (1): 0.6,
            (2): 0.7,
            (0, 1): 0.8,
            (0, 2): 0.9,
            (1, 2): 0.95,
            (0, 1, 2): 1.0
        }, ...]
    '''
    df = pd.read_csv(file_path)
    df = df.reset_index()
    coverage_dict_list = []
    currViewport = -1
    coverage_dict = {}
    for index, row in df.iterrows():
        targetView = row['targetView']
        if targetView != currViewport:
            if currViewport != -1:
                coverage_dict_list.append(coverage_dict)
            coverage_dict = {}
            currViewport = targetView
        camSet = eval(row['camSet'])
        if kmax == None or len(camSet) <= kmax:
            coverage_dict[camSet] = row[model]
    coverage_dict_list.append(coverage_dict)
    return coverage_dict_list

def main():
    start_time = time.time()
    coverage_table = computeQualityModel(tv_filename, sv_filename, ORDER)
    end_time = time.time()
    print(end_time-start_time)
                        
if __name__ == '__main__':
    main()
    
