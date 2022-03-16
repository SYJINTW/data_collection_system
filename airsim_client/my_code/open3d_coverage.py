from email import header
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
from tomlkit import string


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

class Camera_pose:
    name = ""
    ue = [0,0,0,0,0,0] # X, Y, Z, Yaw, Pitch, Roll -> meters and degrees
    airsim = [0,0,0,0,0,0] # X, Y, Z, Yaw, Pitch, Roll -> meters and degrees
    miv = [0,0,0,0,0,0] # X, Y, Z, Yaw, Pitch, Roll -> meters and degrees
    camera = [
            [0,0,0], # eye
            [0,0,0], # center
            [0,0,0]  # up
            ]

    position = [0, 0, 0] # X, Y, Z
    rotation = [0, 0, 0] # Yaw, Pitch, Roll
     
    def __init__(self, _name, _array=[0,0,0,0,0,0],_cam_starting_point=[0,0,0],_coordinate = 'ue'):
        self.name = _array[0]
        _array = [float(i) for i in _array]
        _cam_starting_point = [float(i) for i in _cam_starting_point]
        if _coordinate == 'ue':
            self.ue = _array
            self.airsim = self.ue_to_airsim(self.ue)
            self.miv = self.ue_to_miv(self.ue)
            self.camera = self.airsim_to_camera(
                                                self.airsim[0],
                                                self.airsim[1],
                                                self.airsim[2],
                                                self.airsim[3],
                                                self.airsim[4],
                                                self.airsim[5],
                                                _cam_starting_point
                                                )
        elif _coordinate == 'airsim':
            self.airsim = _array
            self.ue = self.airsim_to_ue(self.airsim)
            self.miv = self.ue_to_miv(self.ue)
            self.camera = self.airsim_to_camera(
                                                self.airsim[0]*100-_cam_starting_point[0]*100,
                                                self.airsim[1]*100-_cam_starting_point[1]*100,
                                                self.airsim[2]*100-_cam_starting_point[2]*100,
                                                self.airsim[3],
                                                self.airsim[4],
                                                self.airsim[5]
                                                )
        elif _coordinate == 'miv':
            print('Not yet')
        else:
            print('error')
    

    def ue_to_airsim(self, ue_arr: list)->list:
        airsim_arr = [
                    ue_arr[0],
                    ue_arr[1],
                    -ue_arr[2],
                    -ue_arr[3],
                    ue_arr[4],
                    ue_arr[5],
                    ]
        return airsim_arr

    def airsim_to_ue(self, airsim_arr: list)->list:
        ue_arr = [
                    airsim_arr[0],
                    airsim_arr[1],
                    -airsim_arr[2],
                    -airsim_arr[3],
                    airsim_arr[4],
                    airsim_arr[5],
                    ]
        return ue_arr

    def ue_to_miv(self, ue_arr: list):
        miv_arr = [
                    ue_arr[0], 
                    -ue_arr[1], 
                    ue_arr[2],
                    ue_arr[3], 
                    -ue_arr[4],
                    ue_arr[5]
                    ]
        return miv_arr

    def airsim_to_camera(self, x: float, y: float, z: float, yaw: float, pitch: float, roll: float)->list:

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

        return [eye,center,up]

# ============================================================

def get_primitive_ids(scene, _eye, _center, _up, dir_name, filename, pose_idx):
    save_dir = f'./raw_data/{dir_name}/coverage_data/{filename}'
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
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
    df.to_csv(f'{save_dir}/{pose_idx}_primitive_ids.csv', index=False)
    
    num_arr = np.zeros(MAX_NUM_TRI+1)
    unique, counts = np.unique(ans_datas, return_counts=True)
    for idx in range(unique.shape[0]):
        if unique[idx] == 4294967295:
            num_arr[0] = counts[idx]
        else:
            num_arr[unique[idx]] = counts[idx]

    df_bin = pd.DataFrame(num_arr)
    df_bin.to_csv(f'{save_dir}/{pose_idx}_bin.csv', index=False)
    
    return num_arr

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

def get_coverage_data(scene, tv_poses, dir_name, filename):
    num_arr = []
    for pose_idx in range(len(tv_poses)):
        tv_pose = tv_poses[pose_idx]
        primitive_ids_data = get_primitive_ids(scene, tv_pose.camera[0], tv_pose.camera[1], tv_pose.camera[2], dir_name, filename, pose_idx)
        num_arr.append(primitive_ids_data)
    return num_arr

def coverage_table_generator(tv_bin_arrs, sv_bin_arrs, orders, dir_name, filename):
    '''
    generate the coverage table by binary arrays to a csv_file

    output_filename:
    {tv_filename}_{sv_filename}_merge.csv
    
    foramt:
    [targetView,camSet,coverage]
    '''
    save_dir = f'./raw_data/{dir_name}/coverage_results'
    Path(save_dir).mkdir(parents=True, exist_ok=True)
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
    
    pd.DataFrame(coverages).to_csv(f'{save_dir}/{filename}_results.csv', header=header_arr, index=False)
    return coverages_dict_list

def computeQualityModel(objfile_PATH: Path, tv_poses: Camera_pose, sv_poses: Camera_pose, order: int, dir_name: str, group_idx: int):
    # generate scene
    mesh = o3d.io.read_triangle_mesh(objfile_PATH)
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)
    
    # get tv and sv binary array
    tv_bin_arrs = get_coverage_data(scene, tv_poses, dir_name, f'{dir_name}_out{group_idx}')
    sv_bin_arrs = get_coverage_data(scene, sv_poses, dir_name, f'{dir_name}_in{group_idx}')
    return coverage_table_generator(tv_bin_arrs, sv_bin_arrs, order, dir_name, f'{dir_name}_in{group_idx}_to_{dir_name}_out{group_idx}')
   
def qualityLoader(file_path: str, kmax: int = None, model: str ='coverage') -> list:
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
    # objfile_PATH = './obj_source/bunny_mesh_merge.OBJ'
    # start_time = time.time()
    # computeQualityModel(objfile_PATH, tv_poses, sv_poses, order: int, filename)
    # end_time = time.time()
    # print(end_time-start_time)
    print("coverage")
                        
if __name__ == '__main__':
    main()
    
