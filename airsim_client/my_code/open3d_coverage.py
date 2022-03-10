from re import S
import open3d as o3d
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
from itertools import combinations

# ============================================================

OPEN3D_RENDER = False
CAL_COVERAGE = False
COVERAGE_TABLE = True

# ============================================================

FOV = 90 # degree
WIDTH = 1280 # pixel
HEIGHT = 720 # pixel
OBJ_PATH = './obj_source/bunny_mesh_merge.OBJ'
MAX_NUM_TRI = 8754 # get from obj file
VIEW_START_POINT = [0,0,5] # meters
ORDER = 3

# ============================================================

sv_camera_positions = ['round']
sv_scenes = ['bunny']
tv_camera_positions = ['round']
tv_scenes = ['bunny']

# ============================================================

def get_primitive_ids(mesh_obj_path, _eye, _center, _up, filename, pose_filename):
    mesh = o3d.io.read_triangle_mesh(mesh_obj_path)
    # print(mesh)
    # print(np.asarray(mesh.vertices))
    print(np.asarray(mesh.triangles))
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
    print('Num of unknown: ', base_num_arr[0])
    
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
        os.system('mkdir ./coverage_data')
    if not os.path.isdir(f'./coverage_data/{filename}'):
        os.system(f'mkdir ./coverage_data/{filename}')
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

def coverage_table_generator(order, tv_filename, sv_filename):
    '''
    generate the coverage table to a csv_file

    output_filename:
    {tv_filename}_{sv_filename}_merge.csv
    
    foramt:
    [targetView,camSet,coverage]
    '''
    # find tv_num
    tv_poses = pd.read_csv(f'./pose_traces/{tv_filename}/{tv_filename}_airsim.csv')
    tv_num = tv_poses.shape[0]
    print(tv_num)
    
    # find sv_num
    sv_poses = pd.read_csv(f'./pose_traces/{sv_filename}/{sv_filename}_airsim.csv')
    sv_num = sv_poses.shape[0]
    print(sv_num)

    for tv_idx in range(tv_num):
        tv_bin = pd.read_csv(f'./coverage_data/{tv_filename}/{tv_filename}_{tv_idx}_bin.csv').to_numpy().flatten()
        # C(max_of_sv,order) -> choose 'order' of source views 
        combins = [c for c in combinations(range(sv_num+1), order)]
        for combin in combins:
            print(combin)
            sv_bin = np.zeros(MAX_NUM_TRI+1)
            for sv_idx in combin:
                tmp_sv_bin = pd.read_csv(f'./coverage_data/{sv_filename}/{sv_filename}_{sv_idx}_bin.csv').to_numpy().flatten()
                sv_bin = np.maximum(sv_bin, tmp_sv_bin)
            

def main():

    mesh_obj_path = OBJ_PATH
    
    if not os.path.isdir('./coverage_data'):
        os.system('mkdir ./coverage_data')
    
    # pre do
    if OPEN3D_RENDER:
        for tv_camera_position in tv_camera_positions:
            for tv_scene in tv_scenes:
                tv_filename = f'{tv_camera_position}_{tv_scene}'
                get_coverage_data(OBJ_PATH, tv_filename)

        for sv_camera_position in sv_camera_positions:
            for sv_scene in sv_scenes:
                sv_filename = f'{sv_camera_position}_{sv_scene}'
                get_coverage_data(OBJ_PATH, sv_filename)

    if CAL_COVERAGE:
        if not os.path.isdir('./coverage_results'):
            os.system('mkdir ./coverage_results')
        for tv_camera_position in tv_camera_positions:
            for tv_scene in tv_scenes:
                tv_filename = f'{tv_camera_position}_{tv_scene}'
                tv_poses = pd.read_csv(f'./pose_traces/{tv_filename}/{tv_filename}_airsim.csv').to_numpy()
                for sv_camera_position in sv_camera_positions:
                    for sv_scene in sv_scenes:
                        sv_filename = f'{sv_camera_position}_{sv_scene}'
                        if not os.path.isdir(f'./coverage_results/{tv_filename}_to_{sv_filename}'):
                            os.system(f'mkdir ./coverage_results/{tv_filename}_to_{sv_filename}')
                        coverages = []
                        for tv_pose in tv_poses:
                            # tv_pose = [Name,X,Y,Z,Yaw,Pitch,Roll]
                            base_view = pd.read_csv(f'./coverage_data/{tv_filename}/{tv_filename}_{tv_pose[0]}_primitive_ids.csv').to_numpy()
                            
                            sv_poses = pd.read_csv(f'./pose_traces/{sv_filename}.csv').to_numpy()
                            for sv_pose in sv_poses:
                                # sv_pose = [Name,X,Y,Z,Yaw,Pitch,Roll]
                                src_view = pd.read_csv(f'./coverage_data/{sv_filename}/{sv_filename}_{sv_pose[0]}_primitive_ids.csv').to_numpy()
                                
                                # calculate the coverage of target view by source view
                                coverage = cal_coverage_by_np(base_view, src_view)
                                coverages.append([tv_pose[0],sv_pose[0],coverage])
                        df = pd.DataFrame(coverages)
                        df.to_csv(f'./coverage_results/{tv_filename}_to_{sv_filename}/{tv_filename}_to_{sv_filename}_basic.csv', header=['tv','sv','coverage'],index=False)
    
    if COVERAGE_TABLE:
        for tv_camera_position in tv_camera_positions:
            for tv_scene in tv_scenes:
                tv_filename = f'{tv_camera_position}_{tv_scene}'
                for sv_camera_position in sv_camera_positions:
                    for sv_scene in sv_scenes:
                        sv_filename = f'{sv_camera_position}_{sv_scene}'
                        for order in range(1,ORDER+1):
                            coverage_table_generator(order, tv_filename, sv_filename)
        


                        # coverages = []
                        # coverage_table_generator(ORDER, tv_filename, sv_filename)

if __name__ == '__main__':
    # bin = pd.read_csv(f'./coverage_data/round_bunny/round_bunny_0_bin.csv').to_numpy().flatten()
    # print(bin)
    main()