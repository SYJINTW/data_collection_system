import numpy as np
import pandas as pd
import csv
import open3d as o3d
from pathlib import Path

# ============================================================

import CameraPose

# ============================================================

def import_cameras_pose(csvfile_PATH: Path, VIEW_START_POINT: int) -> list:
    cameras_pose = []
    with open(csvfile_PATH, 'r') as csv_f:
        rows = csv.reader(csv_f) # [t,x,y,z,roll,pitch,yaw] in airsim
        next(rows) # skip header
        for row in rows:
            cameras_pose.append(CameraPose.CameraPose('',[row[1],row[2],row[3],row[6],row[5],row[4]],VIEW_START_POINT,'airsim'))
    return cameras_pose

def get_primitive_ids(scene: o3d.cpu.pybind.t.geometry.RaycastingScene, _eye: list, _center: list, _up: list, MAX_NUM_TRI: int):

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

def sample_all_poses_greedy( 
    all_poses: list, 
    threshold_coverage: float, downsample_num: int, num_of_cam: int, 
    scene: o3d.cpu.pybind.t.geometry.RaycastingScene, MAX_NUM_TRI: int):
    '''
    '''
    
    ref_bin = []
    num_of_tv = 0
    poses = []

    for all_pose in all_poses:
        poses.extend(all_pose)
    # print(len(poses))

    tv_arr = np.zeros(len(poses)) # store if the frame is chosen
    coverage_arr = np.zeros(len(poses)) # store the percent of the coverage
    for poseIdx in np.arange(0,len(poses),downsample_num):
        pose = poses[poseIdx]
        bin_arr = get_primitive_ids(scene, pose.camera[0], pose.camera[1], pose.camera[2], MAX_NUM_TRI)
        if num_of_tv == 0: # choose the first tv
            num_of_tv = num_of_tv + 1
            ref_bin = bin_arr
            tv_arr[poseIdx] = 1
            coverage_arr[poseIdx] = 1
        else:
            merge_bin = np.minimum(ref_bin, bin_arr)
            coverage = np.sum(merge_bin) / (1280*720)
            # print(coverage)
            if coverage <= threshold_coverage: # choose the new ref bin and set new frame
                # print('Choose coverage: ', coverage)
                num_of_tv = num_of_tv + 1
                ref_bin = np.maximum(ref_bin, bin_arr)
                tv_arr[poseIdx] = 1
                coverage_arr[poseIdx] = coverage

    print(f'Number of tv: {int(sum(tv_arr))}')
    name_idx = 0
    pose_data = []
    pose_datas_airsim = []
    pose_datas_miv = []
    for poseIdx in range(len(poses)):    
        if tv_arr[poseIdx] == 1:
            pose = poses[poseIdx]
            pose_data.append(pose)
            pose_datas_airsim.append([
                            f'v{name_idx}',
                            pose.airsim[0],
                            pose.airsim[1],
                            pose.airsim[2],
                            pose.airsim[3],
                            pose.airsim[4],
                            pose.airsim[5]
                            ])
            pose_datas_miv.append([
                            f'v{name_idx}',
                            pose.miv[0],
                            pose.miv[1],
                            pose.miv[2],
                            pose.miv[3],
                            pose.miv[4],
                            pose.miv[5]
                            ])
            name_idx = name_idx + 1

    return pose_data, pose_datas_airsim, pose_datas_miv
    
def choose_pose_traces(poses_in_group: list, threshold_coverage: float, downsample_num: int, num_of_cam: int, scene: o3d.cpu.pybind.t.geometry.RaycastingScene, MAX_NUM_TRI: int):
    '''
    '''
    num_of_tv_arr = []
    tv_arrs = []
    coverage_arrs = []
    for poses in poses_in_group:
        
        num_of_tv = 0
        ref_bin = []
        tv_arr = np.zeros(len(poses))
        coverage_arr = np.zeros(len(poses))

        for pose_idx in np.arange(0,len(poses),downsample_num):
            pose = poses[pose_idx]
            bin_arr = get_primitive_ids(scene, pose.camera[0], pose.camera[1], pose.camera[2], MAX_NUM_TRI)

            if pose_idx == 0: # choose the first frame as the ref
                num_of_tv = num_of_tv + 1
                ref_bin = bin_arr
                tv_arr[pose_idx] = 1
                coverage_arr[pose_idx] = 1
            else:
                merge_bin = np.minimum(ref_bin, bin_arr)
                coverage = np.sum(merge_bin) / (1280*720)
                # print(coverage)
                if coverage <= threshold_coverage: # choose the frame and set it as the new ref
                    print(coverage)
                    num_of_tv = num_of_tv + 1
                    ref_bin = bin_arr
                    tv_arr[pose_idx] = 1
                    coverage_arr[pose_idx] = coverage
        
        num_of_tv_arr.append(num_of_tv)
        tv_arrs.append(tv_arr)
        coverage_arrs.append(coverage_arr)
    
    count = 0
    max_add_round = 5
    while sum(num_of_tv_arr) < num_of_cam and count < max_add_round:
        threshold_coverage = ((1 - threshold_coverage) / 2) + threshold_coverage
        # print(f'threshold_coverage: {threshold_coverage}')
        min_pose_idx = num_of_tv_arr.index(min(num_of_tv_arr))
        flag = True
        for idx in range(len(coverage_arrs[min_pose_idx])):
            if tv_arrs[min_pose_idx][idx] == 1:
                flag = True
                continue
            if coverage_arrs[min_pose_idx][idx] <= threshold_coverage and flag:
                num_of_tv_arr[min_pose_idx] = num_of_tv_arr[min_pose_idx] + 1
                tv_arrs[min_pose_idx][idx] = 1
                flag = False
        count = count + 1

    name_idx = 0
    pose_data = []
    pose_datas_airsim = []
    pose_datas_miv = []
    for poses_idx in range(len(poses_in_group)):
        # print(len(poses))
        for idx in range(len(poses_in_group[poses_idx])):
            if tv_arrs[poses_idx][idx] == 1:
                pose = poses_in_group[poses_idx][idx]
                pose_data.append(pose)
                pose_datas_airsim.append([
                                f'v{name_idx}',
                                pose.airsim[0],
                                pose.airsim[1],
                                pose.airsim[2],
                                pose.airsim[3],
                                pose.airsim[4],
                                pose.airsim[5]
                                ])
                pose_datas_miv.append([
                                f'v{name_idx}',
                                pose.miv[0],
                                pose.miv[1],
                                pose.miv[2],
                                pose.miv[3],
                                pose.miv[4],
                                pose.miv[5]
                                ])
                name_idx = name_idx + 1

    return pose_data, pose_datas_airsim, pose_datas_miv

def splitPoses(workdir_PATH: Path, csvfile_PATH_list: list,
            num_of_group: int, num_of_frame: int, threshold_coverage: float, downsample_num: int, num_of_cam: int, 
            scene: o3d.cpu.pybind.t.geometry.RaycastingScene, MAX_NUM_TRI: int, VIEW_START_POINT: int):
    '''
    Args:
        workdir_PATH:
        csvfile_PATH_list: a list of csvfile_PATH (list of Path)
        num_of_group: the number of groups should split
        num_of_frame: the number of frames in each group
        downsample_num:
        num_of_cam:
    '''

    total_frames = num_of_group*num_of_frame

    # read all the poses
    all_poses = []
    for csvfile_PATH in csvfile_PATH_list:
        poses = import_cameras_pose(csvfile_PATH, VIEW_START_POINT)[:total_frames]
        all_poses.append(poses)

    print(f'Total number of poses: {len(all_poses)}')
    # for group_idx in range(len(all_poses)):
    #     print(f'Number of pose traces in pose {group_idx}: {len(all_poses[group_idx])}')
        
    # split all the poses by group and save to csv
    raw_pose_path = Path(workdir_PATH,'pose_traces','raw_poses')
    raw_pose_path.mkdir(parents=True, exist_ok=True)
    poses_in_groups = [] # three dimension, 1. group 2. poses 3. CameraPose
    for idx in np.arange(0,total_frames,num_of_frame):
        group_of_poses = []
        for all_pose_idx in range(len(all_poses)):
            part_of_poses = all_poses[all_pose_idx][idx:idx+num_of_frame]
            group_of_poses.append(part_of_poses)
            tmp = [i.airsim for i in part_of_poses]
            # print(f'{raw_pose_path}/{csvfile_PATH_list[all_pose_idx].stem}_group{int(idx/num_of_frame)}_raw.csv')
            pd.DataFrame(tmp).to_csv(f'{raw_pose_path}/{csvfile_PATH_list[all_pose_idx].stem}_group{int(idx/num_of_frame)}_raw.csv', header=['X','Y','Z','Yaw','Pitch','Roll'],index=False)
        poses_in_groups.append(group_of_poses)

    # mkdir for airsim and miv
    airsim_pose_path = Path(workdir_PATH,'pose_traces','airsim_poses')
    airsim_pose_path.mkdir(parents=True, exist_ok=True)
    miv_pose_path = Path(workdir_PATH,'pose_traces','miv_poses')
    miv_pose_path.mkdir(parents=True, exist_ok=True)

    # choose pose traces for some source views
    pose_data_arr = []
    pose_datas_airsim_arr = []
    pose_datas_miv_arr = []
    for idx in range(len(poses_in_groups)):
        pose_data, pose_datas_airsim, pose_datas_miv = choose_pose_traces(poses_in_groups[idx], threshold_coverage, downsample_num, num_of_cam, scene, MAX_NUM_TRI)
        print(f'Group {idx} get {len(pose_datas_airsim)} source views')
        pose_data_arr.append(pose_data)
        pose_datas_airsim_arr.append(pose_datas_airsim)
        pose_datas_miv_arr.append(pose_datas_miv)
        pd.DataFrame(pose_datas_airsim).to_csv(f'{airsim_pose_path}/poseAll_group{idx}_airsim.csv', header=['Name','X','Y','Z','Yaw','Pitch','Roll'],index=False)
        pd.DataFrame(pose_datas_miv).to_csv(f'{miv_pose_path}/poseAll_group{idx}_miv.csv', header=['Name','X','Y','Z','Yaw','Pitch','Roll'],index=False)
    
    return pose_data_arr

def splitPoses_for_generator(
    workdir_PATH: Path, 
    all_poses: list,
    groupIdx: int, num_of_frame: int, 
    threshold_coverage: float, downsample_num: int, num_of_cam: int, 
    scene: o3d.cpu.pybind.t.geometry.RaycastingScene, MAX_NUM_TRI: int):
    '''
    Args:
    '''
    startFrame = groupIdx*num_of_frame
    endFrame = startFrame+num_of_frame

    all_poses = [i[startFrame:endFrame] for i in all_poses]

    # create dir for splited raw poses
    raw_pose_path = Path(workdir_PATH,'pose_traces','raw_poses')
    raw_pose_path.mkdir(parents=True, exist_ok=True)

    # save the splited raw data in airsim to csv
    for all_pose_idx in range(len(all_poses)):
        poses = [i.airsim for i in all_poses[all_pose_idx]]
        pd.DataFrame(poses).to_csv(f'{raw_pose_path}/pose{all_pose_idx}_group{groupIdx}_raw.csv', 
                                    header=['X','Y','Z','Yaw','Pitch','Roll'],
                                    index=False)

    # deal with all the poses
    pose_data, pose_datas_airsim, pose_datas_miv = sample_all_poses_greedy(all_poses, 
                                                                        threshold_coverage, downsample_num, num_of_cam, 
                                                                        scene, MAX_NUM_TRI)

    # store pose trace for airsim and miv
    airsim_pose_path = Path(workdir_PATH,'pose_traces','airsim_poses')
    airsim_pose_path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pose_datas_airsim).to_csv(f'{airsim_pose_path}/poseAll_group{groupIdx}_airsim.csv', header=['Name','X','Y','Z','Yaw','Pitch','Roll'],index=False)
    miv_pose_path = Path(workdir_PATH,'pose_traces','miv_poses')
    miv_pose_path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pose_datas_miv).to_csv(f'{miv_pose_path}/poseAll_group{groupIdx}_miv.csv', header=['Name','X','Y','Z','Yaw','Pitch','Roll'],index=False)
    
    return pose_data
    
def main():
    print('split_pose')

if __name__ == '__main__':
    main()  

