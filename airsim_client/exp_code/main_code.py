import split_poses
import open3d_coverage
import open3d as o3d 
from pathlib import Path
import numpy as np

# ============================================================

def split_and_cov_table(workdir_PATH: Path, 
                        objfile_PATH: Path, 
                        csvfile_PATH_list: Path, 
                        num_of_group: int,
                        num_of_frame: int,
                        threshold_coverage: float,
                        downsample_num: int,
                        num_of_cam: int,
                        order: int,
                        VIEW_START_POINT: list)->list:
    
    # generate scene
    mesh = o3d.io.read_triangle_mesh(str(objfile_PATH))
    MAX_NUM_TRI = np.asarray(mesh.triangles).shape[0]
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)

    # read all raw poses
    total_frames = num_of_group*num_of_frame
    all_poses = []
    for csvfile_PATH in csvfile_PATH_list:
        poses = split_poses.import_cameras_pose(csvfile_PATH, VIEW_START_POINT)[:total_frames]
        all_poses.append(poses)

    # split the poses by group
    for groupIdx in range(num_of_group):
        print(f'Group {groupIdx}')
        pose_data = split_poses.splitPoses_for_generator(
            workdir_PATH,
            all_poses,
            groupIdx, num_of_frame, 
            threshold_coverage, downsample_num, num_of_cam, 
            scene, MAX_NUM_TRI)
        coverage_table = open3d_coverage.computeQualityModel(
            workdir_PATH, 
            pose_data, pose_data, 
            order, groupIdx, scene, MAX_NUM_TRI)
        
        poses = np.empty((0,6))
        for pose in pose_data:
            poses = np.append(poses, np.array([[pose.airsim[0],pose.airsim[1],pose.airsim[2],pose.airsim[5],pose.airsim[4],pose.airsim[3]]]), axis=0)

        yield poses, coverage_table
    
def main():
    workdir_PATH = Path('./test')
    objfile_PATH = Path('./test/objSrc/bunny_mesh_merge.OBJ')
    csvfile_PATH_list = [Path(f'./test/raw_poses/pose{i}.csv') for i in range(5)]
    num_of_group = 3
    num_of_frame = 30
    threshold_coverage = 0.8
    downsample_num = 5
    num_of_cam = 12
    order = 2
    VIEW_START_POINT = [0,0,5] # meters

    split_and_cov_table_generator = split_and_cov_table(workdir_PATH,
                                                        objfile_PATH,
                                                        csvfile_PATH_list,
                                                        num_of_group,
                                                        num_of_frame,
                                                        threshold_coverage,
                                                        downsample_num,
                                                        num_of_cam,
                                                        order,
                                                        VIEW_START_POINT)
    
    for idx in range(num_of_group):
        poses, table = next(split_and_cov_table_generator)
        print(poses)

if __name__ == '__main__':
    main()