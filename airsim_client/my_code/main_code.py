from pathlib import Path
import choose_pose_traces
import open3d_coverage

def split_pose_and_generate_coverage_table(csvfile_PATH: Path, objfile_PATH: Path, downsample_num: int, threshold_coverage: float, num_for_group: int, dir_name: str, order: int)->list:
    '''
    Args:
        csvfile_PATH: the path of the raw pose traces file. [t,x,y,z,roll,pitch,yaw]
        objfile_PATH: the path of the OBJ file
        downsample_num: we will check skip downsample number of frames
        threshold_coverage: we will get the frame which coverage is below threshold coverage. Should fill in a float between [0.0,1.0]
        num_for_group: the number of frames in a time interval
        dir_name: just for the intermediate data. Suggest to fill in a name that we can recognize which experiment parameters we are setting
        order: for generating coverage table

    Returns:
        the list of Camera_pose
        the list of dictionaries: [{CamSet: coverage}, {(0,): 1.0, (1,): 0.5}, {(0,): 1.0, (1,): 0.5}]
    '''
    output_poses_generator = choose_pose_traces.choose_pose_traces(csvfile_PATH, objfile_PATH, downsample_num, threshold_coverage, num_for_group, dir_name)
    for group_idx, output_camera_poses in output_poses_generator:
        # print(group_idx)
        # for output_camera_pose in output_camera_poses:
        #     print(output_camera_pose.ue)
        coverage_table = open3d_coverage.computeQualityModel(objfile_PATH,output_camera_poses,output_camera_poses,order,dir_name,group_idx)
        yield output_camera_poses, coverage_table

def main():
    csvfile_PATH = './pose_traces/raw_pose_from_capture.csv'
    objfile_PATH = './obj_source/bunny_mesh_merge.OBJ'
    downsample_num = 10
    threshold_coverage = 0.7
    num_for_group = 5
    dir_name = 'test_pose'
    order = 3

    coverage_table_generator = split_pose_and_generate_coverage_table(csvfile_PATH, objfile_PATH, downsample_num, threshold_coverage, num_for_group, dir_name,order)
    poses, coverage_table =  next(coverage_table_generator)
    print(type(poses))
    print(poses[0].ue)
    print(type(poses[0].ue))

                        
if __name__ == '__main__':
    main()