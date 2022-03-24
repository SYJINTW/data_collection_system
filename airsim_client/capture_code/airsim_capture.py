import setup_path
import airsim
import time
import csv
import cv2
import math
import numpy as np
import pandas as pd
import os
import json
from pathlib import Path

# Basic setting
# ============================================================

CAPTURE_TEXTURE = True
CAPTURE_DEPTH = True

# setting.json setting for airsim server
# ============================================================

SETTINGS = {
  "SeeDocsAt": "https://github.com/Microsoft/AirSim/blob/master/docs/settings.md",
  "SettingsVersion": 1.2,
  "SimMode": "ComputerVision",
  "CameraDefaults": {
    "CaptureSettings": [
            {
                "ImageType": 0,
                "Width": 1280,
                "Height": 720,
                "FOV_Degrees": 90,
                "AutoExposureSpeed": 100,
                "MotionBlurAmount": 0
            },
            {
                "ImageType": 1,
                "Width": 1280,
                "Height": 720,
                "FOV_Degrees": 90,
                "AutoExposureSpeed": 100,
                "MotionBlurAmount": 0
            },
            {
                "ImageType": 2,
                "Width": 1280,
                "Height": 720,
                "FOV_Degrees": 90,
                "AutoExposureSpeed": 100,
                "MotionBlurAmount": 0
            },
            {
                "ImageType": 4,
                "Width": 1280,
                "Height": 720,
                "FOV_Degrees": 90,
                "AutoExposureSpeed": 100,
                "MotionBlurAmount": 0
            }
        ]
    },
    "Recording": {
        "RecordInterval": 2
    },
    "CameraDirector": {
        "FollowDistance": 0
    }
}

RESOLUTION = [
    SETTINGS['CameraDefaults']['CaptureSettings'][0]['Width'],
    SETTINGS['CameraDefaults']['CaptureSettings'][0]['Height'],
]

# ============================================================

class Camera_pose:
    name = ""
    position = [0, 0, 0] # X, Y, Z
    rotation = [0, 0, 0] # Yaw, Pitch, Roll
    
    def __init__(self, _name='', _array=[0,0,0,0,0,0]):
        self.name = _name
        self.position = [float(x) for x in _array[0:3]]
        self.rotation = [float(x) for x in _array[3:6]]
        # print(self.position)
        # print(self.rotation)
    
# ============================================================

def import_airsim_pose(csvfile_PATH): # airsim data from SM [X,Y,Z,Roll,Pitch,Yaw]
    cameras_pose = []
    with open(csvfile_PATH, 'r') as csv_f:
        rows = csv.reader(csv_f)
        next(rows) # skip header
        pose_idx = 0
        for row in rows:
            row = row[0].split()
            if not (float(row[0]) == 0 and float(row[1]) == 0 and float(row[2]) == 0):  
                cameras_pose.append(Camera_pose(f'v{pose_idx}', [row[0],row[1],row[2],row[5],row[4],row[3]]))
                pose_idx = pose_idx + 1
    return cameras_pose

def import_raw_pose(csvfile_PATH): # raw airsim data [X,Y,Z,Yaw,Pitch,Roll]
    cameras_pose = []
    with open(csvfile_PATH, 'r') as csv_f:
        rows = csv.reader(csv_f)
        next(rows) # skip header
        pose_idx = 0
        for row in rows:
            cameras_pose.append(Camera_pose(f'v{pose_idx}', [row[0],row[1],row[2],row[3],row[4],row[5]]))
            pose_idx = pose_idx + 1
    return cameras_pose

def set_camera_pose_to_airsim(client, camera_pose):
    '''
    Change the position in airsim related to the starting point

    class camera_pose
    camera_pose.position[x, y, z]
    camera_pose.rotation[yaw, pitch, roll]
    '''
    client.simSetCameraPose("front_center", airsim.Pose(
        airsim.Vector3r(0, 0, 0), airsim.to_quaternion(0, 0, 0)))
    client.simSetVehiclePose(
        airsim.Pose(
            # airsim.Vector3r(x, y, z)
            airsim.Vector3r(
                x_val=camera_pose.position[0],
                y_val=camera_pose.position[1],
                z_val=camera_pose.position[2]
                ),
            # airsim.to_quaternion(pitch, roll, yaw)
            airsim.to_quaternion(
                yaw=camera_pose.rotation[0]*math.pi/180,
                pitch=camera_pose.rotation[1]*math.pi/180,
                roll=camera_pose.rotation[2]*math.pi/180
                ),
            ),
        True
    )
    pose = client.simGetVehiclePose()
    
def output_texture_responses_to_yuv(savedir_PATH: Path, name_responses: list, tex_responses: list):
    '''
    This function will output the texture output file in yuv10le format.
    '''

    yuv_frames = [] # to store all the 16 bits frames
    
    for response_idx, response in enumerate(tex_responses):
        filename = f'{savedir_PATH}/{name_responses[response_idx]}'
        if not response.pixels_as_float: # Scene
            # save png
            # Get from https://github.com/microsoft/AirSim/blob/master/docs/image_apis.md#using-airsim-images-with-numpy
            # get numpy array
            img1d = np.fromstring(
                response.image_data_uint8, dtype=np.uint8)  # get numpy array
            print(img1d.shape)
            # reshape array to 4 channel image array H X W X 4
            img_rgb = img1d.reshape(response.height, response.width, 3)
            # # original image is fliped vertically
            # img_rgb = np.flipud(img_rgb)
            # write to png 
            airsim.write_png(os.path.normpath(f'{filename}_tex.png'), img_rgb)
            # save png end

            # turn png into yuv
            os.system(
                f"powershell ffmpeg -y \
                    -i {filename}_tex.png \
                    -pix_fmt yuv420p10le \
                    {filename}_texture_{RESOLUTION[0]}x{RESOLUTION[1]}_yuv420p10le.yuv"
                )

def cal_surface(v1, v2, v3):
    vector1 = v1-v2
    vector2 = v1-v3
    n_vector = np.cross(vector1, vector2)
    d = -np.dot(v1,n_vector)
    return np.append(n_vector, d)

def find_point_in_surface(surface_para, point, viewport):
    line_vector = viewport - point
    n_vector = np.array(surface_para[0], surface_para[1], surface_para[2])
    k = -(np.dot(point, n_vector)+surface_para[3])/np.dot(line_vector, n_vector)
    
def output_depth_responses_to_yuv(savedir_PATH, name_responses, depth_responses, zmin, zmax):
    '''
    This function will output the depth output file in yuv16le format.
    The depth information capture by airsim.ImageType.DisparityNormalized
    cannot supply correct depth image for MIV.
    So that we have to transform the depth information
    from airsim to 16 bit for MIV type.
    '''
     
    for response_idx, response in enumerate(depth_responses):
        filename = f'{savedir_PATH}/{name_responses[response_idx]}'
        if response.pixels_as_float:
            response_float_data = response.image_data_float
            response_float_data = 0.125/np.array(response_float_data)
            response_float_data = response_float_data.flatten()
            depth_16bit = (((1/response_float_data-1/zmax) /
                        (1/zmin-1/zmax)) * 65535)
            depth_16bit = depth_16bit.astype(np.int16)
            yuv_frames = np.append(depth_16bit, np.full(
                int(len(depth_16bit)/2), 32768, dtype=np.int16))
            print(f"Type {response.image_type}, size {len(response.image_data_float)}")
            print(response.height, response.width)
            print(depth_16bit.dtype)
            with open(f"{filename}_depth_{RESOLUTION[0]}x{RESOLUTION[1]}_yuv420p16le.yuv", mode='wb') as f:
                yuv_frames.tofile(f)
    
def z_boundary(depth_responses):
    zmin = math.inf
    zmax = -math.inf
    for response in depth_responses:
        zmin_tmp, zmax_tmp = get_zmin_zmax(response)
        zmin = min(zmin, zmin_tmp)
        zmax = max(zmax, zmax_tmp)
        print(f'zmin {zmin}, zmax {zmax}')
    return zmin, zmax

def get_zmin_zmax(depth_response):
    '''
        return (min, max) of disp
        @para responses: [disp, scene]
    '''
    response = depth_response.image_data_float
    response = 0.125/np.array(response) # from disp to depth
    return response.min(), response.max()

def generate_camera_para_json(cameras_pose, num_frames, zmin, zmax, contentName):
    '''
    return json object
    This function will generate a json file for MIV 
    '''
    camera_parameter = {}
    camera_parameter['Version'] = '4.0'
    camera_parameter["BoundingBox_center"] = [0, 0, 0]
    camera_parameter["Fps"] = 30
    camera_parameter["Content_name"] = contentName
    camera_parameter["Frames_number"] = num_frames
    camera_parameter["lengthsInMeters"] = True
    camera_parameter["sourceCameraNames"] = [
        camera_pose.name for camera_pose in cameras_pose]
    camera_parameter["cameras"] = []
    for camera_pose in cameras_pose:
        camera = {}
        camera["BitDepthColor"] = 10
        camera["BitDepthDepth"] = 16
        camera["Name"] = camera_pose.name
        camera["Depth_range"] = [zmin, zmax]
        camera["DepthColorSpace"] = "YUV420"
        camera["ColorSpace"] = "YUV420"
        MIV_camera_pose = convert_airsim_coordinate_to_MIV_coordinate(
            camera_pose)
        camera["Position"] = MIV_camera_pose.position
        camera["Rotation"] = MIV_camera_pose.rotation
        camera["Resolution"] = RESOLUTION
        camera["Projection"] = "Perspective"
        camera["HasInvalidDepth"] = False
        camera["Depthmap"] = 1
        camera["Background"] = 0
        # F = w / (2 * tan(FOV/2))
        camera["Focal"] = [
            camera["Resolution"][0] / (2 * math.tan(90/2 * math.pi/180)), camera["Resolution"][0] / (2 * math.tan(90/2 * math.pi/180))]
        # print(camera["Focal"])
        # w / 2, h / 2
        camera["Principle_point"] = [
            camera["Resolution"][0]/2, camera["Resolution"][1]/2]
        camera_parameter["cameras"].append(camera)
    viewport_parameter = camera_parameter["cameras"][0].copy()
    viewport_parameter["Name"] = "viewport"
    viewport_parameter["Position"] = [0.0, 0.0, 0.0]
    viewport_parameter["Rotation"] = [0.0, 0.0, 0.0]
    viewport_parameter["HasInvalidDepth"] = True
    camera_parameter["cameras"].append(viewport_parameter)
    return camera_parameter

def convert_airsim_coordinate_to_MIV_coordinate(airsim_camera_pose):
    MIV_camera_pose = Camera_pose()
    # x, y, z
    MIV_camera_pose.position = [
        airsim_camera_pose.position[0], -airsim_camera_pose.position[1], -airsim_camera_pose.position[2]]
    # yaw, pitch, roll
    MIV_camera_pose.rotation = [
        -airsim_camera_pose.rotation[0], -airsim_camera_pose.rotation[1], airsim_camera_pose.rotation[2]]
    return MIV_camera_pose

# ============================================================

def capture_main(workdir_PATH: Path, csvfile_PATH: Path):
    '''
    workdir_PATH:
    csvfile_Path:
    '''

    # create save dir
    groupNum = csvfile_PATH.stem.split('_')[1]
    savedir_PATH = Path(workdir_PATH,'capture_SV',f'group{groupNum}')
    savedir_PATH.mkdir(parents=True, exist_ok=True)

    # read pose traces (where cameras should pose and rotate)

    pose_traces = import_airsim_pose(csvfile_PATH)
    
    # connect to the AirSim simulator
    client = airsim.VehicleClient()
    client.confirmConnection()

    name_responses = []
    texture_responses = []
    depth_responses = []
    
    for pose_trace in pose_traces:
        name_responses.append(pose_trace.name)
        set_camera_pose_to_airsim(client, pose_trace) # change camera position
        time.sleep(1)
        client.simPause(True)
        print('Finish changing position')
        if CAPTURE_TEXTURE:
            if CAPTURE_DEPTH:
                responses = client.simGetImages([
                    airsim.ImageRequest('', airsim.ImageType.Scene, False, False),
                    airsim.ImageRequest('', airsim.ImageType.DisparityNormalized, True),
                    ])
                texture_responses.append(responses[0])
                depth_responses.append(responses[1])
            else:
                responses = client.simGetImages([
                    airsim.ImageRequest('', airsim.ImageType.Scene, False, False),
                    ])
                texture_responses.append(responses[0])
        else:
            continue
        print('Retrieved images: %d' % len(responses))
        client.simPause(False)
    
    # # dir to store yuv file
    # save_dir = Path(workdir_PATH,'capture_data',f'group{groupNum}')
    # save_dir.mkdir(parents=True, exist_ok=True)

    output_texture_responses_to_yuv(savedir_PATH, name_responses, texture_responses)
    
    if CAPTURE_DEPTH:
        zmin, zmax = z_boundary(depth_responses)
        output_depth_responses_to_yuv(savedir_PATH, name_responses, depth_responses, zmin, zmax)
        camera_parameter = generate_camera_para_json(pose_traces, 1, zmin, zmax, workdir_PATH.stem)
        with open(f'{savedir_PATH}/group{groupNum}.json', 'w') as f:
            json.dump(camera_parameter, f)

def capture_gt(workdir_PATH: Path, csvfile_PATH: Path):
    '''
    workdir_PATH:
    poseNum:
    groupNum:
    '''

    # create save dir
    filename_split = csvfile_PATH.stem.split('_')
    savedir_PATH = Path(workdir_PATH,'capture_GT',filename_split[0],filename_split[1])
    savedir_PATH.mkdir(parents=True, exist_ok=True)

    # read pose traces (where cameras should pose and rotate)
    pose_traces = import_raw_pose(csvfile_PATH)
    
    # connect to the AirSim simulator
    client = airsim.VehicleClient()
    client.confirmConnection()

    name_responses = []
    texture_responses = []
    
    for pose_trace in pose_traces:
        name_responses.append(pose_trace.name)
        set_camera_pose_to_airsim(client, pose_trace) # change camera position
        time.sleep(1)
        client.simPause(True)
        print('Finish changing position')
        responses = client.simGetImages([
            airsim.ImageRequest('', airsim.ImageType.Scene, False, False),
            ])
        texture_responses.append(responses[0])
        print('Retrieved images: %d' % len(responses))
        client.simPause(False)
    
    output_texture_responses_to_yuv(savedir_PATH, name_responses, texture_responses)

def merge_gt(workdir_PATH: Path, poseNum: int, groupNum: int):
    
    yuvfile_PATH = Path(workdir_PATH,'ground_truth',f'pose{poseNum}',f'group{groupNum}')
    
    save_PATH = Path(workdir_PATH,'ground_truth')
    
    all_gt_yuv = yuvfile_PATH.glob('*_texture_1280x720_yuv420p10le.yuv')
    gt_num = len([i.stem for i in all_gt_yuv])
    
    cat_list = ''
    for idx in range(gt_num):
        cat_list = cat_list + f"file v{idx}_texture_1280x720_yuv420p10le.yuv\n"
    with open(f'{yuvfile_PATH}\merge.txt', 'w') as f:
        f.write(cat_list)

# ============================================================

def main():
    all_workdir_PATH = Path('.').glob('idF5_*')
    for workdir_PATH in all_workdir_PATH:
        print(workdir_PATH)
        for csvfile_PATH in Path(workdir_PATH).glob('sourceView_*'):
            print(csvfile_PATH)
            capture_main(workdir_PATH, csvfile_PATH)

def gt_main():
    all_workdir_PATH = Path('.').glob('idF5_*')
    for workdir_PATH in all_workdir_PATH:
        print(workdir_PATH)
        for csvfile_PATH in Path(workdir_PATH,'pose_traces','raw_poses').glob('*_raw.csv'):
            print(csvfile_PATH)
            capture_gt(workdir_PATH, csvfile_PATH)

if __name__ == '__main__':
    # gt_main()
    start_time = time.time()
    main()
    end_time = time.time()
    print(end_time-start_time)


