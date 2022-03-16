import numpy as np
from scipy.spatial.transform import Rotation as R

class Camera_pose:
    name = ""
    ue = np.zeros(6) # X, Y, Z, Yaw, Pitch, Roll -> meters and degrees
    airsim = np.zeros(6) # X, Y, Z, Yaw, Pitch, Roll -> meters and degrees
    miv = np.zeros(6) # X, Y, Z, Yaw, Pitch, Roll -> meters and degrees
    camera = np.array([
            np.zeros(3), # eye
            np.zeros(3), # center
            np.zeros(3)  # up
            ])

    position = np.zeros(3) # X, Y, Z
    rotation = np.zeros(3) # Yaw, Pitch, Roll
     
    def __init__(self, _name, _array=[0,0,0,0,0,0],_cam_starting_point=[0,0,0],_coordinate='ue'):
        
        '''
        _name: name of this pose
        _array: [x,y,z,yaw,pitch,roll]
        _cam_starting_point: [x,y,z]
        _coordinate: which coordinate is the input _array
        '''

        self.name = _name
        _array = [float(i) for i in _array]
        _cam_starting_point = [float(i) for i in _cam_starting_point]
        if _coordinate == 'ue':
            self.ue = np.array(_array)
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
        airsim_arr = np.array([
                    ue_arr[0],
                    ue_arr[1],
                    -ue_arr[2],
                    -ue_arr[3],
                    ue_arr[4],
                    ue_arr[5],
                    ])
        return airsim_arr

    def airsim_to_ue(self, airsim_arr: list)->list:
        ue_arr = np.array([
                    airsim_arr[0],
                    airsim_arr[1],
                    -airsim_arr[2],
                    -airsim_arr[3],
                    airsim_arr[4],
                    airsim_arr[5],
                    ])
        return ue_arr

    def ue_to_miv(self, ue_arr: list):
        miv_arr = np.array([
                    ue_arr[0], 
                    -ue_arr[1], 
                    ue_arr[2],
                    ue_arr[3], 
                    -ue_arr[4],
                    ue_arr[5]
                    ])
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

        return np.array([eye,center,up])
