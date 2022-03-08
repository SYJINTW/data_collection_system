import pandas as pd
import numpy as np
import math
from scipy.spatial.transform import Rotation as R

filename = 'bunny_mesh_merge'

def vertex_in_the_normal_side(vertex_datas, v1, v2, point):
    '''
    vetex data = [id, x, y, z]
    '''
    surface_nor = np.cross(v1,v2)
    d = np.dot(point,surface_nor.T)
    surface_check = [
                    [1,0],
                    [0,surface_nor[0]],
                    [0,surface_nor[1]],
                    [0,surface_nor[2]],
                    ]
    vertex_datas = vertex_datas.dot(surface_check)
    vertex_bin = vertex_datas[:,-1]
    vertex_bin = np.where(vertex_bin>=d,True,vertex_bin)
    vertex_bin = np.where(vertex_bin<d,False,vertex_bin)
    # print(vertex_bin.shape[0])
    vertex_bin = np.append([False],vertex_bin)
    # print(vertex_bin.shape[0])
    return vertex_bin
    

def del_face_from_vertex(face_datas, vertex_bin):
    for idx in range(face_datas.shape[0]-1,-1,-1):
        if vertex_bin[int(face_datas[idx][1]-1)] == 0 or vertex_bin[int(face_datas[idx][5]-1)] == 0 or vertex_bin[int(face_datas[idx][9]-1)] == 0:
            face_datas = np.delete(face_datas, idx, 0)
    return face_datas

def rotate_all_the_point(x,y,z,yaw,pitch,roll,vertex_datas):
    # rotation matrix
    # | cos(yaw)cos(pitch) -cos(yaw)sin(pitch)sin(roll)-sin(yaw)cos(roll) -cos(yaw)sin(pitch)cos(roll)+sin(yaw)sin(roll)|
    # | sin(yaw)cos(pitch) -sin(yaw)sin(pitch)sin(roll)+cos(yaw)cos(roll) -sin(yaw)sin(pitch)cos(roll)-cos(yaw)sin(roll)|
    # | sin(pitch)          cos(pitch)sin(roll)                            cos(pitch)sin(roll)|
    
    
    Rz_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [          0,            0, 1]])
    Ry_pitch = np.array([
        [ np.cos(pitch), 0, np.sin(pitch)],
        [             0, 1,             0],
        [-np.sin(pitch), 0, np.cos(pitch)]])
    Rx_roll = np.array([
        [1,            0,             0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]])
    # R = RzRyRx
    rotMat = np.round_(np.dot(Rz_yaw, np.dot(Ry_pitch, Rx_roll)), decimals=2)
    rotMat = np.c_[[0,0,0],rotMat]
    rotMat = np.vstack([np.array([1,0,0,0]),rotMat])
    # print(vertex_datas.shape[0])
    # print(vertex_datas.shape[1])

    vertex_datas_rotate = np.dot(rotMat, vertex_datas.T).T
    # print(vertex_datas_rotate.shape[0])
    # print(vertex_datas_rotate.shape[1])
    return vertex_datas_rotate

def find_point_in_triangle(vertex_datas, v1, v2, v3, viewpoint, point):
    '''
    vetex_data = [id, x, y, z]
    v1 = [x,y,z]
    v2 = [x,y,z]
    v3 = [x,y,z]
    point = [x, y, z]
    '''
        


def main():
    x = -200
    y = 0
    z = 0
    yaw = 0
    pitch = 0
    roll = 0

    viewport_pos = np.array([x,y,z])
    yaw = np.radians(yaw)
    pitch = np.radians(pitch)
    roll = np.radians(roll)

    # viewport = [x,y,z,yaw,pitch,roll]

    # mid vector [1,0,0]
    # a [1,-1,1.8]
    # b [1,1,1.8]
    # c [1,1,-1.8]
    # d [1,-1,-1.8]
    viewpoint_mid = np.array([1,0,0])
    viewpoint_a = np.array([1,-1,1.8])
    viewpoint_b = np.array([1,1,1.8])
    viewpoint_c = np.array([1,1,-1.8])
    viewpoint_d = np.array([1,-1,-1.8])
    
    # read file
    # tri_data = [id,v1,v2,v3]
    tri_datas = pd.read_csv(f'./obj_source/{filename}_tri_datas.csv').to_numpy()
    # vertex_data = [id,x,y,z]
    vertex_datas = pd.read_csv(f'./obj_source/{filename}_vertex_datas.csv').to_numpy()

    # change all point into viewpoint cooridinate
    rotate_all_the_point(x,y,z,yaw,pitch,roll,vertex_datas)

    # create a binary list to store vertex usage
    vertex_bin = np.ones(vertex_datas.shape[0]+1)
    tri_bin = np.ones(tri_datas.shape[0]+1)
    vertex_bin[0] = 0
    tri_bin[0] = 0
    
    # 1
    # delete outside(-)
    vertex_bin = np.logical_and(vertex_in_the_normal_side(vertex_datas, viewpoint_a, viewpoint_b, [0,0,0]), vertex_bin)
    print(vertex_bin)
    print(np.count_nonzero(vertex_bin))
    vertex_bin = np.logical_and(vertex_in_the_normal_side(vertex_datas, viewpoint_b, viewpoint_c, [0,0,0]), vertex_bin)
    print(vertex_bin)
    print(np.count_nonzero(vertex_bin))
    vertex_bin = np.logical_and(vertex_in_the_normal_side(vertex_datas, viewpoint_c, viewpoint_d, [0,0,0]), vertex_bin)
    print(vertex_bin)
    print(np.count_nonzero(vertex_bin))
    vertex_bin = np.logical_and(vertex_in_the_normal_side(vertex_datas, viewpoint_d, viewpoint_a, [0,0,0]), vertex_bin)
    print(vertex_bin)
    print(np.count_nonzero(vertex_bin))
    
    # 2
    # change tri binary array
    for tri_data in tri_datas:
        for i in range(1,4):
            if vertex_bin[tri_data[i]] == False:
                tri_bin[tri_data[0]] = False
    print("Numbers of triangle: ", tri_bin.shape[0])
    print("Numbers of TRUE triangle: ", np.count_nonzero(tri_bin))


    




    # # 3
    # for tri_data in tri_datas:
    #     if tri_bin[tri_data[0]] == True:
    #         v1 = np.array(vertex_datas[tri_data[1]-1][1:])
    #         v2 = np.array(vertex_datas[tri_data[2]-1][1:])
    #         v3 = np.array(vertex_datas[tri_data[3]-1][1:])
    #         vertex_bin = np.logical_and(vertex_in_the_normal_side(vertex_datas, v2, v1, [0,0,0]), vertex_bin)
    #         vertex_bin = np.logical_and(vertex_in_the_normal_side(vertex_datas, v3, v2, [0,0,0]), vertex_bin)
    #         vertex_bin = np.logical_and(vertex_in_the_normal_side(vertex_datas, v1, v3, [0,0,0]), vertex_bin)

    # # 4
    # # change tri binary array
    # for tri_data in tri_datas:
    #     for i in range(1,4):
    #         if vertex_bin[tri_data[i]] == False:
    #             tri_bin[tri_data[0]] = False
    # print("Numbers of triangle: ", tri_bin.shape[0])
    # print("Numbers of TRUE triangle: ", np.count_nonzero(tri_bin))

    # 5
    

    # df = pd.DataFrame(tri_datas)
    # df_v = pd.DataFrame(vertex_datas)
    # df.to_csv(f'./obj_source/test.csv', header=['f_id', 'v1_id', 'x1', 'y1', 'z1', 'v2_id', 'x2', 'y2', 'z2', 'v3_id', 'x3', 'y3', 'z3', 'nor_check'], index=False)
    # df_v.to_csv(f'./obj_source/test_v.csv', header=['id', 'x', 'y', 'z', 'check'], index=False)



if __name__ == '__main__':
    main()
    # A = np.array([[1,2,3],[4,5,6]])
    # print(A)
    # print(A.T)
    # print(A.T.T)
    