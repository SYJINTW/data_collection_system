from tabnanny import check
import pandas as pd
import numpy as np
import math
from scipy.spatial.transform import Rotation as R

filename = 'bunny_mesh_merge'

def vertex_in_the_pos_normal_side(vertex_datas, v1, v2, surface_point):
    '''
    vetex data = [id, x, y, z]
    v1 = [x,y,z]
    v2 = [x,y,z]
    surface_point = [x,y,z]
    '''
    # print(vertex_datas)
    surface_nor = np.cross(v1,v2)
    d = np.dot(surface_point,surface_nor.T)
    surface_check = [
                    [1,0],
                    [0,surface_nor[0]],
                    [0,surface_nor[1]],
                    [0,surface_nor[2]],
                    ]
    vertex_d = np.dot(vertex_datas, surface_check)
    vertex_d[:,1] = vertex_d[:,1] - d
    vertex_bin = vertex_d[:,1]
    vertex_bin = np.where(vertex_bin>=0, 1, vertex_bin)
    vertex_bin = np.where(vertex_bin<0, 0, vertex_bin)
    return vertex_bin
    
def del_tri(tri_datas,vertex_bin):
    tri_datas_shape = tri_datas[:,1:] # delete id
    a = tri_datas_shape.shape[0]
    b = tri_datas_shape.shape[1]
    # print(a,b)
    tri_datas_shape = tri_datas_shape.flatten()
    # print(tri_datas_shape)
    tri_datas_shape[:] = tri_datas_shape[:] - 1
    # print(tri_datas_shape)
    tri_vertex_bin = vertex_bin[tri_datas_shape]
    # print(tri_vertex_bin)
    tri_vertex_bin = tri_vertex_bin.reshape(a,b)
    # print(tri_vertex_bin)
    tri_bin = np.logical_and(tri_vertex_bin[:,0],tri_vertex_bin[:,1])
    tri_bin = np.logical_and(tri_vertex_bin[:,2],tri_bin)
    # print(tri_bin)
    # print(np.count_nonzero(tri_bin))
    return tri_bin


def rotate_all_the_point(x,y,z,yaw,pitch,roll,vertex_datas):
    '''
    vertex_data = [id,x,y,z]
    '''
    vertex_datas[:,1] = vertex_datas[:,1] - x
    vertex_datas[:,2] = vertex_datas[:,2] - y
    vertex_datas[:,3] = vertex_datas[:,3] - z

    r = R.from_euler('zyx', [yaw, pitch, roll], degrees=True)
    rotation_mat = r.as_matrix()
    rotation_mat = np.c_[[0,0,0], rotation_mat]
    # print(rotation_mat)
    rotation_mat = np.vstack([[1,0,0,0],rotation_mat])
    # print(rotation_mat)
    rotation_mat_inv = np.linalg.inv(rotation_mat)
    # print(rotation_mat_inv)
    vertex_datas_rotate = np.dot(rotation_mat_inv, vertex_datas.T).T
    
    return vertex_datas_rotate

# def find_neg_normal(tri_datas,vertex_datas,tri_bin,vertex_bin):
#     tri_vertex_datas = tri_datas[:,1:]
#     for idx in range(tri_datas.shape[0]):
#         vertex_datas[tri_datas[idx][1]][1]



def check_vertex_and_tri(tri_datas,vertex_datas,tri_bin,vertex_bin):
    '''
    tri_data = [id,v1,v2,v3]
    vertex_data = [id,x,y,z]
    '''
    for idx in range(tri_datas.shape[0]):
        if tri_bin[idx] == 1:
            tri_data = tri_datas[idx]
            v1 = vertex_datas[tri_data[1]-1][1:]
            v2 = vertex_datas[tri_data[2]-1][1:]
            v3 = vertex_datas[tri_data[3]-1][1:]

            # three face
            surface_nor = np.cross(v1,v2)
            d = np.dot([0,0,0],surface_nor.T)
            surface_check = [
                    [1,0],
                    [0,surface_nor[0]],
                    [0,surface_nor[1]],
                    [0,surface_nor[2]],
                    ]
            vertex_d = np.dot(vertex_datas, surface_check)
            vertex_d[:,1] = vertex_d[:,1] - d
            vertex_bin_1 = vertex_d[:,1] - d
            vertex_bin_1 = np.where(vertex_bin_1>=0, 1, vertex_bin_1)
            vertex_bin_1 = np.where(vertex_bin_1<0, 0, vertex_bin_1)
            
            # in front of the surface 
            v21 = v2 - v1
            v31 = v3 - v1
            surface_nor = np.cross(v21,v31)
            d = np.dot(v1,surface_nor.T)
            surface_check = [
                    [1,0],
                    [0,surface_nor[0]],
                    [0,surface_nor[1]],
                    [0,surface_nor[2]],
                    ]
            vertex_d = np.dot(vertex_datas, surface_check)
            vertex_d[:,1] = vertex_d[:,1] - d
            vertex_bin_2 = vertex_d[:,1]
            vertex_bin_2 = np.where(vertex_bin_2>=0, 1, vertex_bin_2)
            vertex_bin_2 = np.where(vertex_bin_2<0, 0, vertex_bin_2)

            vertex_bin_12 = np.logical_or(vertex_bin_1, vertex_bin_2)
            vertex_bin = np.logical_and(vertex_bin_12, vertex_bin)
            print('vertex_bin: ', vertex_bin)
            print(np.count_nonzero(vertex_bin))
            tri_bin = del_tri(tri_datas,vertex_bin)*tri_bin
            print('tri_bin: ', tri_bin)
            print(np.count_nonzero(tri_bin))
    return vertex_bin, tri_bin
            

def main(_x,_y,_z,_yaw,_pitch,_roll):
    x = _x # cm
    y = _y # cm
    z = _z # cm
    yaw = _yaw # degree
    pitch = _pitch # degree
    roll = _roll # degree

    # mid vector [1,0,0]
    # a [1,-1,1.8]
    # b [1,1,1.8]
    # c [1,1,-1.8]
    # d [1,-1,-1.8]
    viewpoint = np.array([0,0,0])
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

    # create a binary list to store vertex usage
    vertex_bin = np.ones(vertex_datas.shape[0])
    tri_bin = np.ones(tri_datas.shape[0])

    # 1
    # change all point into viewpoint cooridinate
    vertex_datas = rotate_all_the_point(x,y,z,yaw,pitch,roll,vertex_datas)
    
    # 2
    # delete outside(-)
    vertex_bin = vertex_in_the_pos_normal_side(vertex_datas, viewpoint_a, viewpoint_b, viewpoint)*vertex_bin
    print(vertex_bin)
    print(np.count_nonzero(vertex_bin))
    vertex_bin = vertex_in_the_pos_normal_side(vertex_datas, viewpoint_b, viewpoint_c, viewpoint)*vertex_bin
    print(vertex_bin)
    print(np.count_nonzero(vertex_bin))
    vertex_bin = vertex_in_the_pos_normal_side(vertex_datas, viewpoint_c, viewpoint_d, viewpoint)*vertex_bin
    print(vertex_bin)
    print(np.count_nonzero(vertex_bin))
    vertex_bin = vertex_in_the_pos_normal_side(vertex_datas, viewpoint_d, viewpoint_a, viewpoint)*vertex_bin
    print(vertex_bin)
    print(np.count_nonzero(vertex_bin))
    
    # 3
    # change tri binary array
    tri_bin = del_tri(tri_datas,vertex_bin)*tri_bin
    print('tri_bin: ', tri_bin)
    print(np.count_nonzero(tri_bin))

    # 4
    # delete normal dot positive
    # find_neg_normal(tri_datas,vertex_datas,tri_bin,vertex_bin)

    # 5
    vertex_bin, tri_bin = check_vertex_and_tri(tri_datas, vertex_datas, tri_bin, vertex_bin)
    print(np.count_nonzero(vertex_bin))
    print(np.count_nonzero(tri_bin))

    df = pd.DataFrame(tri_bin)
    df_v = pd.DataFrame(vertex_bin)
    df.to_csv(f'./obj_source/test.csv', index=False)
    df_v.to_csv(f'./obj_source/test_v.csv', index=False)
    
    return vertex_bin, tri_bin


if __name__ == '__main__':
    v1, t1 = main(-200,0,0,0,0,0)
    v2, t2 = main(-200.0,0.0,0.0,0.0,-0.0,90)
    
    print(np.count_nonzero(t1))
    print(np.count_nonzero(np.logical_and(t1,t2)))
    
    t = np.count_nonzero(np.logical_and(t1,t2)) / np.count_nonzero(t1)
    print(t)
    # a = np.array([1,2,3,4,5])
    # b = np.array([1,2,1,2,1])
    # print(a[b])


    
    
    