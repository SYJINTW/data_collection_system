import numpy as np
import pandas as pd
import csv
import math

filename = 'test_cube'

def get_obj_file_datas():
    f = open(f'./obj_source/{filename}.OBJ', encoding="utf-8")
    vertex_id = 1
    vertex = []
    triangle_id = 1
    triangle = []
    for line in f.readlines():
        datas = line.split()
        # print(datas)
        if datas:
            if datas[0] == 'v':
                x = float(datas[1])
                y = float(datas[3])
                z = float(datas[2])
                vertex.append([vertex_id,x,y,z])
                vertex_id = vertex_id + 1
            elif datas[0] == 'f':
                v1 = int(datas[1].split('/')[0])
                v2 = int(datas[2].split('/')[0])
                v3 = int(datas[3].split('/')[0])
                triangle.append([triangle_id,v1,v2,v3])
                triangle_id = triangle_id + 1
    f.close
    df = pd.DataFrame(vertex)
    df.to_csv(f'./obj_source/{filename}_vertex.csv', header=['id','x','y','z'], index=False)
    df = pd.DataFrame(triangle)
    df.to_csv(f'./obj_source/{filename}_triangle.csv', header=['id','v1','v2','v3'], index=False)

    vertexs = np.array(vertex)
    triangles = np.array(triangle)

    return vertexs, triangles, len(vertex), len(triangle)

def read_vertex_triangle_csv():
    # vertex
    vertex = np.empty(4)
    with open(f'./obj_source/{filename}_vertex.csv', 'r') as csv_f:
        rows = csv.reader(csv_f)
        next(rows) # skip header
        for row in rows:
            # print(row)
            vertex = np.append(vertex, row, axis=0)
    print(vertex[0])

def cal_surface(v1, v2, v3):
    vector1 = v2-v1
    vector2 = v3-v1
    n_vector = np.cross(vector1, vector2)
    d = -np.dot(v1,n_vector)
    return np.append(n_vector, d)

def find_point_in_surface(surface, point, viewpoint):
    line_vector = viewpoint - point
    n_vector = np.array([surface[0], surface[1], surface[2]])
    line_dot_normal = np.dot(line_vector, n_vector)
    # no or infinite ans
    if line_dot_normal == 0:
        return False, np.empty(1)
    # one ans
    else:
        k = -(np.dot(point, n_vector)+surface[3])/line_dot_normal
        return True, point + k*line_vector

def triangle_cover(a,b,c,point):
    v1 = b-a
    vp1 = point-a
    v2 = c-b
    vp2 = point-b
    v3 = a-c
    vp3 = point-c
    n1 = np.cross(v1, vp1)
    n2 = np.cross(v2, vp2)
    n3 = np.cross(v3, vp3)
    # print(n1)
    # print(n2)
    # print(n3)
    if (n1*n2 >= 0).all():
        if (n1*n2*n3 >= 0).all():
            return True
    return False

def point_distance(p1, p2, viewpoint):
    v1 = p1-viewpoint
    v2 = p2-viewpoint
    return np.dot(v1,v1) > np.dot(v2,v2)



def test():
    viewpoint = np.array([-200,0,0])
    vertexs, triangles, NUM_OF_VERTEX, NUM_OF_TRIANGLE = get_obj_file_datas()

    print('Vertex: ', NUM_OF_VERTEX)
    print('Triangle: ', NUM_OF_TRIANGLE)
    
    vertexs_dict = {}
    for vertex in vertexs:
        vertexs_dict[vertex[0]] = vertex[1:]
    
    bit_vertex = np.ones(NUM_OF_VERTEX+1)
    bit_triangle = np.ones(NUM_OF_TRIANGLE+1)

    for triangle in triangles:
        if bit_vertex[triangle[1]] == 0 or bit_vertex[triangle[2]] == 0 or bit_vertex[triangle[3]] == 0:
            bit_triangle[triangle[0]] = 0
            print(f'delete_triangle {triangle[0]}')
        else: 
            for i in range(1, NUM_OF_VERTEX+1):
                if i not in triangle[1:] and bit_vertex[i] == 1:
                    point = vertexs_dict.get(i)
                    v1 = vertexs_dict.get(triangle[1])
                    # print('v1: ', v1)
                    v2 = vertexs_dict.get(triangle[2])
                    # print('v2: ', v2)
                    v3 = vertexs_dict.get(triangle[3])
                    # print('v3: ', v3)
                    surface = cal_surface(v1, v2, v3)
                    # print('surface: ', surface)
                    EXIST_ON_SURFACE, point_on_surface = find_point_in_surface(surface, point, viewpoint)
                    # print('point_on_surface: ', point_on_surface)
                    if EXIST_ON_SURFACE:
                        # if the point is in the triangle
                        if triangle_cover(v1,v2,v3,point_on_surface):
                            # check if the triangle or point is near to the viewpoint
                            if point_distance(point, point_on_surface, viewpoint):
                                bit_vertex[i] = 0
                                print(f'delete_vertex {i}')
    print(np.count_nonzero(bit_vertex))
    print(np.count_nonzero(bit_triangle))

    
                    

                
                 
    # p1 = np.array([0,0,0])
    # p2 = np.array([10,0,0])
    # p3 = np.array([0,10,0])
    # p = np.array([5,5,0])
    # if triangle_cover(p1,p2,p3,p):
    #     print('Cover')
    # else:
    #     print('No cover')

if __name__ == '__main__':
    test()

# if __name__ == '__main__':
#     vertexs, triangles = get_obj_file_datas()
    
