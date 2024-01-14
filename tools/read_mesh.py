# from dolfin import Mesh, Point, cpp,plot
import matplotlib.pyplot as plt
from dolfin import *
import numpy as np
import random
# # 使用示例
# file_path = "path/to/your/mesh.txt"
# prefix_to_find = "8 outside_tri"
def read_custom_txt_mesh(points,outside_tri):
    # 提取节点和单元信息
    num_nodes=points.shape[0]
    num_triangles=len(outside_tri)
    nodes = [Point(float(data[1]), float(data[2])) for data in points]
    triangles = [list(map(int, data)) for data in outside_tri]
    
    
   # 找到三角形中使用的所有节点的索引
    used_indices = set()
    for triangle in triangles:
        used_indices.update(triangle)

    # 创建一个映射旧节点索引到新节点索引的字典
    old_to_new_indices = {}
    filtered_nodes = []
    new_index = 0
    for old_index, node in enumerate(nodes):
        if old_index in used_indices:
            filtered_nodes.append(node)
            old_to_new_indices[old_index] = new_index
            new_index += 1

    # 更新三角形列表中的节点索引
    filtered_triangles = [[old_to_new_indices[idx] for idx in triangle] for triangle in triangles]
   
   
    match_index=None
    # 创建DOLFIN的网格对象
    mesh = Mesh()
    filtered_nodes=np.array(filtered_nodes)
    # 添加节点
    editor = cpp.mesh.MeshEditor()
    editor.open(mesh, "triangle", 2, 2)
    editor.init_vertices(filtered_nodes.shape[0])
    editor.init_cells(num_triangles)

    for i, point in enumerate(filtered_nodes):
        editor.add_vertex(i, point)

    # # 添加单元
    for i, triangle in enumerate(filtered_triangles):
        editor.add_cell(i, triangle)

    # 完成网格初始化
    editor.close()

    return mesh


def getmesh(points,outside_tri):
    mesh= read_custom_txt_mesh(points,outside_tri)
    # plt.figure()
    # plot(mesh)
    # plt.savefig(f'./mesh{random.random()}.png')
    
    # plt.close()
    return mesh