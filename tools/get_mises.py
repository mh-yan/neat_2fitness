import pickle
from matplotlib import pyplot as plt
import utils.data2txt as data2txt
import neat2
import math
import numpy
import utils.utils as utils

import fun1
import fun2
import numpy as np
import utils.shape as shape
import fun3


def point_xy(shapex, shapey):
    l=2*shapex+1
    w=2*shapey+1
    x = np.linspace(0, 1, l)
    y = np.linspace(0, 1, w)
    X = np.zeros((w, l))
    Y = np.zeros((w, l))

    X[:, :] = x
    Y[:, :] = y.reshape(-1, 1)
    input_xy = np.stack((X.flatten(), Y.flatten()), axis=1)
    # normalize the input_xyz
    for i in range(2):
        input_xy[:, i] = utils.normalize(input_xy[:, i], 0, 1)  # [-1, 1]
    input_xy[:, 0]*=orig_size_xy[0]
    input_xy[:, 1]*=orig_size_xy[1]
    return input_xy


config = neat2.Config(
    neat2.DefaultGenome, neat2.DefaultReproduction,
    'maze_config.ini')
import utils.read_mesh as read_mesh

orig_size_xy = (1, 1)
density = 45
threshold = 0.5

# 要求是square
shapex = orig_size_xy[0]*density
shapey = orig_size_xy[1]*density
pointcloud = point_xy(shapex, shapey)
Tri=shape.triangulation(shapex,shapey)

index=2
net=shape.load_net(f'./output_fun3_2000/best_genome_{index}.pkl',config)
outputs = []
fitness=[]
for point in pointcloud:
    output = net.activate(point)
    outputs.append(output)
outputs = np.array(outputs)
near = 1e-5
outputs = utils.scale(outputs)
# volumn=0
# for i,data in enumerate(pointcloud):
#     if outputs[i]<0.5:
#         volumn+=1
#     if data[0]<=-0.8:
#         outputs[i]=0.0
#     if data[1]<data[0]+0.3:
#         outputs[i]=1.
#     if data[1]>data[0]+0.3 and data[1]<=data[0]+0.6+near:
#         outputs[i]=0.0
#     if data[0]>-0.2 and data[1]>data[0]+0.6-near:
#         outputs[i]=1.
        
for i,data in enumerate(pointcloud):
        if data[0]<=-0.8:
            outputs[i]=0.0
        if data[1]<=data[0]+0.3-near or data[1]<=data[0]+0.3+near:
            outputs[i]=1.
        if data[1]>data[0]+0.3+near and data[1]<=data[0]+0.6+near:
            outputs[i]=0.0
        if data[0]>-0.2 and data[1]>data[0]+0.6-near:
            outputs[i]=1.
# split = main.split2inside(outputs, pointcloud)
# inside = split['inside']
# load = split['load']
# load_change = {key: [0, -100] for key in load}
# support = split['support']
# support_change = {key: [1, 1] for key in support}
n_x=shapex*2+1
n_y=shapey*2+1
outputs_square = outputs.reshape(n_y,n_x)

Index, X, Y, Cat = shape.find_contour(a=outputs_square, thresh=threshold, pcd=pointcloud,shapex=shapex,shapey=shapey)
x_values = X.flatten()
y_values = Y.flatten()
# x_values*=125
# y_values*=125
cat_values = Cat.flatten()
index_values = Index.flatten()
index_x_y_cat = np.concatenate(
    (index_values.reshape(-1, 1), x_values.reshape(-1, 1), y_values.reshape(-1, 1), cat_values.reshape(-1, 1)),
    axis=1)
# 上下加框
# volumn=0
# for i,data in enumerate(index_x_y_cat):
#     if data[3]==1:
#         volumn+=1
#     if data[2]<=-0.8 or data[2]>=0.8:
#         index_x_y_cat[i,-1]=1
#     if  data[1]>=1.8:
#         index_x_y_cat[i, -1] = 1
# 1,2  1是outside ，2是inside
# point + outside_tri
index_x_y_cat[:,1:-1]*=250
outside_tri=shape.get_outside_Tri(Tri, index_x_y_cat)
read_mesh.getmesh(index_x_y_cat[:,0:-1],outside_tri)
# shape.draw_shape(index_x_y_cat,outside_tri=outside_tri)
# f1,f2=fun1.fenics_fitness(index_x_y_cat,outside_tri)