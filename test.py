import sys


import pickle
from matplotlib import pyplot as plt
# import tools.data2txt as data2txt
import neat2
import math
import numpy
import tools.utils as utils
import numpy as np
import tools.shape as shape
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


orig_size_xy = (1, 1)
# 换密度plot
density = 20
threshold = 0.5

# 要求是square
shapex = orig_size_xy[0]*density
shapey = orig_size_xy[1]*density
pointcloud = point_xy(shapex, shapey)
Tri=shape.triangulation(shapex,shapey)


# the root directory of neural network 
path=f"./output_final"
path2=f"./output/output_gen2"
utils.plotall(config=config,path=path,thresh=threshold,pcd=pointcloud,Tri=Tri,shapex=shapex,shapey=shapey)

        
    

        

