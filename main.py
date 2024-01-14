import multiprocessing
import os
import pickle
import random

import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import neat2
import numpy as np
import tools.shape as shape
import tools.utils as utils
# import visualize
import fitness_f.period as fit_period
from tools.read_mesh import getmesh
import cv2

def square2parallel(input_xy):
    affine_matrix = np.array([[1.5, 0.5],   # 水平拉伸
                          [0, 1]])      # 垂直不变

    # 应用仿射变换
    output_xy = cv2.transform(input_xy.reshape(1, -1, 2), affine_matrix)
    print(output_xy[0].shape)
    return np.array(output_xy[0])


# 从左到右，从下到上
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
        input_xy[:, i] = utils.normalize(input_xy[:, i], 0, 1)  #[0,1]
    input_xy=square2parallel(input_xy)
    input_xy[:, 0]*=orig_size_xy[0]
    input_xy[:, 1]*=orig_size_xy[1]
    return input_xy


def get_load_support(pcd):
    load=[]
    support=[]
    for i,point in enumerate(pcd):
        if point[0] == 3 :
            if point[1]<=-0.8:
                load.append(i)

    for i, point in enumerate(pcd):
        if point[0] == -3 :
            if point[1] >= 0.8 or  point[1] <=-0.8:
                support.append(i)

    return load,support


def split2inside(output, pointcloud):
    n_x=shapex*2+1
    n_y=shapey*2+1
    pcd = pointcloud
    X = pcd[:, 0].copy()
    Y = pcd[:, 1].copy()
    trian_X = X.reshape(n_y, n_x).copy()
    trian_Y = Y.reshape(n_y, n_x).copy()
    a = output.reshape(n_y, n_x).copy()
    train_p_out = a
    if_inside = train_p_out > threshold
    inside_X = trian_X[if_inside]
    inside_Y = trian_Y[if_inside]
    inside = np.stack((inside_X.flatten(), inside_Y.flatten()), axis=1)
    split_parts = {}
    split_parts["inside"] = inside
    split_parts["all_square"] = a
    load,supp=get_load_support(pcd)
    split_parts['load'] = load
    split_parts['support'] = supp
    return split_parts


def eval_genome(genome, config):
    net = neat2.nn.FeedForwardNetwork.create(genome, config)
    outputs = []
    fitness=[]
    for point in pointcloud:
        output = net.activate(point)
        outputs.append(output)
    outputs = np.array(outputs)
    outputs = utils.scale(outputs)

    split = split2inside(outputs, pointcloud)
    outputs_square = split['all_square']

    Index, X, Y, Cat = shape.find_contour(a=outputs_square, thresh=threshold, pcd=pointcloud,shapex=shapex,shapey=shapey)
    x_values = X.flatten().copy()
    y_values = Y.flatten().copy()
    cat_values = Cat.flatten().copy()
    index_values = Index.flatten().copy()
    index_x_y_cat = np.concatenate(
        (index_values.reshape(-1, 1), x_values.reshape(-1, 1), y_values.reshape(-1, 1), cat_values.reshape(-1, 1)),
        axis=1)

    outside_tri=shape.get_outside_Tri(Tri, index_x_y_cat)
    mesh=getmesh(index_x_y_cat,outside_tri)
    # 求最大值？？
    f1,f2=fit_period.getfit(mesh)
    return [f1,f2]



def run_experiment(config_path, n_generations=100):
    config = neat2.Config(neat2.DefaultGenome, neat2.DefaultReproduction,config_path)
    p = neat2.Population(config)

    pe = neat2.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    best_genomes = p.run(pe.evaluate, n=n_generations)
    p1=[]
    p2=[]
    utils.check_and_create_directory("./output_final")
    for i,(k,g) in enumerate(best_genomes.items()):
        print(f"{k}: is{g.fitness}")
        p1.append(g.fitness[0])
        p2.append(g.fitness[1])
        with open(f'./output_final/best_genome_{i}.pkl', 'wb') as f:
            pickle.dump(g, f)
    bg=list(best_genomes.items())[0][1]
    print('\nBest genome: \n%s' % (bg))
    # 画pareto front
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    ax.scatter(p1, p2, color="green")
    ax.set_xlabel('f1', fontweight='bold')
    ax.set_ylabel('f2', fontweight='bold')
    ax.set_title(' Pareto front',fontweight='bold')  # 添加标题
    # 添加颜色条
    plt.show()
    plt.savefig(f"./output_final/pareto_front.png")
    plt.close()
    

orig_size_xy = (1, 1)
density = 10
n_generations =1
threshold = 0.5
# 要求是square
shapex = orig_size_xy[0]*density
shapey = orig_size_xy[1]*density
pointcloud = point_xy(shapex, shapey)
Tri = shape.triangulation(shapex, shapey)
if __name__ == '__main__':
    random_seed = 33
    random.seed(random_seed)
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'maze_config.ini')
    run_experiment(config_path, n_generations=n_generations)
