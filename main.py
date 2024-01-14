import multiprocessing
import os
import pickle
import random

import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import neat2
import numpy as np
import fun2
import fun3
import fitness_function
import shape
import fun1
import utils
# import visualize
import period as fit_period
from read_mesh import getmesh
# Todo 非方形点云
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
        input_xy[:, i] = utils.normalize(input_xy[:, i], 0, 1)  # [-1, 1]
    # input_xy[:, 0]*=orig_size_xy[0]
    # input_xy[:, 1]*=orig_size_xy[1]
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
    X = pcd[:, 0]
    Y = pcd[:, 1]
    trian_X = X.reshape(n_y, n_x)
    trian_Y = Y.reshape(n_y, n_x)
    a = output.reshape(n_y, n_x)
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
    near = 1e-5
    outputs = utils.scale(outputs)
    # outputs=outputs-2
    volumn=0
    # for i,data in enumerate(pointcloud):
    #     # if outputs[i]<0.5:
    #     #     volumn+=1
    #     if data[0]<=-0.8:
    #         outputs[i]=0.0
    #     if data[1]<=data[0]+0.3-near:
    #         outputs[i]=1.
    #     if data[1]>data[0]+0.3+near and data[1]<=data[0]+0.6+near:
    #         outputs[i]=0.0
    #     if data[0]>-0.2 and data[1]>data[0]+0.6-near:
    #         outputs[i]=1.
    
   
    # for i,data in enumerate(pointcloud):
    #     if data[1]<=-0.8 or data[1]>=0.8:
    #         outputs[i]=0
    #     if  data[0]>=1.8:
    #         outputs[i] = 0
            
    volumn=0
    for  i in range(pointcloud.shape[0]):
        if outputs[i]<0.5:
            volumn+=1

    split = split2inside(outputs, pointcloud)
    outputs_square = split['all_square']
    # outputs_square[5:15,5:15]=2
    Index, X, Y, Cat = shape.find_contour(a=outputs_square, thresh=threshold, pcd=pointcloud,shapex=shapex,shapey=shapey)
    x_values = X.flatten()
    y_values = Y.flatten()
    cat_values = Cat.flatten()
    index_values = Index.flatten()
    index_x_y_cat = np.concatenate(
        (index_values.reshape(-1, 1), x_values.reshape(-1, 1), y_values.reshape(-1, 1), cat_values.reshape(-1, 1)),
        axis=1)

    outside_tri=shape.get_outside_Tri(Tri, index_x_y_cat)
    # f1,f2=fun3.fenics_fitness(index_x_y_cat,outside_tri)
    mesh=getmesh(index_x_y_cat,outside_tri)
    f1,f2=fit_period.getfit(mesh)
    return [volumn,f2,f1]



def run_experiment(config_path, n_generations=100):
    config = neat2.Config(neat2.DefaultGenome, neat2.DefaultReproduction,config_path)
    p = neat2.Population(config)
    # p.add_reporter(neat2.StdOutReporter(True))
    # stats = neat2.StatisticsReporter()
    # p.add_reporter(stats)
    pe = neat2.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    best_genomes = p.run(pe.evaluate, n=n_generations)
    p1=[]
    p2=[]
    p3=[]
    for i,(k,g) in enumerate(best_genomes.items()):
        print(f"{k}: is{g.fitness}")
        p1.append(g.fitness[0])
        p2.append(g.fitness[1])
        p3.append(g.fitness[2])
        with open(f'./output/best_genome_{i}.pkl', 'wb') as f:
            pickle.dump(g, f)
    bg=list(best_genomes.items())[0][1]
    print('\nBest genome: \n%s' % (bg))
 
    # for i,g in enumerate((list(best_genomes.items()))):


    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(p1, p2, p3, color="green")
    ax.set_xlabel('volumn', fontweight='bold')
    ax.set_ylabel('strain energy', fontweight='bold')
    ax.set_zlabel('mises stresse', fontweight='bold')
    # 添加颜色条
    plt.show()
    node_names = {-1: 'X', -2: 'Y', 0: 'Output'}
    # visualize.draw_net(config, bg, True, node_names=node_names)
    # visualize.plot_stats(stats, ylog=False, view=True)
    # visualize.plot_species(stats, view=True)
    # shape.getshape('./output/best_genome_0.pkl', config, threshold, pointcloud, Tri,shapex,shapey)
    # g_best_ever = stats.best_genome()
    # with open('./output/best_genome_ever.pkl', 'wb') as f:
    #     pickle.dump(g_best_ever, f)


# scale xy mask fun

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
