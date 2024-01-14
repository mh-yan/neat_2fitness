import pickle
from matplotlib import pyplot as plt
import data2txt
import neat2
import math
import numpy
import utils
import numpy as np
import shape
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
density = 20
threshold = 0.5

# 要求是square
shapex = orig_size_xy[0]*density
shapey = orig_size_xy[1]*density
pointcloud = point_xy(shapex, shapey)
Tri=shape.triangulation(shapex,shapey)




# f1=[]
# f2=[]
# f3=[]
# for i in range(0,84,8):
#     with open(f'./output_fun3/best_genome_{i}.pkl', 'rb') as f:
#         g=pickle.load(f)
#         f1.append(g.fitness[0])
#         f2.append(g.fitness[1])
#         f3.append(g.fitness[2])
#         print(g.fitness,i)

# plt.legend()
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(f1,f2,f3,label='final front',zorder=2,alpha=0.9)
# ax.set_xlabel('volumn')
# ax.set_ylabel('total_strain_energy')
# ax.set_zlabel('max_vm_stress')

# # # =======================================画3d
# f1=[]
# f2=[]
# f3=[]
# for i in range(26):
#     with open(f'./output_gen0_fun3/best_genome_gen0_{i}.pkl', 'rb') as f:
#         g=pickle.load(f)
#         f1.append(g.fitness[0])
#         f2.append(g.fitness[1])
#         f3.append(g.fitness[2])
#         print(g.fitness,i)
# ax.scatter(f1,f2,f3,label='first front',c='r',alpha=0.5,zorder=1)
# a_f1=np.mean(f1)
# a_f2=np.mean(f2)
# a_f3=np.mean(f3)
# print("avg_gen0:",a_f1,a_f2,a_f3)


# 11 5 12 20
# 3 4 5
# plt.savefig('./final_front.png')
# for index in range(30):
# output_fun1_5000
index=5
# with open(f'./output_gen0/best_genome_gen0_{index}.pkl','rb') as f:
#         genome=pickle.load(f)
#         print(genome.fitness)
# # plt.figure(figsize=(26,13))
# # tail=f""
shape.getshape(f'/output/best_genome_{index}', config, threshold, pointcloud, Tri,shapex,shapey)
plt.savefig(f"./output/b{index}.png")

        

        

