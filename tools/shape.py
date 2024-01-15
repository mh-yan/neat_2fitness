import math
import pickle
import tools.data2txt as data2txt
import neat2
import  numpy as np
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
import tools.utils as utils

def triangulation(shapex,shapey):
    
    
    num_point_x=2*shapex+1
    num_point_y=2*shapey+1
    # num_squares_x = int((shapex - 1) / 2)
    # num_squares_y = int((shapey - 1) / 2)
    num_squares = int(shapex * shapey)
    Tri = np.zeros((num_squares * 8, 3))
    Index = np.zeros((num_point_y, num_point_x))
    n=0
    k = 0
    for i in range(num_point_y):
        for j in range(num_point_x):
            Index[i, j] = n
            n += 1
    for ii in range(shapey):
        for jj in range(shapex):
            # i,j is the index of point which is the left top of the square
            i = ii * 2
            j = jj * 2

            # ====================画三角形
            Tri[k, :] = [Index[i, j], Index[i + 1, j], Index[i, j + 1]]
            Tri[k + 1, :] = [Index[i + 1, j], Index[i + 1, j + 1], Index[i, j + 1]]
            Tri[k + 2, :] = [Index[i, j + 1], Index[i + 1, j + 1], Index[i + 1, j + 2]]
            Tri[k + 3, :] = [Index[i, j + 2], Index[i, j + 1], Index[i + 1, j + 2]]
            Tri[k + 4, :] = [Index[i + 1, j], Index[i + 2, j], Index[i + 2, j + 1]]
            Tri[k + 5, :] = [Index[i + 1, j], Index[i + 2, j + 1], Index[i + 1, j + 1]]
            Tri[k + 6, :] = [Index[i + 1, j + 1], Index[i + 2, j + 1], Index[i + 1, j + 2]]
            Tri[k + 7, :] = [Index[i + 2, j + 1], Index[i + 2, j + 2], Index[i + 1, j + 2]]
            k += 8
    return Tri
def find_contour(a, thresh, pcd,shapex,shapey):
    num_point_x=2*shapex+1
    num_point_y=2*shapey+1
    # num_squares_x = int((a.shape[1] - 1) / 2)
    # num_squares_y = int((a.shape[0] - 1) / 2)
    num_squares = int(shapex * shapey)
    X = pcd[:, 0].reshape(a.shape[0], a.shape[1]).copy()  # 先横向再纵向
    Y = pcd[:, 1].reshape(a.shape[0], a.shape[1]).copy()
    Index = np.zeros((a.shape[0], a.shape[1]))
    Cat = (a > thresh) + 1
    # create index 先横向再纵向
    n = 0  # 点
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            Index[i, j] = n
            n += 1

    new_a = np.copy(a)
    new_a[a > thresh] = 1
    new_a[a < thresh] = -1
    l = new_a[:, 0:-2]
    r = new_a[:, 2:]
    t = new_a[0:-2, :]
    b = new_a[2:, :]
    flag_x = t * b
    flag_y = l * r
    k = 0
    a = a - thresh
    min_r = 0.2
    max_r = 0.8
    # ii,jj is the index of square
    for ii in range(shapey):
        for jj in range(shapex):

            # i,j is the index of point which is the left top of the square
            i = ii * 2
            j = jj * 2

            # ========================画边框
            # p1
            if Cat[i,j]==Cat[i,j+2]:
                Cat[i,j+1]=Cat[i,j]
            if Cat[i+2,j]==Cat[i+2,j+2]:
                Cat[i+2,j+1]=Cat[i+2,j+2]
            if Cat[i,j]==Cat[i+2,j]:
                Cat[i+1,j]=Cat[i+2,j]
            if Cat[i,j+2]==Cat[i+2,j+2]:
                Cat[i+1,j+2]=Cat[i+2,j+2]
            
            if Cat[i,j]==Cat[i,j+2]==Cat[i+2,j]==Cat[i+2,j+2]:
                Cat[i+1,j+1]=Cat[i,j]
            
            # p2
            num_1=0
            num_2=0
            if Cat[i,j]==1:
                num_1+=1
            else:
                num_2+=1
            if Cat[i,j+2]==1:
                num_1+=1
            else:
                num_2+=1
            if Cat[i+2,j]==1:
                num_1+=1
            else:
                num_2+=1
            if Cat[i+2,j+2]==1:
                num_1+=1
            else:
                num_2+=1
            
            if num_1==3:
                Cat[i+1,j+1]=1
            if num_2==3:
                Cat[i+1,j+1]=2
                
            # 计算实际的i，j的x，y
            if flag_y[i, j] == -1:
                rho = np.abs(a[i, j]) / (np.abs(a[i, j]) + np.abs(a[i, j + 2]))
                rho = min(max_r, max(min_r, rho))
                if np.isnan(rho).any():
                    print("flag_y[i, j]", np.abs(a[i, j]), np.abs(a[i, j]), np.abs(a[i + 2, j]))

                X[i, j + 1] = (1 - rho) * X[i, j] + rho * X[i, j + 2]
                Cat[i, j + 1] = 0  # mark as boundary

            if flag_y[i + 2, j] == -1:
                rho = np.abs(a[i + 2, j]) / (np.abs(a[i + 2, j]) + np.abs(a[i + 2, j + 2]))
                rho = min(max_r, max(min_r, rho))

                if np.isnan(rho).any():
                    print("flag_y[i, j + 2]", abs(a[i, j + 2]), abs(a[i, j + 2]), abs(a[i + 2, j + 2]))

                X[i + 2, j + 1] = (1 - rho) * X[i + 2, j] + rho * X[i + 2, j + 2]
                Cat[i + 2, j + 1] = 0  # mark as boundary

            if flag_x[i, j] == -1:
                rho = np.abs(a[i, j]) / (np.abs(a[i, j]) + np.abs(a[i + 2, j]))
                rho = min(max_r, max(min_r, rho))

                if np.isnan(rho).any():
                    print(" flag_x[i, j]", abs(a[i, j]), abs(a[i, j]), abs(a[i, j + 2]))

                Y[i + 1, j] = (1 - rho) * Y[i, j] + rho * Y[i + 2, j]
                Cat[i + 1, j] = 0  # mark as boundary

            if flag_x[i, j + 2] == -1:
                rho = np.abs(a[i, j + 2]) / (np.abs(a[i, j + 2]) + np.abs(a[i + 2, j + 2]))
                rho = min(max_r, max(min_r, rho))

                if np.isnan(rho).any():
                    print("flag_x[i + 2, j]", abs(a[i + 2, j]), abs(a[i + 2, j]), abs(a[i + 2, j + 2]))

                Y[i + 1, j + 2] = (1 - rho) * Y[i, j + 2] + rho * Y[i + 2, j + 2]
                Cat[i + 1, j + 2] = 0  # mark as boundary

            if Cat[i, j + 1] + Cat[i + 2, j + 1] == 0  :  # 上下是边界
                X[i + 1, j + 1] = (X[i, j + 1] + X[i + 2, j + 1]) * 0.5
                Cat[i + 1, j + 1] = 0

            if Cat[i + 1, j] + Cat[i + 1, j + 2] == 0 :
                Y[i + 1, j + 1] = (Y[i + 1, j] + Y[i + 1, j + 2]) * 0.5
                Cat[i + 1, j + 1] = 0

    if np.isnan(X).any():
        print("x nan")
    if np.isnan(Y).any():
        print("y nan")
    return Index, X, Y, Cat
def get_tri_cat(Tri,index_x_y_cat):
    cat=[]
    for i, tri in enumerate(Tri):
        flag=0
        for node in tri:
            # Todo 下标回去
            if (index_x_y_cat[int(node), -1] == 2):
                flag=2
                break
                # outside_tri.append(tri)
                # break
        if flag==0:
            cat.append(1)
        else:
            cat.append(2)
    return cat

def get_outside_Tri(Tri,index_x_y_cat):
    outside_tri=[]
    for i, tri in enumerate(Tri):
        flag=0
        for node in tri:
            # Todo 下标回去
            if (index_x_y_cat[int(node), -1] == 2):
                flag=1
                break
                # outside_tri.append(tri)
                # break
        if flag==0:
            outside_tri.append(tri)
    return outside_tri

def split_tri(Tri,index_x_y_cat):
    
    tri_cat=[]
    for i, tri in enumerate(Tri):
        flag = 2
        for node in tri:
            # Todo 下标回去
            if (index_x_y_cat[int(node), -1] == 1):
                flag = 1
                tri_cat.append(flag)
                break
        if flag == 2:
            tri_cat.append(flag)
    return tri_cat
def load_net(path,config):
    with open(f'{path}','rb') as f:
        genome=pickle.load(f)
        net=neat2.nn.FeedForwardNetwork.create(genome,config)
    return net

def getshape(path,config,thresh,pcd,Tri,shapex,shapey,save_name=None):
    net=load_net(path,config)
    outputs = []
    for point in pcd:
        output = net.activate(point)
        outputs.append(output)
    outputs = np.array(outputs)
    outputs=utils.scale(outputs)
    near = 1e-11

    outputs=outputs.reshape(2*shapey+1,2*shapex+1)
    Index, X, Y, Cat=find_contour(outputs,thresh,pcd,shapex,shapey)

    x_values = X.flatten()
    y_values = Y.flatten()
    # scale
    # x_values*=250
    # y_values*=250
    cat_values = Cat.flatten()
    index_values = Index.flatten()

    # 假设tri是包含三角形索引的数组
    # 假设a是包含索引、x坐标、y坐标和类别的数组
    index_x_y_cat = np.concatenate(
        (index_values.reshape(-1, 1), x_values.reshape(-1, 1), y_values.reshape(-1, 1), cat_values.reshape(-1, 1)),
        axis=1)
    triangles = Tri
    indices = index_x_y_cat[:, 0]
    x_coords = index_x_y_cat[:, 1]
    y_coords = index_x_y_cat[:, 2]
    cat=get_tri_cat(Tri,index_x_y_cat)
    plt.figure(figsize=(9, 9))
    plt.triplot(x_coords, y_coords, triangles, '-', lw=0.1)
    plt.tripcolor(x_coords, y_coords, triangles,facecolors=cat, edgecolors='k',cmap='PuBu_r')
    plt.show()
    path_str=path.split(".")[1]
    print("path is :",path_str)
    plt.savefig(f'.{path_str}.png')
    plt.close()
    

def draw_shape(points,outside_tri):
    for triangle_indices in outside_tri:
        # 获取三角形的三个顶点坐标
        x_triangle = points[triangle_indices.astype(int),1]
        y_triangle = points[triangle_indices.astype(int),2]

        # 添加第一个点到最后一个点的线段
        x_triangle = np.append(x_triangle, x_triangle[0])
        y_triangle = np.append(y_triangle, y_triangle[0])

        # 画连接三个点的线段 y,x互换
        plt.plot(x_triangle, y_triangle, marker='o', linestyle='-', markersize=0.001, color='gray')
