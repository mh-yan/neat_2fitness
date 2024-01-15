import numpy as np
import os
import matplotlib.pyplot as plt
import tools.shape as shape
def normalize(x, b=None, u=None):
    if b != None:
        x -= b
    else:
        x -= np.min(x)
    with np.errstate(divide='ignore', invalid='ignore'):
        if u != None:
            x /= u
        else:
            x /= np.max(x)
    x = np.nan_to_num(x)
    # adjust the range to [x1,x2]
    x *= 2
    x -= 1
    return x


def scale(x, min=0, max=1):
    # scale x to [min, max]
    x = np.nan_to_num(x)
    x = (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-10)
    return x


def check_and_create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        
def plotall(config,path,thresh,pcd,Tri,shapex,shapey):
    
        length_dir=len([f for f in os.listdir(path)])-1
        print("length_dir is :",length_dir)
        f1=[]
        f2=[]
        for i in range(length_dir):
            genome_path=f"{path}/best_genome_{i}.pkl"
            g=shape.getshape(genome_path,config,thresh,pcd,Tri,shapex,shapey)
