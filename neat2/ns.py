from neat2.archive import archive
from scipy.spatial.distance import cdist
import numpy as np
import copy
class novelty_search(object):
    
    def __init__(self,config) -> None:
        self.threshold=config.novelty_threshold
        self.knn=config.knn
        self.archive=archive(config)
        self.timeout=0
        self.degrade_rate=config.degrade_rate
        self.upgrade_rate=config.upgrade_rate
    def eval_novelty(self,pop_off):
        
        arch=self.archive.data
        if len(arch)==0:
            arch=[]
        pop_off_novelty=pop_off+arch
        dismatrix=self.cal_knn_dis(pop_off,pop_off_novelty)
        novelties=[self.cal_novelty(dis,self.knn) for dis in dismatrix]
        
        num_add_gen=0
        # distribute the novelty 
        for i,g in enumerate(pop_off):
            g.novelty=novelties[i]
            # update the archive
            if g.novelty>self.threshold or len(self.archive)<self.archive.archive_len:
                self.update_archive(g)
                num_add_gen+=1
        self.adjust_threshold(num_add_gen)
        self.update_age()
        # print(f"threshold :{self.threshold} and num_add :{num_add_gen}")
        
    def cal_knn_dis(self,pop_off,pop_off_novelty):
        """
        p1=[[f1,f2]...] from pop_off
        p2=[[f1,f2]...] from pop_off_novelty        
        return dismatirx
        """
        # print(pop_off)
        p1=[ [g.fitness[0],g.fitness[1]] for g in pop_off]
        p2=[ [g.fitness[0],g.fitness[1]] for g in pop_off_novelty]
        dismatrix=cdist(p1,p2,'euclidean')
        return dismatrix
    
    def cal_novelty(self,dis,knn):
        novelty=0
        idx=np.argsort(dis)
        # +1 exclude the element itself
        novelty=np.mean(dis[idx[1:knn+1]])
        return novelty
        
    def update_archive(self,genome):
        self.archive.data.sort(key=lambda g : (g.age,-g.novelty))
        if len(self.archive)>=self.archive.archive_len:
            # remove the lowest one
            self.archive.data.pop()
            self.archive.data.append(copy.deepcopy(genome))
        else:
            self.archive.data.append(copy.deepcopy(genome))
            
            
    #TODO 自适应threshold
    def adjust_threshold(self,num_add):
        if num_add==0:
            self.timeout+=1
            if self.timeout>=5:
                self.threshold*=self.degrade_rate
                self.timeout=0
        if num_add>=3:
            self.threshold*=self.upgrade_rate
        
    def update_age(self):
        for g in self.archive.data:
            g.age+=1