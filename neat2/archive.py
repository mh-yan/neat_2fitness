
import os
import pickle as pkl
import numpy as np
from collections import deque

class archive(object):
    
    def __init__(self,config) -> None:
        self.data=[]
        self.archive_len=config.archive_len
        self.min_novelty=0
    def __len__(self):
        return len(self.data)
    
    def __iter__(self):
        return self.data.__iter__()
    
    def __next__(self):
        return self.data.__next__()
    
    def __getitem__(self, item):
        return self.data[item]
    
    
    def update(self,data):  
        """
        data is deepcopy set of genoems
        """    
        
        return 
    
    
    def store(self,data):
        self.data.append(data)
        
    # def get_min_novelty(self):
    #     nv=[g.novelty for g in self.data]
    #     self.min_novelty=min(nv)
    #     print(f"the min novelty in archive is {self.min_novelty}")