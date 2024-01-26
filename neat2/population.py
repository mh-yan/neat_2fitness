"""Implements the core evolution algorithm."""
import time

from neat2.math_util import mean
from neat2.reporting import ReporterSet

import pickle
import tools.utils as utils
import math
import matplotlib.pyplot as plt
import numpy as np
from neat2.ns import novelty_search
class CompleteExtinctionException(Exception):
    pass


class Population(object):
    """
    This class implements the core evolution algorithm:
        1. Evaluate fitness of all genomes.
        2. Check to see if the termination criterion is satisfied; exit if it is.
        3. Generate the next generation from the current population.
        4. Partition the new generation into species based on genetic similarity.
        5. Go to 1.
    """

    def __init__(self, config,pcd, initial_state=None):
        self.reporters = ReporterSet()
        self.config = config
        self.pcd=pcd
        self.reproduction = config.reproduction_type(config.reproduction_config,self.reporters)
        self.ns_search=novelty_search(config)
        if initial_state is None:
            # Create a population from scratch, then partition into species.
            self.population = self.reproduction.create_new(config.genome_type,
                                                           config.genome_config,
                                                           config.pop_size)
            self.generation = 0
        else:
            self.population, self.generation = initial_state

        self.best_genomes = None

    def run(self, fitness_function, n=None):

        k = 0
        while n is None or k < n:
            k += 1
            print(f"===========begin iteration: {k}===========")
            
            self.population ,self.best_genomes= self.reproduction.reproduce(self.config,self.population,
                                                          self.config.pop_size, self.generation,fitness_function,self.pcd,self.ns_search)
            p1 = []
            p2 = []
            utils.check_and_create_directory(f"./output/output_gen{k}")
            for i,g in enumerate(self.population):
                p1.append(g.fitness[0]+g.fitness[1])
                p2.append(g.novelty)
                with open(f'./output/output_gen{k}/best_genome_{i}.pkl', 'wb') as f:
                    pickle.dump(g, f)
            fig=plt.figure(figsize=(10,10))
            ax = fig.add_subplot(111)
            ax.scatter(p1, p2, color="red")
            ax.set_title(' Pareto front',fontweight='bold')  # 添加标题
            ax.set_xlabel('f1', fontweight='bold')
            ax.set_ylabel('novelty', fontweight='bold')
            plt.show()
            plt.savefig(f"./output/output_gen{k}/pareto_front.png")
            plt.close()
            print("mean of the pareto front is :",np.mean([i.fitness[0] for i in self.best_genomes]),np.mean([i.novelty for i in self.best_genomes]))
            print("max fitness is :",max([i.fitness[0] for i in self.best_genomes]),max([i.fitness[0] for i in self.population]))
            self.generation += 1


        return self.best_genomes
