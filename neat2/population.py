"""Implements the core evolution algorithm."""
import time

from neat2.math_util import mean
from neat2.reporting import ReporterSet

import pickle

import math
import matplotlib.pyplot as plt
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

    def __init__(self, config, initial_state=None):
        self.reporters = ReporterSet()
        self.config = config
        self.reproduction = config.reproduction_type(config.reproduction_config,self.reporters)

        if initial_state is None:
            # Create a population from scratch, then partition into species.
            self.population = self.reproduction.create_new(config.genome_type,
                                                           config.genome_config,
                                                           config.pop_size)
            self.generation = 0
        else:
            self.population, self.generation = initial_state

        self.best_genomes = None

    # def add_reporter(self, reporter):
    #     self.reporters.add(reporter)
    #
    # def remove_reporter(self, reporter):
    #     self.reporters.remove(reporter)

    def run(self, fitness_function, n=None):

        k = 0
        while n is None or k < n:
            k += 1
            print("begin")
            # s=time.time()
            # Create the next generation from the current generation.
            self.population ,self.best_genomes= self.reproduction.reproduce(self.config,self.population,
                                                          self.config.pop_size, self.generation,fitness_function)
            if k==1:
                p1 = []
                p2 = []
                p3=[]
                for i, g in enumerate((list(self.best_genomes.items()))):
                    p1.append(g[1].fitness[0])
                    p2.append(g[1].fitness[1])
                    p3.append(g[1].fitness[2])
                    with open(f'./output_gen0/best_genome_gen0_{i}.pkl', 'wb') as f:
                        pickle.dump(g[1], f)
                fig=plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter3D(p1, p2, p3, color="red")
            print(list(self.best_genomes.items())[0][1].fitness)
            # print(time.time()-s)
            print(self.generation)

            self.generation += 1


        return self.best_genomes
