"""
Runs evaluation functions in parallel subprocesses
in order to evaluate multiple genomes at once.
"""
from multiprocessing import Pool


class ParallelEvaluator(object):
    def __init__(self, num_workers, eval_function, timeout=None, maxtasksperchild=None):

        # Todo multifitness？？
        self.eval_function = eval_function
        self.timeout = timeout
        self.pool = Pool(processes=num_workers, maxtasksperchild=maxtasksperchild)

    def __del__(self):
        self.pool.close()
        self.pool.join()
        self.pool.terminate()

    def evaluate(self, genomes, config):
        jobs = []
        for genome in genomes:
            jobs.append(self.pool.apply_async(self.eval_function, (genome, config)))


        # Todo need to change！！
        # assign the fitness back to each genome
        for job, genome in zip(jobs, genomes):
            fitness,outputs=job.get(timeout=self.timeout)
            genome.fitness = fitness
            genome.outputs=outputs
