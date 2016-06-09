import random
import numpy as np
import time
import matplotlib.pyplot as plt

__author__ = 'Alexandre Catalano'

x_axis = []


class GAMaxOneMatrix(object):
    def __init__(self):
        self.prob_crossover = 0.99607843137254903
        self.prob_mutation = 0.99607843137254903
        # self.prob_crossover = 0.6
        # self.prob_mutation = 0.05
        self.iteration_limit = 10000
        self.population_size = 100
        self.max_one = 192
        self.best_fit = []

    def initial_population(self):
        return np.random.randint(2, size=(self.population_size, self.max_one))

    def print_pop(self, population):
        print population

    def fitness(self, population):
        return np.sum(population, axis=1)

    def check_stop(self, fits_pop, turn):
        for i in range(len(fits_pop[0])):
            fit = fits_pop[0][i]
            if fit == self.max_one:
                print 'victory in %d iterations' % turn
                x_axis.append(turn)
                # print fits_pop[1][i]
                return True
        return False

    def selection(self, fits_pop):
        parent_pairs = []
        best_fit = np.max(fits_pop[0])
        self.best_fit.append(best_fit)
        for individual in range(0, self.population_size, 1):
            random_numbers = random.randrange(0, self.population_size)
            random_numbers_bis = random.randrange(0, self.population_size)
            if fits_pop[0][random_numbers] > fits_pop[0][random_numbers_bis]:
                parent_pairs.append(fits_pop[1][random_numbers])
            else:
                parent_pairs.append(fits_pop[1][random_numbers_bis])
        return parent_pairs

    def crossover(self, father, mother):
        cross_point0 = random.randrange(0, self.max_one)
        cross_point1 = random.randrange(cross_point0, self.max_one)
        kid0 = father.copy()
        kid1 = mother.copy()
        kid0[cross_point0: cross_point1] = mother[cross_point0: cross_point1]
        kid1[cross_point0: cross_point1] = father[cross_point0: cross_point1]
        ret = np.vstack((kid0, kid1))
        return ret

    def mutation(self, child):
        gene = random.randrange(0, len(child))
        if child[gene] == 0:
            child[gene] = 1
        else:
            child[gene] = 0
        return child

    def breed_population(self, fits_pop):
        parents = self.selection(fits_pop)
        next_population = np.empty((self.population_size, self.max_one), dtype=int)
        father = 0
        while father < self.population_size:
            if father + 1 == self.population_size:
                next_population[father] = parents[father]
                break
            mother = father + 1
            cross = random.uniform(0, 1) < self.prob_crossover
            if cross is True or cross is np.True_:
                children = self.crossover(parents[father], parents[mother])
            else:
                children = np.vstack((parents[father], parents[mother]))
            for child in children:
                mutate = random.uniform(0, 1) < self.prob_mutation
                if mutate is True or mutate is np.True_:
                    self.mutation(child)
                next_population[father:mother + 1] = children
            father += 2
        return next_population

    def run(self):
        population = self.initial_population()
        arr = np.empty(0)
        iteration = 0
        for iteration in range(self.iteration_limit):
            start = time.clock()
            fits_pop = [self.fitness(population), population]
            if self.check_stop(fits_pop, iteration):
                return iteration
                break
            population = self.breed_population(fits_pop)
            end = time.clock()
            arr = np.append(arr, (end - start))
        # print np.sum(arr) / len(arr)
        return iteration, self.best_fit


def start_graph():
    tests = [(1, 0.6), (1, 0.1), (1, 0.4), (0.99607843137254903, 0.066666666666666666)]
    arr = np.empty((10, 1000))
    for test in tests:
        prob_cross, prob_mutation = test
        for i in range(0, 10, 1):
            ga = GAMaxOneMatrix()
            ga.prob_crossover = prob_cross
            ga.prob_mutation = prob_mutation
            iteration, best_fit = ga.run()
            arr[i] = best_fit
        x_axis = np.mean(arr, axis=0)
        plt.plot(x_axis, label=test)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
    plt.show()

def start():
    ga = GAMaxOneMatrix()
    print ga.run()

# start_graph()
# start()
