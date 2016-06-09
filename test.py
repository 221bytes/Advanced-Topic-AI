
import matplotlib.pyplot as plt
import GAMaxOneMatrix as MaxOneSolution

from sklearn import datasets, svm, metrics
import numpy as np
digits = datasets.load_digits()

def test(gamma):
    images_and_labels = list(zip(digits.images, digits.target))
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))

    classifier = svm.SVC(gamma=gamma)

    classifier.fit(data[:n_samples / 2], digits.target[:n_samples / 2])

    expected = digits.target[n_samples / 2:]
    predicted = classifier.predict(data[n_samples / 2:])

    arr = predicted - expected
    counts = np.count_nonzero(arr)

    res = 100 - float(counts)/float(len(predicted)) * 100

    return res

class GAClassifier(object):
    def __init__(self):
        self.prob_crossover = 0.6
        self.prob_mutation = 0.05
        self.population_size = 20
        self.iteration_limit = 50
        self.gene_size = 8
        self.nb_parameter = 1
        self.ga = MaxOneSolution.GAMaxOneMatrix()
        self.ga.max_one = self.nb_parameter * self.gene_size
        self.best_fitness = []
        np.set_printoptions(precision=15)

    def initial_population(self):
        return np.random.randint(10, size=(self.population_size, self.nb_parameter * self.gene_size))

    def sfm_fitness(self, population, iteration):
        fitness_population = np.empty(self.population_size)
        for individual in range(self.population_size):
            first_gene = population[individual][:self.gene_size]
            q = first_gene.astype(np.float)
            s = "0."
            for i in first_gene:
                s += str(i)
            print s
            fitness = test(float(s))
            print fitness
            fitness_population[individual] = fitness

        best =  np.max(fitness_population)
        print "Best fitness for iteration %d = %f" % (iteration, best)
        self.best_fitness.append(best)
        return fitness_population

    def run(self):
        self.ga.population_size = self.population_size
        self.ga.prob_mutation = self.prob_mutation
        self.ga.prob_crossover = self.prob_crossover
        self.ga.iteration_limit = self.iteration_limit
        population = self.initial_population()
        for i in range(0, self.iteration_limit, 1):
            print 'iteration %d' % i
            fits_pop = [self.sfm_fitness(population, i), population]
            index = np.argmax(fits_pop[0])
            # print self.gene_to_noise_params(fits_pop[1][index], True), fits_pop[0][index]
            population = self.ga.breed_population(fits_pop)
        return self.best_fitness

def start_graph():
    ga = GAClassifier()
    x_axis = ga.run()
    plt.plot(x_axis, label="best fitness")

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0., title='Iteration limit influence on GA')
    plt.xlabel('iterations')
    plt.ylabel('fitness')
    plt.show()

def start():
    ga = GAClassifier()
    ga.run()

start_graph()
