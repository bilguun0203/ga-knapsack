import random
import numpy as np
import argparse

SILENT = False


class Chromosome:
    def __init__(self, genes_length):
        self.genes = np.random.choice([False, True], size=genes_length)
        self.fitness = 0
        self.weight = 0
    def calc_fitness(self, data, criterion):
        fitness = np.sum(data[self.genes], axis=0)
        while fitness[1] > criterion:
            ones = np.where(np.array(self.genes) == True)[0]
            self.genes[ones[random.randint(0, len(ones)-1)]] = False
            fitness = np.sum(data[self.genes], axis=0)
        self.fitness = fitness[0]
        self.weight = fitness[1]
        return [self.fitness, self.weight]

    def crossover(self, individual, pos):
        offspring1 = Chromosome(len(self.genes))
        offspring2 = Chromosome(len(individual.genes))
        offspring1.genes = self.genes.copy()
        offspring2.genes = individual.genes.copy()
        tmp = offspring2.genes[pos:].copy()
        offspring2.genes[pos:], offspring1.genes[pos:] = offspring1.genes[pos:], tmp
        return offspring1, offspring2

    def mutation(self, probability):
        self.genes = [not g if random.random(
        ) <= probability else g for g in self.genes]
        return self


class Population:
    def __init__(self, data, population_size, genes_length, criterion):
        self.data = data
        self.criterion = criterion
        self.population_size = population_size
        self.chr_fitness = np.zeros(self.population_size)
        self.chr_weight = np.zeros(self.population_size)
        self.offs_fitness = np.zeros(self.population_size)
        self.offs_weight = np.zeros(self.population_size)
        self.fittest_chromosomes = []
        self.genes_length = genes_length
        self.init_population(genes_length)
        self.offsprings = []

    def init_population(self, genes_length):
        self.chromosomes = [Chromosome(genes_length)
                            for i in range(self.population_size)]
        return self

    def calc_fitness(self, offspring=False):
        fitness = []
        weight = []
        chromosomes = self.offsprings if offspring else self.chromosomes
        for i in chromosomes:
            f, v = i.calc_fitness(self.data, self.criterion)
            fitness.append(f)
            weight.append(v)
        if offspring:
            self.offs_fitness = fitness
            self.offs_weight = weight
        else:
            self.chr_fitness = fitness
            self.chr_weight = weight
        return self

    def get_fittest(self):
        self.calc_fitness()
        return self.chromosomes[np.argsort(self.chr_fitness)]

    def crossover(self):
        self.offsprings = []
        for i in range(int(self.population_size/2)):
            o1, o2 = self.chromosomes[random.randint(0, self.population_size-1)].crossover(
                self.chromosomes[random.randint(0, self.population_size-1)], int(self.genes_length/2))
            self.offsprings.append(o1)
            self.offsprings.append(o2)
        return self


def load(path):
    n = 0
    w = 0
    F = []
    W = []
    with open(path, 'r') as f:
        line = f.readline()
        n = int(line)
        while len(F) < n:
            line = f.readline()
            if line != '':
                items = line.split(' ')
                for i in items:
                    F.append(float(i))
        line = f.readline()
        w = int(line)
        while len(W) < n:
            line = f.readline()
            if line != '':
                items = line.split(' ')
                for i in items:
                    W.append(float(i))
    return n, w, np.array((F, W)).T


def generate_data(n, wrange, vrange):
    f = np.random.randint(1, wrange, size=n)
    w = np.random.randint(1, vrange, size=n)
    return np.array((f, w)).T


def check_condition(population):
    fitness_v = np.unique(population.chr_fitness, return_counts=True)
    max_num = np.max(fitness_v[1])
    return max_num / len(population.chr_fitness) <= 0.9


def fit(population, generation, mutation_probability=0.01, selection='group'):
    global SILENT
    current_generation = 1
    pop.calc_fitness()
    best_i = np.argmax(population.chr_fitness)
    while current_generation <= generation and check_condition(population):
        # print('Generation', current_generation)
        selection = group_selection if selection == 'group' else roulette_selection
        population.chromosomes = selection(population)
        population.calc_fitness()
        population.crossover()
        population.offsprings = [ind.mutation(
            mutation_probability) for ind in population.offsprings]
        population.calc_fitness(offspring=True)
        current_generation += 1
        best_i = np.argmax(population.chr_fitness)
        if not SILENT:
            print(population.chromosomes[best_i].fitness)
    return population.chromosomes[best_i]


def fitness_val(ind):
        return ind.fitness


def roulette_selection(population):
    chromosomes = population.chromosomes + population.offsprings
    chromosomes.sort(key=fitness_val, reverse=True)
    selected_chromosomes = []
    selected_chromosomes.append(chromosomes.pop(0))
    selected_chromosomes.append(chromosomes.pop(0))
    fsum = sum(ind.fitness for ind in chromosomes)
    for i in range(population.population_size-2):
        limit = random.randint(0, fsum)
        accsum = 0
        for ind in chromosomes:
            accsum += ind.fitness
            if accsum > limit:
                selected_chromosomes.append(ind)
                break
    return selected_chromosomes



def group_selection(population):
    chromosomes = population.chromosomes + population.offsprings
    chromosomes.sort(key=fitness_val, reverse=True)
    selected_chromosomes = []
    selected_chromosomes.append(chromosomes.pop(0))
    selected_chromosomes.append(chromosomes.pop(0))
    total_chromosomes = len(chromosomes)
    for i in range(population.population_size-2):
        perc = random.randint(0, 99)
        bottom = 0
        top = int(total_chromosomes*0.5)
        if perc >= 50 and perc < 80:
            bottom = top + 1
            top = int(total_chromosomes*0.8)
        elif perc >= 80 and perc < 95:
            bottom = int(total_chromosomes*0.8) + 1
            top = int(total_chromosomes*0.95)
        elif perc >= 95:
            bottom = int(total_chromosomes*0.95) + 1
            top = total_chromosomes - 1
        selected_chromosomes.append(chromosomes[random.randint(bottom, top)])
    return selected_chromosomes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Genetic Algorightm for 0-1 Knapsack problem')
    parser.add_argument('file_path', metavar='file_path', type=str, help='problem file path')
    parser.add_argument('-p', '--population', metavar='population_size', type=int,
                        default=100, help='population size (default: 100)')
    parser.add_argument('-g', '--generation', metavar='generation', type=int,
                        default=1000, help='max generation (default: 1000)')
    parser.add_argument('-m', '--mutation', metavar='mutation', type=float,
                        default=0.1, help='mutation probability (default: 0.1)')
    parser.add_argument('-s', '--selection', metavar='selection', type=str, choices=['group', 'roulette'],
                        default='group', help='selection algorithm (default: group)')
    parser.add_argument('--silent', action='store_true', help='silence output')
    args = parser.parse_args()
    SILENT = args.silent
    N, W, data = load(args.file_path)
    if not SILENT:
        print('Initializing population...')
    pop=Population(data, args.population, N, W)
    if not SILENT:
        print('Fitting...')
    solution = fit(pop, args.generation, mutation_probability=(args.mutation if 0 <= args.mutation <= 1 else 0.1), selection=args.selection)
    if not SILENT:
        print('\nBest solution: ')
        print('\tGenes:', np.argwhere(solution.genes).tolist())
        print('\tFitness:', solution.fitness)
        print('\tWeight:', solution.weight)
    else:
        print(solution.fitness)
