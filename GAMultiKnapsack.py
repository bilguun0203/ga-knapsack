import random
import numpy as np
import argparse

SILENT = False


class Chromosome:
    def __init__(self, chromosome_length):
        self.genes = np.random.choice([False, True], size=chromosome_length)
        self.fitness = 0
        self.weights = 0

    def calc_fitness(self, data, constraints):
        self.repair(data, constraints)
        fitness = np.sum(np.array(data[0])[self.genes])
        weights = []
        for i, w in enumerate(np.array(data[1])):
            weights.append(np.sum(w[self.genes]))
        self.fitness = fitness
        self.weights = weights
        return [self.fitness, self.weights]

    def repair(self, data, constraints):
        for i, w in enumerate(np.array(data[1])):
            weight = np.sum(w[self.genes])
            while weight > constraints[i]:
                ones = np.where(np.array(self.genes) == True)[0]
                self.genes[ones[random.randint(0, len(ones)-1)]] = False
                weight = np.sum(w[self.genes])
        return self

    def crossover(self, chromosome):
        offspring1 = Chromosome(len(self))
        offspring2 = Chromosome(len(chromosome))
        offspring1.genes = self.genes.copy()
        offspring2.genes = chromosome.genes.copy()
        pos = random.randint(0, len(offspring1)-1)
        tmp = offspring2.genes[pos:].copy()
        offspring2.genes[pos:], offspring1.genes[pos:] = offspring1.genes[pos:], tmp
        return offspring1, offspring2

    def mutation(self, probability):
        self.genes = [not g if random.random(
        ) <= probability else g for g in self.genes]
        return self

    def __len__(self):
        return len(self.genes)

    def __str__(self):
        return str(self.genes)


class Population:
    def __init__(self, data, population_size, chromosome_length, constraint_size, constraints):
        self.data = data
        self.constraints = constraints
        self.constraint_size = constraint_size
        self.population_size = population_size
        # Хромосомын үнэлгээ, жингүүд
        self.chr_fitness = np.zeros(self.population_size)
        self.chr_weights = np.zeros(self.population_size)
        # Шинээр үүссэн хромосомын үнэлгээ, жингүүд
        self.offs_fitness = np.zeros(self.population_size)
        self.offs_weights = np.zeros(self.population_size)
        self.chr_length = chromosome_length
        self.init_population()
        self.offsprings = []

    def init_population(self):
        self.chromosomes = [Chromosome(self.chr_length) for i in range(self.population_size)]
        return self

    def calc_fitness(self):
        self.chr_fitness = []
        self.chr_weights = []
        for ch in self.chromosomes:
            f, w = ch.calc_fitness(self.data, self.constraints)
            self.chr_fitness.append(f)
            self.chr_weights.append(w)
        return self

    def crossover(self, rate=0.85):
        nextgen = []
        while len(self.chromosomes) > 1:
            parent1 = self.chromosomes.pop(0)
            parent2 = self.chromosomes.pop(0)
            ng1, ng2 = parent1.crossover(parent2) if (
                random.random() <= rate) else (parent1, parent2)
            nextgen.append(ng1)
            nextgen.append(ng2)
        self.chromosomes = nextgen
        return self


def load(path, optimal_val=False):
    n = 0
    m = 0
    constraints = []
    F = []
    WEIGHTS = []
    constraints = []
    optimal = 0
    with open(path, 'r') as f:
        # Нөхцөл болон эд зүйлсийн тоог унших
        line = f.readline()
        items = line.split(' ')
        m = int(items[0])
        n = int(items[1])
        # Эд зүйлсийн үнэлгээг унших
        while len(F) < n:
            line = f.readline()
            if line != '':
                items = line.split(' ')
                for i in items:
                    F.append(float(i))
        # Нөхцлүүдийг унших
        while len(constraints) < m:
            line = f.readline()
            if line != '':
                items = line.split(' ')
                for i in items:
                    constraints.append(float(i))
        # Эд зүйлсийн weight-үүдийг унших
        for i in range(m):
            weights = []
            while len(weights) < n:
                line = f.readline()
                if line != '':
                    items = line.split(' ')
                    for i in items:
                        weights.append(float(i))
            WEIGHTS.append(weights)
        if optimal_val:
            line = f.readline()
            line = f.readline()
            optimal = float(line)
    return n, m, constraints, np.array((F, WEIGHTS)).T, optimal


def check_condition(population):
    fitness_v = np.unique(population.chr_fitness, return_counts=True)
    max_num = np.max(fitness_v[1])
    return max_num / len(population.chr_fitness) <= 0.9


def fit(population, generation, mutation_probability=0.01, selection='group', crossover_rate=0.85):
    global SILENT
    current_generation = 1
    pop.calc_fitness()
    best_i = np.argmax(population.chr_fitness)
    while current_generation <= generation and check_condition(population):
        # print('Generation', current_generation)
        selection = group_selection if selection == 'group' else roulette_selection
        population.chromosomes.sort(key=fitness_val, reverse=True)
        elits = []
        elits.append(population.chromosomes.pop(0))
        elits.append(population.chromosomes.pop(0))
        population.chromosomes = selection(population)
        population.crossover(rate=crossover_rate)
        population.chromosomes = [ind.mutation(mutation_probability) for ind in population.chromosomes]
        for e in elits:
            population.chromosomes.append(e)
        population.calc_fitness()
        current_generation += 1
        best_i = np.argmax(population.chr_fitness)
        if not SILENT:
            print(population.chromosomes[best_i].fitness)
    return population.chromosomes[best_i]


def fitness_val(ind):
        return ind.fitness


def roulette_selection(population):
    population.calc_fitness()
    selected_chromosomes = []
    fsum = sum(chromosome.fitness for chromosome in population.chromosomes)
    for i in range(len(population.chromosomes)):
        limit = random.randint(0, fsum)
        accsum = 0
        for chromosome in population.chromosomes:
            accsum += chromosome.fitness
            if accsum > limit:
                selected_chromosomes.append(chromosome)
                break
    return selected_chromosomes


def group_selection(population):
    selected_chromosomes = []
    total_chromosomes = len(population.chromosomes)
    for i in range(total_chromosomes):
        perc = random.randint(0, 100)
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
        if bottom >= top:
            bottom = top-1
        selected_chromosomes.append(
            population.chromosomes[random.randint(bottom, top)])
    return selected_chromosomes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Genetic Algorightm for 0-1 Knapsack problem')
    parser.add_argument('file_path', metavar='file_path',type=str, help='problem file path')
    parser.add_argument('-p', '--population', metavar='population_size', type=int,
                        default=100, help='population size (default: 100)')
    parser.add_argument('-g', '--generation', metavar='generation', type=int,
                        default=1000, help='max generation (default: 1000)')
    parser.add_argument('-m', '--mutation', metavar='mutation', type=float,
                        default=0.001, help='mutation probability (default: 0.1)')
    parser.add_argument('-s', '--selection', metavar='selection', type=str, choices=['group', 'roulette'],
                        default='group', help='selection algorithm (default: group)')
    parser.add_argument('-r', '--crossover_rate', metavar='crossover_rate', type=float,
                        default=0.85, help='crossover rate (default: 0.85)')
    parser.add_argument('--silent', action='store_true', help='silence output')
    parser.add_argument('--optimal', action='store_true', help='read optimal value')
    args = parser.parse_args()
    SILENT = args.silent
    N, M, constraints, data, optimal = load(
        args.file_path, optimal_val=args.optimal)
    if not SILENT:
        print('Initializing population...')
    pop = Population(data, args.population, N, M, constraints)
    if not SILENT:
        print('Fitting...')
    solution = fit(pop, args.generation, mutation_probability=(
        args.mutation if 0 <= args.mutation <= 1 else 0.001), selection=args.selection, crossover_rate=args.crossover_rate)
    if not SILENT:
        print('\nBest solution: ')
        print('\tGenes:', np.argwhere(solution.genes).tolist())
        print('\tFitness:', solution.fitness)
        print('\tWeight:', solution.weights)
        if args.optimal:
            print('\tFitness - Optimal:', (solution.fitness - optimal))
    else:
        if args.optimal:
            print("{}, {}".format(solution.fitness, (solution.fitness - optimal)))
        else:
            print(solution.fitness)
