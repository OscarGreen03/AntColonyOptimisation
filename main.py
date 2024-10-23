from multiprocessing import Pool
from random import uniform
from bisect import bisect_left
import numpy as np
from tabulate import tabulate
class AntColony:
    def __init__(self):
        self.start = []
        self.graph = []
        self.bin_count = 0
        self.item_count = 0

    def evaporate(self, e=0.9):
        self.graph *= e
        self.start *= e


    def set_bin_graph(self, bin_count, item_count):
        self.graph = np.random.rand(item_count, bin_count ,bin_count)
        self.start = np.random.rand(bin_count)
        self.bin_count = bin_count
        self.item_count = item_count
        #self.graph = [[] for _ in range(item_count)]
        #for i in range(len(self.graph)):
        #    self.graph[i] = [np.random.rand(bin_count) for _ in range(bin_count)]

    def get_next_step(self, current_item, current_bin):
        if current_bin == "START":
            weights = self.start
        elif current_item == self.item_count:
            return "END"
        else:
            weights = self.graph[current_item - 1, current_bin]
        total_pheromone = np.sum(weights)
        probabilities = weights / total_pheromone
        choice = np.random.choice(self.bin_count, p=probabilities)
        return choice


        '''if current_bin == "START":
            weights = self.start
            cum_weight = np.cumsum(self.start)
            choice = uniform(0, cum_weight[-1])
            item_index = np.searchsorted(cum_weight, choice)
            return item_index
        elif current_item == self.item_count:
            return "END"

        # get the values at the item and bin location
        weights = self.graph[current_item, current_bin]
        cum_weights = np.cumsum(weights)
        choice = uniform(0, cum_weights[-1])
        item_index = np.searchsorted(cum_weights, choice)
        return item_index'''



    def get_new_path(self):
        path = ["START"]
        while path[-1] != "END":
            path.append(self.get_next_step(len(path) - 1, path[-1]))
        return path

    def evaluate_fitness(self, path, items):
        bins = np.zeros(self.bin_count)
        path = path[1:-1]
        for i in range(len(path)):
            #print(f'Putting item {i} {items[i]} into bin {path[i]}')
            bins[path[i]] += items[i]


        return max(bins) - min(bins)


    def update_pheremones(self, path, fitness):
        pheromone_amount = 100/fitness


        self.start[path[1]] += pheromone_amount

        for i in range(1, len(path) -1):
            if path[i+1] == "END":
                break

            current_bin = path[i]
            next_bin = path[i+1]

            self.graph[i-1, current_bin, next_bin] += pheromone_amount



def BPP(bin_count, items, iterations, p, e, colony=None):
    if colony is None:
        colony = AntColony()
        colony.set_bin_graph(bin_count, len(items))

    best_path = []
    best_fitness = float('inf')

    for i in range(iterations):

        paths = [colony.get_new_path() for _ in range(p)]
        for path in paths:
            fitness = colony.evaluate_fitness(path, items)
            colony.update_pheremones(path, fitness)

            if fitness < best_fitness:
                best_fitness = fitness
                best_path = path
                #print(f"Generation {i} found with fitness: {fitness}")
        colony.evaporate(e)

    ## final generation of paths
    best_path = []
    best_fitness = float('inf')
    for path in paths:
        fitness = colony.evaluate_fitness(path, items)
        if fitness < best_fitness:
            best_fitness = fitness
            best_path = path
    return best_fitness







# main array containing each column of the problem as an item in the array
# each item in this main array is an array of the bins and their weights to each other bin
# this array cont
# [start->col1, col1 , col2   , col3 ,..., coln-1->end]
# col1 = [bin1, bin2, bin3]
# bin1 = [nextb1, nextb2, nextb3, nextb4]




# each row is its index
#



#BPP(10, [20, 30, 40, 30, 20, 10, 20, 30, 14, 64, 12, 34, 23, 63, 36, 28], 1000, 100, 0.9)

if __name__ == '__main__':
    tests = [
        (100, 0.9),
        (100, 0.6),
        (10, 0.9),
        (10, 0.6)
    ]
    items_1 = list(range(1, 501))
    items_2 = [(i ** 2) / 2 for i in range(1, 501)]

    for bin_count in [10, 50]:
        for test in tests:
            print("------------------")
            print(f"p = {test[0]}" + " " * (4 - len(str(test[0]))) + f"with e = {test[1]}")
            iterations = 10000//test[0]
            results = []

            #colony = AntColony()
            #colony.set_bin_graph(10, len(items_1))

            for i in range(5):
                print(f"{i + 1}", end="")

                results.append(BPP(bin_count, items_1, iterations, test[0], test[1], colony=None))

            print(f"\nResults b={bin_count}, p={test[0]}, e={test[1]}")
            for result in results:
                print(result)
            print("------------------")



    for test in tests:
        print("------------------")
        print(f"p = {test[0]}" + " " * (4 - len(str(test[0]))) + f"with e = {test[1]}")

        iterations = 10000//test[0]

        best_fitness, best_path  = BPP(10, items_1, iterations, test[0], test[1])
        # print("Best path is: " + str(best_path))
        print("Fitness: " + str(best_fitness))

    for test in tests:
        print("------------------")
        print(f"p = {test[0]}" + " " * (4 - len(str(test[0]))) + f"with e = {test[1]}")
        best_fitness, best_path = BPP(10, items_2, 10000, test[0], test[1])
        # print("Best path is: " + str(best_path))
        print("Fitness: " + str(best_fitness))