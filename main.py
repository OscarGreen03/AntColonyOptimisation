import numpy as np
from time import time


class AntColony:
    def __init__(self):
        self.start = []
        self.graph = []
        self.bin_count = 0
        self.item_count = 0

    def evaporate(self, e):
        self.graph *= e
        self.start *= e

    def set_bin_graph(self, bin_count, item_count):
        self.graph = np.random.rand(item_count, bin_count, bin_count)
        self.start = np.random.rand(bin_count)
        self.bin_count = bin_count
        self.item_count = item_count
        # self.graph = [[] for _ in range(item_count)]
        # for i in range(len(self.graph)):
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

    def get_new_path(self):
        path = ["START"]
        while path[-1] != "END":
            path.append(self.get_next_step(len(path) - 1, path[-1]))
        return path

    def evaluate_fitness(self, path, items):
        bins = np.zeros(self.bin_count)
        path = path[1:-1]
        for i in range(len(path)):
            # print(f'Putting item {i} {items[i]} into bin {path[i]}')
            bins[path[i]] += items[i]

        return max(bins) - min(bins)

    def update_pheremones(self, path, fitness):
        pheromone_amount = 100 / fitness

        self.start[path[1]] += pheromone_amount

        for i in range(1, len(path) - 1):
            if path[i + 1] == "END":
                break

            current_bin = path[i]
            next_bin = path[i + 1]

            self.graph[i - 1, current_bin, next_bin] += pheromone_amount


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
                # print(f"Generation {i} found with fitness: {fitness}")
        colony.evaporate(e)

    ## final generation of paths
    best_path = []
    best_fitness = float('inf')
    for path in paths:
        fitness = colony.evaluate_fitness(path, items)
        if fitness < best_fitness:
            best_fitness = fitness
            best_path = path
    # check_pheromone_distribution(colony)
    # simulate_path_generation(colony, items)
    return best_fitness


def simulate_path_generation(colony, items):
    path = ["START"]
    current_item = 0
    current_bin = "START"
    print("Simulating path generation:")
    while True:
        if current_bin == "START":
            weights = colony.start
            total_pheromone = np.sum(weights)
            probabilities = weights / total_pheromone
        elif current_item == colony.item_count:
            print("Reached the end of items.")
            path.append("END")
            break
        else:
            weights = colony.graph[current_item - 1, current_bin]
            total_pheromone = np.sum(weights)
            probabilities = weights / total_pheromone


        most_probable_next_bin = np.argmax(probabilities)


        choice = np.random.choice(colony.bin_count, p=probabilities)


        print(f"At item {current_item}, current bin: {current_bin}")
        print(f"Weights to next bins: {weights}")
        print(f"Probabilities: {probabilities}")
        print(f"Most probable next bin: {most_probable_next_bin}")
        print(f"Chosen next bin: {choice}")
        print("-" * 50)


        path.append(choice)

        current_item += 1
        current_bin = choice

        if current_item == colony.item_count:
            path.append("END")
            print("Reached the end of items.")
            break


    fitness = colony.evaluate_fitness(path, items)
    print(f"Final path: {path}")
    print(f"Fitness of the path: {fitness}")
    return path


def check_pheromone_distribution(colony):
    graph_weights = colony.graph.flatten()
    start_weights = colony.start.flatten()
    all_weights = np.concatenate([graph_weights, start_weights])

    mean_weight = np.mean(all_weights)
    min_weight = np.min(all_weights)
    max_weight = np.max(all_weights)
    std_weight = np.std(all_weights)

    zero_weights = np.sum(all_weights == 0)
    total_weights = all_weights.size


    sorted_weights = np.sort(all_weights)[::-1]  # sort descending
    cumulative_weights = np.cumsum(sorted_weights)
    total_pheromone = cumulative_weights[-1]
    top_10_percent_index = int(0.1 * total_weights)
    weight_top_10_percent = cumulative_weights[top_10_percent_index]
    proportion_top_10_percent = weight_top_10_percent / total_pheromone


    print('Pheromone weights statistics:')
    print(f'Mean: {mean_weight}')
    print(f'Min: {min_weight}')
    print(f'Max: {max_weight}')
    print(f'Standard Deviation: {std_weight}')
    print(f'Number of zero weights: {zero_weights} out of {total_weights}')
    print(f'Proportion of total pheromone in top 10% weights: {proportion_top_10_percent:.4f}')

    # change threshold where needed
    if zero_weights / total_weights > 0.5:
        print('More than 50% of the pheromone weights are zero. Weights may be trending to zero.')
    if proportion_top_10_percent > 0.9:
        print(
            'More than 90% of the pheromone is concentrated in the top 10% of weights. Paths may be locking in too quickly.')


    return {
        'mean': mean_weight,
        'min': min_weight,
        'max': max_weight,
        'std_dev': std_weight,
        'zero_weights': zero_weights,
        'proportion_top_10_percent': proportion_top_10_percent,
    }


# main array containing each column of the problem as an item in the array
# each item in this main array is an array of the bins and their weights to each other bin
# this array cont
# [start->col1, col1 , col2   , col3 ,..., coln-1->end]
# col1 = [bin1, bin2, bin3]
# bin1 = [nextb1, nextb2, nextb3, nextb4]

if __name__ == '__main__':
    start_time = time()
    tests = [
        (100, 0.9),
        (100, 0.6),
        (10, 0.9),
        (10, 0.6)
    ]
    items_1 = list(range(1, 501))
    items_2 = [(i ** 2) / 2 for i in range(1, 501)]

    for bin_count, item_list in [[10, items_1], [50, items_2]]:
        for test in tests:
            print("------------------")
            print(f"b = {bin_count}, p = {test[0]}" + " " * (4 - len(str(test[0]))) + f", e = {test[1]}")
            iterations = 10000 // test[0]
            results = []
            for i in range(5):
                results.append(BPP(bin_count, item_list, iterations, test[0], test[1], colony=None))
                print(results[-1])
            print("------------------")
    end_time = time()
    print(f'Time spent solving: {end_time - start_time}')
