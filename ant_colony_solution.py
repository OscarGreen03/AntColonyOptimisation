import math
import random
from bisect import bisect_left
import numpy as np
import graph_generation
import matplotlib.pyplot as plt


class AntColony(graph_generation.Graph):
    def __init__(self):
        super().__init__()
        self.valid_edges_cache = {}

    def evaporate(self, rate):
        for node in self.graph:
            for edge in self.graph[node]:
                self.graph[node][edge] *= rate






    def get_next_step(self, node, alpha=1, beta=1):
        # alpha is pheromone**alpha
        current_edges = self.graph[node]
        edge_list = list(current_edges.keys())
        if "END" in edge_list:
            return "END"
        if "START" in edge_list:
            edge_list.remove("START")
        if node == "START":
            node = (-1, -1,)

        if node in self.valid_edges_cache.keys():
            valid_edges = self.valid_edges_cache[node]
        else:
            valid_edges = [edge for edge in edge_list if isinstance(edge, tuple) and (edge[1] > node[1])]
            self.valid_edges_cache[node] = valid_edges
        ## all edges where is a tuple and item (index 1) is larger than the previous item
        ## this means it can only move foward



        weights = [current_edges[edge] for edge in valid_edges]
        cum_weights = []
        total = 0
        for weight in weights:
            total += weight
            cum_weights.append(total)
        choice = random.uniform(0, cum_weights[-1])
        index = bisect_left(cum_weights, choice)
        return valid_edges[index]

        '''#Is this faster??
        weights = np.array([current_edges[edge] for edge in valid_edges])
        cumulative_weights = np.cumsum(weights)
        choice = random.uniform(0, cumulative_weights[-1])
        index = np.searchsorted(cumulative_weights, choice)
        return valid_edges[index]'''




        '''
        sum_weight = sum([current_edges[edge] for edge in valid_edges])
        choice = random.uniform(0, sum_weight)
        c = 0
        for edge in valid_edges:

            c = c + current_edges[edge]
            if c >= choice:
                return edge
        
        raise Exception("No Node Found")
        '''

    def produce_path(self):

        current_edges = {}
        visited_edges = {}
        current_node = "START"
        path = ["START"]
        # if "START" in self.graph.keys():
        #    current_edges = self.graph["START"]
        #    print("Current Edges: " + str(current_edges))
        # else:
        #   print("START not found??")
        #   print("Graph Keys: " + str(self.graph.keys()))

        while current_node != "END":
            next_step = self.get_next_step(current_node)
            path.append(next_step)
            current_node = next_step
        # print("Path Completed")

        return path

    def evaluate_fitness(self, path, items, bin_count):

        bins = [0 for _ in range(bin_count)]

        for i in range(len(path)):
            # for each item, add to bin
            if isinstance(path[i], tuple):
                # print("Adding item " + str(i) + ", "+ str(items[i - 1]) + "kg, to bin " + str(path[i][0]))
                bins[path[i][0]] += items[i - 1]
        # print("Fitness: " + str(math.sqrt(max(bins)**2 - min(bins)**2)))
        return max(bins) - min(bins), max(bins), min(bins)
        # return [math.sqrt(max(bins)**2 - min(bins)**2), max(bins), min(bins)]

    def generate_paths(self, n):
        pass

    def update_pheremones(self, path, fitness, weighting=0.001, global_weight=1, upper_bound=0, lower_bound=0):
        # local weight is how much to update path pheromone by score
        added_pheremone = 100 / fitness if fitness > 0 else 100

        for i in range(len(path) - 1):
            self.graph[path[i]][path[i + 1]] += added_pheremone
            # print(100/fitness)
            # score = ((fitness-lower_bound)/(upper_bound-lower_bound)) * global_weight
            # self.graph[path[i]][path[i+1]] += (1 - score) * weighting
            # self.graph[path[i]][path[i + 1]] = min(self.graph[path[i]][path[i+1]], 1)


def BPP(bin_count, items, population=10, max_generations=50, e=0.8):
    path_history = []
    g = graph_generation.produce_bin_graph(bin_count, len(items), lower_bound=0, upper_bound=1)
    colony = AntColony()
    colony.set_graph(g)
    # colony.visualise_graph(show=True)

    # generate a bunch of valid paths (population : n)
    # Evaluate fitness of each
    # Adjust pheramones for the path based on fitness
    # evaporate path
    # repeat until termination conditions met
    best_path = None
    best_fitness = float('inf')
    for i in range(max_generations):
        # colony.print_max_min_weights()
        paths = [[colony.produce_path()] for _ in range(population)]

        for path in paths:
            fitness = colony.evaluate_fitness(path[0], items, bin_count)
            # colony.update_pheremones(path[0])
            # path.extend(colony.evaluate_fitness(path[0], items, bin_count)) # may need to ajust
            colony.update_pheremones(path[0], fitness[0])
            if fitness[0] < best_fitness:
                best_path, best_fitness = path[0], fitness[0]
        # for path in paths:
        # print(path)
        # for path in paths:
        #    if path[1] < best_fitness:
        #        best_path, best_fitness = path[0], path[1]
        #    colony.update_pheremones(path[0], path[1], weighting=1,global_weight=1)

        colony.evaporate(e)
        #path_history.append(colony.as_nx_graph())
        # colony.visualise_graph(i)
        if i % 100 == 0:
            print(i)

    # colony.visualise_graph(show=True)
    # print(colony.graph)
    # print(paths[-1])
    # print("Best path is: " + str(best_path))
    # print("Fitness: " + str(best_fitness))
    #colony.visualise_graph_with_path(best_path, show=True)
    # graph_generation.produce_animation(path_history)
    return best_path, best_fitness


# items_list = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 50]
# [32, 12, 8 , 4, 13, 18, 36, 29, 42]
# print(BPP(50,items_list , 10, 1000, e=0.6))

tests = [
    (100, 0.9),
    (100, 0.6),
    (10, 0.9),
    (10, 0.6)
]
items_list1 = list(range(1, 501))
items_list2 = [(i ** 2) / 2 for i in range(1, 501)]
for test in tests:
    print("------------------")
    print(f"p = {test[0]}" + " " * (4 - len(str(test[0]))) + f"with e = {test[1]}")
    best_path, best_fitness = BPP(10, items_list1, test[0], 1000, e=test[1])
    # print("Best path is: " + str(best_path))
    print("Fitness: " + str(best_fitness))
'''
print("------------------")
print("Second Set:")
for test in tests:
    print("------------------")
    print(f"p = {test[0]}" + " " * (5 - len(str(test[0]))) + f"with e = {test[1]}")
    best_path, best_fitness = BPP(50, items_list2, test[0], 10000, e=test[1])
    # print("Best path is: " + str(best_path))
    print("Fitness: " + str(best_fitness))
print("------------------")
'''
