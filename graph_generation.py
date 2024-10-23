from random import randint, uniform

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.lines import Line2D

class Graph:
    def __init__(self):
        self.graph = {}

    def print_max_min_weights(self):
        maximum = 0
        minimum = float('inf')
        for node in self.graph.keys():
            for edge in self.graph[node].keys():
                weight = self.graph[node][edge]
                if weight > maximum:
                    maximum = weight
                    max_edge = f"{node} {edge}"
                elif weight < minimum:
                    minimum = weight
                    min_edge = f"{node} {edge}"
        print("Max: " + str(maximum) + " " * (30 - len(str(maximum))) + "Min: " + str(minimum))
        print("Max node = " + str(max_edge))
        print("Min node = " + str(min_edge))

    def set_graph(self, graph):
        if isinstance(graph, Graph):
            self.graph = graph.graph
        elif isinstance(graph, dict):
            self.graph = graph
        else:
            print("Type: " + str(type(graph)) + " is not supported. Please use a dict or a Graph")

    def add_node(self, node):
        if node in self.graph:
            raise Exception(f"Node {node} already exists")
        self.graph[node] = {} ## dict of directions and weights

    def add_edge(self, n1, n2, weight):
        if n1 not in self.graph:
            raise Exception(f"Node {n1} not in graph")
        if n2 not in self.graph:
            raise Exception(f"Node {n2} not in graph")
        self.graph[n1][n2] = weight
        #self.graph[n2][n1] = weight

    def edit_weight(self, n1, n2, weight):
        if n1 not in self.graph:
            raise Exception(f"Node {n1} not in graph")
        if n2 not in self.graph:
            raise Exception(f"Node {n2} not in graph")



        self.graph[n1][n2] = weight
        #self.graph[n2][n1] = weight

    def get_nodes(self):
        return self.graph.keys()

    def get_node_connections(self, node):
        return self.graph[node].keys()

    def __str__(self):
        return str(self.graph)

    def as_nx_graph(self):
        G = nx.Graph()
        node_mapping = {}
        for node in self.graph:
            G.add_node(node)
            for edge in self.graph[node]:
                G.add_edge(node, edge, weight=self.graph[node][edge])

        numerical_nodes = []
        for node in G.nodes():
            if isinstance(node, tuple) and len(node) == 2:
                try:
                    j = float(node[0])
                    i = float(node[1])
                    numerical_nodes.append((i, j))
                except ValueError:
                    pass  # Skip nodes where conversion fails

        if numerical_nodes:
            is_list, js_list = zip(*numerical_nodes)
            min_i = min(is_list)
            max_i = max(is_list)
            min_j = min(js_list)
            max_j = max(js_list)
            avg_j = (min_j + max_j) / 2
        else:
            # Default values in case there are no numerical nodes
            min_i = 0
            max_i = 0
            avg_j = 0

        # Create positions
        pos = {}
        for node in G.nodes():
            if isinstance(node, tuple) and len(node) == 2:
                try:
                    j = float(node[0])
                    i = float(node[1])
                    pos[node] = (i, j)
                except ValueError:
                    print(f"Node {node} has non-numeric identifiers.")
                    continue
            elif node == "START":
                # Position "START" node to the left of the min_i
                pos[node] = (min_i - 1, avg_j)
            elif node == "END":
                # Position "END" node to the right of the max_i
                pos[node] = (max_i + 1, avg_j)
            else:
                # Position other non-numeric nodes at a default location
                pos[node] = (max_i + 2, avg_j)
        # pos = {node: (y, x) for node, (x, y) in pos.items()}
        edge_weights = nx.get_edge_attributes(G, 'weight')
        weights = [edge_weights[edge] for edge in G.edges()]
        return G, pos, weights

    def visualise_graph(self, graph_name=None, show=False):

        G = nx.Graph()
        node_mapping = {}
        for node in self.graph:
            G.add_node(node)
            for edge in self.graph[node]:

                G.add_edge(node, edge, weight=self.graph[node][edge])

        numerical_nodes = []
        for node in G.nodes():
            if isinstance(node, tuple) and len(node) == 2:
                try:
                    j = float(node[0])
                    i = float(node[1])
                    numerical_nodes.append((i, j))
                except ValueError:
                    pass  # Skip nodes where conversion fails ( Start and End Nodes)

        if numerical_nodes:
            is_list, js_list = zip(*numerical_nodes)
            min_i = min(is_list)
            max_i = max(is_list)
            min_j = min(js_list)
            max_j = max(js_list)
            avg_j = (min_j + max_j) / 2
        else:
            raise Exception("No Numerical Nodes")

            #min_i = 0
            #max_i = 0
            #avg_j = 0

        # Create positions
        pos = {}
        for node in G.nodes():
            if isinstance(node, tuple) and len(node) == 2:
                try:
                    j = float(node[0])
                    i = float(node[1])
                    pos[node] = (i, j)
                except ValueError:
                    print(f"Node {node} has non-numeric identifiers.")
                    continue
            elif node == "START":
                # Position "START" node to the left of the min_i
                pos[node] = (min_i - 1, avg_j)
            elif node == "END":
                # Position "END" node to the right of the max_i
                pos[node] = (max_i + 1, avg_j)
            else:
                # Position other non-numeric nodes at a default location
                pos[node] = (max_i + 2, avg_j)
        #pos = {node: (y, x) for node, (x, y) in pos.items()}
        #plt.figure(figsize=(30, 20))
        # Draw the graph with specified positions and labels

        edge_weights = nx.get_edge_attributes(G, 'weight')
        weights = [edge_weights[edge] for edge in G.edges()]

        fig, ax = plt.subplots(dpi=600)
        nx.draw_networkx(G, pos, with_labels=True, node_size=50, font_size=4, width=weights, ax=ax) # width=1

        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Node',
                   markerfacecolor='lightblue', markersize=10),
            Line2D([0], [0], color='gray', lw=2, label='Edge')
        ]

        # Add the legend to the plot
        plt.legend(handles=legend_elements, loc='upper right')
        #ax.set_xlabel('Object')
        #ax.set_ylabel('Bin')
        #ax.set_title(graph_name)

        plt.xlabel('Object')
        plt.ylabel('Bin')
        plt.title(f'Graph Visualization {" of " + str(graph_name) if graph_name is not None else "A"}')
        #plt.grid(True)
        #plt.axis('equal')  # Ensure equal scaling on both axes#
        if show:
            plt.show()
        plt.close(fig)
        return fig, ax


    def visualise_graph_with_path(self, path, graph_name=None, show=False):
        G, pos, weights = self.as_nx_graph()
        # Extract the edges in the path
        path_edges = list(zip(path, path[1:]))

        # Create edge colors and widths
        edge_colors = []
        edge_widths = []
        for edge in G.edges():
            if edge in path_edges or (edge[1], edge[0]) in path_edges:  # For undirected graphs
                edge_colors.append('red')
                edge_widths.append(2.5)
            else:
                edge_colors.append('gray')
                edge_widths.append(0.5)

        # Create node colors
        node_colors = []
        for node in G.nodes():
            if node in path:
                node_colors.append('red')
            else:
                node_colors.append('lightblue')

        fig, ax = plt.subplots(dpi=600)
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=50, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=4, ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, ax=ax)

        # Create legend
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Node',
                   markerfacecolor='lightblue', markersize=10),
            Line2D([0], [0], color='gray', lw=2, label='Edge'),
            Line2D([0], [0], color='red', lw=2, label='Path Edge'),
            Line2D([0], [0], marker='o', color='w', label='Path Node',
                   markerfacecolor='red', markersize=10)
        ]

        plt.legend(handles=legend_elements, loc='upper right')
        plt.xlabel('Object')
        plt.ylabel('Bin')
        plt.title(f'Graph Visualization {" of " + str(graph_name) if graph_name is not None else "A"}')

        if show:
            plt.show()
        plt.close(fig)
        return fig, ax

def produce_bin_graph(bin_count, object_count, lower_bound=0, upper_bound=1):
    ## the graph represents the probelem
    newGraph = Graph()

    ## each turn needs to be represented in the graph.
    # Move = Put next item in this box
    # aka: Move to node 3, 1 is put item 1 in box 3
    #newGraph.add_node("START")
    for i in range(bin_count):
        for j in range(object_count):
            newGraph.add_node((i, j))
    # cbin current bin
    for cbin in range(bin_count):
        for cobject in range(object_count - 1):
            for new_bin in range(bin_count):
                # for each node in graph, create connections to each in the next line
                newGraph.add_edge((cbin, cobject), (new_bin, cobject+1), uniform(lower_bound, upper_bound))


                #print("=========")

    newGraph.add_node("START")
    newGraph.add_node("END")
    for i in range(bin_count):
        newGraph.add_edge("START", (i, 0), uniform(lower_bound,upper_bound))
        newGraph.add_edge((i, object_count-1),"END", uniform(lower_bound,upper_bound))

    return newGraph




def get_next_frame(i, frames, ax):
    ax.clear() #clear prev frame
    G, pos, weights = frames[i]
    nx.draw_networkx(G, pos, with_labels=True, node_size=300, font_size=8, width=(weights), ax=ax)
    ax.set_title(f'Graph Visualization of {i}')
    return ax.artists

def produce_animation(graphs):

    #fig1, ax1 = graphs[0].visualise_graph()
    #G, pos, weights = graphs[0]
    fig1, ax1 = plt.subplots(dpi=300)

    #frames = [graphs[i].as_nx_graph() for i in range(1, len(graphs))]
    ani = animation.FuncAnimation(fig1, get_next_frame, fargs=(graphs, ax1), frames=len(graphs))
    ani.save("testani.mp4", writer='ffmpeg', fps=5)


