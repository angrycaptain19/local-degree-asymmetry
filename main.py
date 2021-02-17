import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import time
import math
from bisect import bisect_left
from datetime import datetime
from sklearn.linear_model import LinearRegression


input_types = ["from_file", "barabasi-albert", "triadic", "test"]
# Change value below to run experiment
input_type_num = 1

# https://networkx.org/documentation/stable/reference/index.html

def get_neighbor_summary_degree(graph, node):
    neighbors_of_node = graph.neighbors(node)
    acc = 0
    for neighbor in neighbors_of_node:
        acc += graph.degree(neighbor)
    return acc


def get_neighbor_average_degree(graph, node):
    si = get_neighbor_summary_degree(graph, node)
    return si / graph.degree(node)


def get_friendship_index(graph, node):
    ai = get_neighbor_average_degree(graph, node)
    return ai / graph.degree(node)

# 0 - From file
def experiment_file():
    filename = "musae_git_edges.txt"
    graph = nx.read_edgelist(filename)

    graph_nodes = graph.nodes()

    # b (beta) = friendship index 
    bs = []
    for node in graph_nodes:
        bs.append(get_friendship_index(graph, node))
    
    n, bins, _ = plt.hist(bs, bins=range(40), rwidth=0.85)
    
    # lnt = log(bins), lnb = log(n)
    lnt, lnb = [], []
    for i in range(len(bins) - 1):
        lnt.append(math.log(bins[i] + 1))
        lnb.append(math.log(n[i]) if n[i] != 0 else 0)

    np_lnt = np.array(lnt).reshape(-1, 1)
    np_lnb = np.array(lnb)

    model = LinearRegression()
    model.fit(np_lnt, np_lnb)
    linreg_predict = model.predict(np_lnt)

    f = open("hist_" + filename, "w")
    f.write("t\tb\tlnt\tlnb\tlinreg\t k=" + str(model.coef_) + ", b=" + str(model.intercept_) + "\n")
    for i in range(len(bins) - 1):
        f.write(str(bins[i]) + "\t" + str(int(n[i])) + "\t" + str(lnt[i]) + "\t" + str(lnb[i]) + "\t" + str(linreg_predict[i]) + "\n")
    f.close()
    #nx.draw(graph, with_labels=True)
    #plt.show()

# 1 Barabasi-Albert
def create_ba(n, m):
    m0 = 3
    graph = nx.Graph()
    graph.add_nodes_from(range(0, m0))
    graph.add_edges_from([(0, 1), (1, 2), (0, 2)])

    # get node statistics
    focus_ind = 50
    s_focus, a_focus, b_focus = [], [], []
    
    probabilities = [0]
    # initial probabilities
    for i in range(m0):
        probabilities.append(graph.degree(i) + probabilities[i])

    for i in range(m0, n + 1):
        graph.add_node(i)
            
        # update preferential attachment probabilities | переписать через choices с весами
        sum_degrees = i * 2 * m
        for j in range(1, len(probabilities)):
            probabilities[j] = probabilities[j - 1] + (graph.degree(j - 1) / sum_degrees)

        if focus_ind < i and i % 50 == 0: # сократить
            s_focus.append(get_neighbor_summary_degree(graph, focus_ind))
            a_focus.append(round(get_neighbor_average_degree(graph, focus_ind), 4))
            b_focus.append(round(get_friendship_index(graph, focus_ind), 4))

        # preferential attachment 
        for _ in range(m): # TODO: zero case? check same node case / random.choices
            to_connect = bisect_left(probabilities, random.random()) - 1
            graph.add_edge(i, to_connect)

        probabilities.append(1)

    should_plot = False
    if should_plot:
        nx.draw(graph, with_labels=True, font_size=6, node_size=50)
        
        plt.show()

        s_focus_xrange = [x / len(s_focus) for x in range(len(s_focus))]
        plt.plot(s_focus_xrange, s_focus)
        plt.show()

    return (s_focus, a_focus, b_focus)


def experiment_ba():
    n = 100000
    m = 5
    number_of_experiments = 1
    
    f_s = open("out_ba_s.txt", "a")
    f_a = open("out_ba_a.txt", "a")
    f_b = open("out_ba_b.txt", "a")
    files = (f_s, f_a, f_b)
    start_time = time.time()
    now = datetime.now()
    for f in files:
        # Experiment datetime
        f.write("> n=" + str(n) + " m=" + str(m) + " " + now.strftime("%d/%m/%Y %H:%M:%S") + "\n")
    for _ in range(number_of_experiments):
        # Create random network and get values
        result = create_ba(n, m)
        for j in range(len(result)):
            # Write values to files
            files[j].write(" ".join(str(x) for x in result[j]) + "\n")
    print(("Elapsed time: %s", time.time() - start_time))
        

# 2 Triadic Closure
def create_triadic(n, m, p):
    m0 = 3

    start_time = time.time()

    graph = nx.Graph()
    graph.add_nodes_from(range(0, m0))
    graph.add_edges_from([(0, 1), (1, 2), (0, 2)])

    focus_ind = 50
    s_focus, a_focus, b_focus = [], [], []

    # probabilities for preferential attachment
    probabilities = [0]
    for i in range(m0):
        probabilities.append(graph.degree(i) + probabilities[i]) # на каждом шаге добавлять!!! КРИТИЧНО

    for i in range(m0, n + 1):
        graph.add_node(i)
            
        # update preferential attachment probabilities
        sum_degrees = i * 2 * m
        for j in range(1, len(probabilities)):
            probabilities[j] = probabilities[j - 1] + (graph.degree(j - 1) / sum_degrees)

        # save focus node statistics
        if focus_ind < i and i % 50 == 0:
                s_focus.append(get_neighbor_summary_degree(graph, focus_ind))
                a_focus.append(round(get_neighbor_average_degree(graph, focus_ind), 4))
                b_focus.append(round(get_friendship_index(graph, focus_ind), 4))
        
        # Step 1
        node_i = bisect_left(probabilities, random.random()) - 1
        graph.add_edge(i, node_i)
        # Step 2 / сгенерить m случайных чисел, сравнивать с p. / numpy random samples
        for j in range(1, m):
            neighbors_i = graph[node_i]
            if random.random() < p:
                # Step 2a (any neighboring node)
                neighbor_to_link = random.choice(list(neighbors_i.keys())) #?
                graph.add_edge(i, neighbor_to_link)
            else:
                # Step 2b (any node with preferential attachment)
                to_connect = bisect_left(probabilities, random.random()) 
                graph.add_edge(i, to_connect - 1)


    should_plot = False
    if should_plot:
        #nx.draw(graph, with_labels=True, font_size=6, node_size=50)
        print(time.time() - start_time)
        #plt.show()

        s_focus_xrange = [x / len(s_focus) for x in range(len(s_focus))]
        plt.plot(s_focus_xrange, s_focus)
        plt.show()

    return (s_focus, a_focus, b_focus)


def experiment_triadic():
    n = 10000
    m = 5
    p = 0.75
    number_of_experiments = 99

    should_write = True
    if should_write:
        f_s = open("out_tri_s.txt", "a")
        f_a = open("out_tri_a.txt", "a")
        f_b = open("out_tri_b.txt", "a")
        files = (f_s, f_a, f_b)
        now = datetime.now()
        for f in files:
            f.write("> n=" + str(n) + " m=" + str(m) + " " + now.strftime("%d/%m/%Y %H:%M:%S") + "\n")
        for _ in range(number_of_experiments):
            result = create_triadic(n, m, p)
            for j in range(len(result)):
                files[j].write(" ".join(str(x) for x in result[j]) + "\n")
    else:
        result = create_triadic(n, m, p)
    pass

# 3 Test data
def print_node_values(graph, node_i):
    print("Summary degree of neighbors of node %s (si) is %s" % (node_i, get_neighbor_summary_degree(graph, node_i)))
    print("Average degree of neighbors of node %s (ai) is %s" % (node_i, get_neighbor_average_degree(graph, node_i)))
    print("Friendship index of node %s (bi) is %s" % (node_i, get_friendship_index(graph, node_i)))


def experiment_test():
    filename = "test_graph.txt"

    graph = nx.read_edgelist(filename)
    print_node_values(graph, '1')
    
    nx.draw(graph, with_labels=True)
    plt.show()


if __name__ == "__main__":
    input_type = input_types[input_type_num]
    print("Doing %s experiment" % input_type)
    if input_type == "from_file":
        experiment_file()
    elif input_type == "barabasi-albert":
        experiment_ba()
    elif input_type == "triadic":
        experiment_triadic()
    elif input_type == "test":
        experiment_test()
