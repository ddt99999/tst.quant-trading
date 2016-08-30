# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 17:46:43 2016

@author: tongtz
"""

from scipy.stats import poisson
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import sympy as sp

def plot_line(data, x_axis_title, y_axis_title):
    plt.plot(data)
    plt.xlabel(x_axis_title)
    plt.ylabel(y_axis_title)
    plt.show()

def get_c_constant(lamb, N):
    total = 0
    
    for k in range(1,N):
        total += k*poisson.pmf(k, lamb)
        
    return total
      
def calculate_u(u, c, K):
    sum_u = 0.0
    for k in range(1, K + 1):
        sum_u += (np.exp(-c)*(c**(k-1))/np.math.factorial(k-1)) * u ** (k-1) 
    return sum_u
    
def evaluate_u(c, K, epsilon):
    u = 0.0
    new_u = 0.001
    while abs(new_u - u) > epsilon:
        u = new_u
        new_u = calculate_u(u, c, K)
        
    return new_u
        
'''
N     : total number of power 
'''
def get_equations_in_str(c, N, P):
    expr = '-x+'
    if N==1:
        return str(P(1, c)/c)
    
    for i in range(1, N+1):
        expr += (str(i*P(i, c)/c) + '*x**' + str(i-1) + ('' if (i==N) else '+'))
        
    return expr

def evaluate(c, N, P):
    solutions, x = sp.symbols("solutions x")
    str_expr = get_equations_in_str(c, N, P)
    expr = sp.sympify(str_expr)
    solutions = sp.solveset(sp.Eq(expr), x)

    return solutions
    
def get_solution_list(solution_set):
    solutions = []
    for elem in solution_set:
        solutions.append(elem)
        
    return solutions
  
# For task 2  
def indicator(current_neighbour_degree_dist, next_neighbour_size):
    current_neighbour_degree_sum = 0
    
    for degree in current_neighbour_degree_dist:
        current_neighbour_degree_sum += (degree - 1)
        
    return 1 if current_neighbour_degree_sum == next_neighbour_size else 0
    
def evaluate_graph_nodes_prob(c, degree_dist, P, current_neighbour_count, *neighbours):
    shell_layer_no = len(neighbours)
    
    if shell_layer_no == 1:
        return P(neighbours[0], c)
 
def poisson_dist_generator(lamb, K):
    return np.random.poisson(lamb)
    
def uniform_dist_generator(low, high):
    return np.random.randint(low, high)

# Task 3: To calculate population dynamics
# population = np.array([0,1,0,1,1,1,1,1,0,0,0,0,0,0,1])

def evaluate_population_dynamics(population, random_generator, *args):   
    k = 0
    population_length = len(population)

    # step 1
    while (k < 1 or k > population_length):
        k = random_generator(args[0], args[1])
    
    picked_members = {}

    # step 2
    # get the k-1 populations based on k generated  
    new_n = 0
    if k > 1: 
        for i in range(1,k):
            randint = uniform_dist_generator(0, population_length) 
            picked_members[randint] = population[randint]
        
        # step 3   
        mult_val = 0
    
        for pos,val in picked_members.items():
            mult_val *= (1-val)
        
        new_n = 1 - mult_val
 
    # step 4
    random_pos = uniform_dist_generator(0, population_length)
    population[random_pos] = new_n
    
    # step 4.1 - to calculate the proportion of the zeros in population
    population_with_zeros = np.array([0] * population_length)
    population_with_zeros = population[np.where( population == 0 )]
    proportion_with_zeros = population_with_zeros.size / population.size
    proportion_with_ones = 1 - proportion_with_zeros
    
    return proportion_with_zeros

def population_dynamic_simulation(simulation_count, population_size, random_number_generator, *args):
    random_population = []
    proportions = []
    # initialise the random population
    for i in range(0,population_size):
        random_population.append(uniform_dist_generator(0,2))
        
    population = np.array(random_population)
    
    for i in range(0,simulation_count):
        proportions.append(evaluate_population_dynamics(population, random_number_generator, *args))
    
    return proportions
    
def evaluate_poisson_u(c):
    u = 0.0
    pos_u = 0.0001
    while abs(pos_u - u) > 0.00001:
        u = pos_u
        pos_u = np.exp(c*(u-1))
        
    return u
    
def get_giant_cluster(random_graph):
    return max(nx.connected_component_subgraphs(random_graph), key=len)
    
def get_degree_sequence(graph):
    return list(nx.degree(graph).values()) # degree sequence
    
def get_degree_count_with_k(graph, k):
    degree_sequence=list(nx.degree(graph).values()) # degree sequence
    
    count = 0
    for degree in degree_sequence:
        if degree == k:
            count += 1           
    return count 
    
def get_subgraph_ratio_with_full_random_graph(subgraph_size, full_graph_size):
    return float(subgraph_size / full_graph_size)
    
def find_giant_cluster_probability(total_node, prob):
    random_graph = nx.erdos_renyi_graph(total_node,prob)
    
    # find the largest giant cluster
    largest_giant_cluster = max(nx.connected_component_subgraphs(random_graph), key=len)
    
#    for s in nx.nodes(giant):
#        print('%s %d' % (s,nx.degree(G3,s)))
#    for v in nx.nodes(G3):
#        print('%s %d %f' % (v,nx.degree(G3,v),nx.clustering(G3,v)))
    
    degree_sequence=list(nx.degree(largest_giant_cluster).values()) # degree sequence
    
    largest_giant_cluster_size = len(degree_sequence)
    giant_cluster_prob = largest_giant_cluster_size / total_node
    
    return giant_cluster_prob
    
def get_mean(values, N):
    return sum(values) / N
    
def simulate_giant_cluster_experiment(run_no = 1):
    run_no = 10
    total_node = 1000
    prob = 0.0015
    max_degree = 10
    conditional_prob_with_k_dict = {}
    total_prob_with_k_dict = {}
    
    giant_cluster_probability_list = []

    for k in range(1, max_degree+1):
        conditional_prob_with_k_dict[k] = []
        total_prob_with_k_dict[k] = []

    for i in range(1, run_no + 1):
        random_graph = nx.erdos_renyi_graph(total_node,prob)
        giant_cluster = get_giant_cluster(random_graph)
        degree_sequence = get_degree_sequence(giant_cluster)

        giant_cluster_probability = get_subgraph_ratio_with_full_random_graph(len(degree_sequence), total_node)        
        giant_cluster_probability_list.append(giant_cluster_probability)
                    
        for k in range(1,max_degree+1):
            giant_cluster_degree_count_with_k = get_degree_count_with_k(giant_cluster, k)
            total_prob_with_k = giant_cluster_degree_count_with_k / total_node
            
            total_prob_with_k_dict[k].append(total_prob_with_k)

            random_graph_degree_count_with_k = get_degree_count_with_k(random_graph, k)
                   
            conditional_prob_with_k = 0 if random_graph_degree_count_with_k == 0 else float(giant_cluster_degree_count_with_k / random_graph_degree_count_with_k)
            conditional_prob_with_k_dict[k].append(conditional_prob_with_k)
            #degree_with_k_ratio_list.append(degree_ratio)
            
            #conditional_prob_with_k_dict[k] = degree_with_k_ratio_list
    
    conditional_prob_with_k_bin = {}
    total_prob_with_k_bin = {}
    for k in range(1, max_degree+1):       
        conditional_prob_with_k_bin[k] = get_mean(conditional_prob_with_k_dict[k], run_no)
        total_prob_with_k_bin[k] = get_mean(total_prob_with_k_dict[k], run_no)  
    
    #conditional_prob_with_k_dict[k].append(conditional_probability_with_k)
    #total_prob_with_k_dict[k].append(total_probability_with_k)
    
    giant_cluster_probability = get_mean(giant_cluster_probability_list, run_no)
    print(1 - giant_cluster_probability)
    
    



        
#def draw_graph(graph):
#    # extract nodes from graph
#    nodes = set([n1 for n1, n2 in graph] + [n2 for n1, n2 in graph])
#    
#    # create networkx graph
#    G = nx.Graph()
#    
#    # add nodes 
#    for node in nodes:
#        G.add_node(node)
#        
#    # add edges
#    for edge in graph:
#        G.add_edge(edge[0], edge[1])
#        
#    # draw graph
#    pos = nx.shell_layout(G)
#    nx.draw(G, pos)
#    
#    # show graph
#    plt.show()
#    
#
## draw example
#graph = [(20, 21),(21, 22),(22, 23), (23, 24),(24, 25), (25, 20)]
#draw_graph(graph)
#
## more complex graph
#def draw_graph_2(graph, labels=None, graph_layout='shell',
#               node_size=1600, node_color='blue', node_alpha=0.3,
#               node_text_size=12,
#               edge_color='blue', edge_alpha=0.3, edge_tickness=1,
#               edge_text_pos=0.3,
#               text_font='sans-serif'):
#
#    # create networkx graph
#    G=nx.Graph()
#
#    # add edges
#    for edge in graph:
#        G.add_edge(edge[0], edge[1])
#
#    # these are different layouts for the network you may try
#    # shell seems to work best
#    if graph_layout == 'spring':
#        graph_pos=nx.spring_layout(G)
#    elif graph_layout == 'spectral':
#        graph_pos=nx.spectral_layout(G)
#    elif graph_layout == 'random':
#        graph_pos=nx.random_layout(G)
#    else:
#        graph_pos=nx.shell_layout(G)
#
#    # draw graph
#    nx.draw_networkx_nodes(G,graph_pos,node_size=node_size, 
#                           alpha=node_alpha, node_color=node_color)
#    nx.draw_networkx_edges(G,graph_pos,width=edge_tickness,
#                           alpha=edge_alpha,edge_color=edge_color)
#    nx.draw_networkx_labels(G, graph_pos,font_size=node_text_size,
#                            font_family=text_font)
#
#    if labels is None:
#        labels = range(len(graph))
#
#    edge_labels = dict(zip(graph, labels))
#    nx.draw_networkx_edge_labels(G, graph_pos, edge_labels=edge_labels, 
#                                 label_pos=edge_text_pos)
#
#    # show graph
#    plt.show()
#
#graph = [(0, 1), (1, 5), (1, 7), (4, 5), (4, 8), (1, 6), (3, 7), (5, 9),
#         (2, 4), (0, 4), (2, 5), (3, 6), (8, 9)]
#
## you may name your edge labels
#labels = map(chr, range(65, 65+len(graph)))


# graph example
if __name__ == "__main__":
#    nodes = range(16)
#    #graph = nx.barabasi_albert_graph(10,6)
#    G1 = nx.barabasi_albert_graph(10,5)
#    G2 = nx.barbell_graph(10,10)
    simulate_giant_cluster_experiment(10)
    ans = evaluate_poisson_u(1.5)
    

#    maze=nx.sedgewick_maze_graph()
#    CG1 = nx.disjoint_union(G1,G3)
#    #CG2 = nx.disjoint_union(CG1,maze)
#    
#    nx.draw(G3)
#    plt.show()
    
#    for (u,v,d) in CG1.edges(data='weight'):
#        #if d<0.5: 
#        print('(%d, %d)'%(u,v))
#        
#    print(nx.degree(CG1))
    
    # initial value
#    n1 = 5
#    prob_n1 = np.random.poisson(n1) # level 0
#    
#    # pre-calculate the c
#    lambdas = [l for l in np.arange(0.5,3.5,0.5)]
#    c = {key: 0.0 for key in lambdas}
#    
#    s = np.random.poisson(lam=(100., 500.), size=(100, 2))
#    
#    N = 100
#    
#    for lamb in lambdas:
#        c[lamb] = get_c_constant(lamb, N)
        
    #u_hat = 
    # testing
#    x = sp.symbols("x", real=True, positive=True)
#    sol = sp.symbols("sol", real=True, positive=True)
#    sol = evaluate(1.5, 25, poisson_distribution)
#    solutions = get_solution_list(sol)
    
    proportions = population_dynamic_simulation(10000, 1000, poisson_dist_generator, 2.0, 0)
    plot_line(proportions, "steps", "zeros-population ratio")
#    str_expr = get_equations_in_str(0.5, 15, poisson_distribution)
#    print(str_expr)
#    expr = sp.sympify(str_expr)
#    solutions = sp.solveset(sp.Eq(expr), x)
#    sols = get_solution_list(solutions)
#    
#    solutions_iter = iter(solutions)
#    b = next(solutions_iter) + 1
    
    # level 1
    
    
    #nx.draw(maze)
    #plt.show()
    #draw_graph(graph)