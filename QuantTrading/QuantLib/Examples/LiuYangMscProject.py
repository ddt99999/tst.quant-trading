# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 17:46:43 2016

@author: tongtz
"""

from scipy.stats import poisson
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def get_c_constant(lamb, N):
    total = 0
    
    for k in range(1,N):
        total += k*poisson.pmf(k, lamb)
        
    return total
        

        
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
#    G3 = nx.erdos_renyi_graph(5,0.15)
#    maze=nx.sedgewick_maze_graph()
#    CG1 = nx.disjoint_union(G1,G3)
#    #CG2 = nx.disjoint_union(CG1,maze)
#    
#    nx.draw(CG1)
#    plt.show()
#    
#    for (u,v,d) in CG1.edges(data='weight'):
#        #if d<0.5: 
#        print('(%d, %d)'%(u,v))
#        
#    print(nx.degree(CG1))
    
    # initial value
    n1 = 5
    prob_n1 = np.random.poisson(n1) # level 0
    
    # pre-calculate the c
    lambdas = [l for l in np.arange(0.5,3.5,0.5)]
    c = {key: 0.0 for key in lambdas}
    
    s = np.random.poisson(lam=(100., 500.), size=(100, 2))
    
    N = 100
    
    for lamb in lambdas:
        c[lamb] = get_c_constant(lamb, N)
        
    #u_hat = 
    
    # level 1
    
    
    #nx.draw(maze)
    #plt.show()
    #draw_graph(graph)