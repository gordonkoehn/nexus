#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyzes network characteristics of networkX networks.

This module shall contain various functions to classify networks by:
    - degree distribution
    - clustering coefficient
    - shortest path
    - small world coefficient


Created on Tue Jul 19 17:20:42 2022

@author: gordonkoehn
"""

import networkx as nx
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import collections
import pandas as pd


def draw_graph(G: nx.classes.graph.Graph, title = None):
    """Draw nx graph with kamada kawai layout.
    
    Parameters
    ----------
    G : networkx.classes.graph.Graph
        Graph to analyze
        
    (title =  None : str)
        (optional argument to put title on plot)
    """
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize = (7,7))
    ## plotting options
    options = {
    'node_color': 'lightsteelblue',
    'node_size': 250,
    'width': 1,
    'arrowstyle': '-|>',
    'arrowsize': 10,
    }  
    nx.draw_networkx(G, ax=axes, pos=nx.kamada_kawai_layout(G), arrows=True, **options, with_labels=True,font_size =10, font_weight='regular')
    
    if title is not None:
        axes.set_title(title)


class netInspector():
    """Implements a vararity of network analytics.
    
    It was intended to be used for directed graphs.
    """

    def __init__(self, G: nx.classes.graph.Graph, graph_type : str, genParams : dict):
        """Initialize netInspector object with a given Graph G.
        
        Parameters
        ----------
        G : networkx.classes.graph.Graph
            Graph to analyze
        """
        self.G = G
        self.graph_type = graph_type
        self.genParams = genParams
        
        self.degree_sequence = None
        self.degree_hist = None
        self.node_degree_avg = None
        self.average_clustering = None
        self.average_shortest_path = None
        # self.small_wolrd_coeffs = None
        self.power_law_coeffs = None
        
        self.x_in = None
        self.x_out = None 
        self.out_degree_hist = None
        self.in_degree_hist = None
        self.out_power_law_coeffs = None
        self.in_power_law_coeffs = None
        self.small_world_coeffs = None
        self.totalEdges = None
    

    def __str__(self):
        """
        Print network properties.

        Returns
        -------
        None.

        """
        output = ""
        
        if (self.graph_type is not None):
            output += "Graph type:" + self.graph_type + "\n"
            
            output += "Generation Parameters: \n"
            for key, value in self.genParams.items():
                output += f"{key} : {value:2.2f}\n"
            output += "\n"
            
            # if (self.graph_type == "scale-free"):
               
            #     output += "= Theoretical exponents =\n"
            #     output += f"X_in = {self.x_in : 2.2f}\n"
                
            #     output += f"X_out = {self.x_out : 2.2f}\n"
            
    
        
        # if (self.degree_sequence is not None):
        #     output += f"Degree sequence\n {self.degree_sequence} \n\n"
            
        # if (self.degree_hist is not None):
        #      output +=  "== Degree histogram == \n"
        #      # just printing out nicly with pandas
        #      degree_hist = pd.DataFrame.from_dict(self.degree_hist, columns=['nodes'],orient='index')
        #      degree_hist['degree'] = degree_hist.index
        #      degree_hist = degree_hist.sort_values(by=['degree'])
        #      columns_titles = ["degree","nodes"]
        #      degree_hist=degree_hist.reindex(columns=columns_titles)
        #      output += (degree_hist.to_string(index=False))
        #      output += "\n"
        if (self.totalEdges is not None):
            output += f"Total Edges: {self.totalEdges}\n"
                 
        if (self.node_degree_avg is not None): 
            output += f"Average Node Degree: {self.node_degree_avg:.2f}\n"
            
        if (self.average_clustering is not None): 
            output += f"Average Clustering Coefficient: {self.average_clustering:.2f}\n"
            
        # if (self.small_wolrd_coeffs is not None):
        #     output += f"Small-World Coefficients:\n" 
        #     output += f"sigma ={self.small_wolrd_coeffs['sigma']:.5f} \n"
        #     output += f"gamma ={self.small_wolrd_coeffs['sigma']:.5f} \n"
        
        if (self.average_shortest_path is not None):
            output += f"Average shortest path: {self.average_shortest_path : .2f}\n"
            
        if (self.power_law_coeffs is not None):
            output += f"Power-Law fit: gamma={self.power_law_coeffs['popt'][1] : 2.3f}\n"
            
             
        if (self.small_world_coeffs is not None):
            output += "=Small-worldness="
            output += f"Sigma: {self.power_law_coeffs['sigma']: 2.3f}\n"
            output += f"Gamma: {self.power_law_coeffs['gamma']: 2.3f}\n"
            
        
        return output
    
    
    
    def eval_all(self):
        """Evaluate all network characteristics."""
        self.degree_seq_hist()
           
        self.out_degree_seq_hist()
        self.in_degree_seq_hist()
        
        self.avg_node_degree()
        self.compute_clustering()
        #self.compute_small_world_coeff()
        
        #catch exception of shortes path fails due to weakly connected graph os such
        try: 
            self.compute_shortest_path()
        except Exception as err:
            print(err)
        
        self.calcTotalEdges()
        
        #self.compute_small_world_coeff()
        
        # calculate theoretical scale free in- /out- degrees
        if (self.graph_type == "scale-free"):
            self.power_law_fit()
            
            a = self.genParams['alpha']
            b = self.genParams['beta']
            g = self.genParams['gamma']
            d_out = self.genParams['delta_in']
            d_in = self.genParams['delta_out']
            
            c1 = (a + b) / (1 + d_in * (a + g))
            c2 = (b + g) / (1 + d_out *(a + g))
            
            x_in = 1 + 1/c1
            x_out = 1 + 1/c2 
            
            self.x_in = x_in
            self.x_out = x_out
     
            
  

    def degree_seq_hist(self): 
        """Calculate and self-assigns degree sequence and degree histrogram .
        
        Parameters
        ----------
        self.G : nx.graph.Graph
           graph to analyze
            
        Returns
        -------
        None
        """
        degree_sequence = [d for n, d in self.G.degree()]  # degree sequence
    
        self.degree_sequence =degree_sequence
        
        hist = {}
        for d in degree_sequence:
            if d in hist:
                hist[d] += 1
            else:
                hist[d] = 1
            
        self.degree_hist = hist
        
    def out_degree_seq_hist(self):
        """Calculate out-degree sequence and histogram and self assigns it."""
        out_degree_sequence = [d for n, d in self.G.out_degree()]  # degree sequence

        self.out_degree_sequence =out_degree_sequence
        
        hist = {}
        for d in out_degree_sequence:
            if d in hist:
                hist[d] += 1
            else:
                hist[d] = 1
                
        self.out_degree_hist = hist
        
    def in_degree_seq_hist(self):
        """Calculate in-degree sequence and histogram and self assigns it."""
        in_degree_sequence = [d for n, d in self.G.in_degree()]  # degree sequence
         
        self.in_degree_sequence = in_degree_sequence
          
        hist = {}
        for d in in_degree_sequence:
            if d in hist:
                hist[d] += 1
            else:
                hist[d] = 1
                  
        self.in_degree_hist = hist
               
        
    def avg_node_degree(self):
        """Claculate and self-assign average node degree."""
        self.node_degree_avg = np.mean(self.degree_sequence)
        
        
    
    def compute_clustering(self):
        """Calculate clustering and average clustering and self-assigns it."""
        #need to convert directed to undirected graph
        self.average_clustering = nx.approximation.average_clustering(self.G.to_undirected() , trials=1000, seed=10)
        
      
    # is not implemented for directed??
    # def cumpute_small_world_coeff(self):
    #     """Compute the small world coefficients and self-assign."""
    #     sigma = nx.algorithms.smallworld.sigma(self.G)
    #     omega = nx.algorithms.smallworld.omega(self.G)
    #     coeffs =dict()
    #     coeffs['sigma'] = sigma
    #     coeffs['omega'] = omega
        
    #     self.small_wolrd_coeffs = coeffs
        
        
    def compute_shortest_path(self):
        """Compute average shortest path length of network and self-assign."""
        self.average_shortest_path =  nx.average_shortest_path_length(self.G)
        
    def power_law_fit(self):
        """Calculate power-law exponents of in-/out- and all-degrees by linear regression."""
        
        def func(x, a, b, c):
            return a * np.exp(-b * x) + c
        
        
        ## all connections
        xdata = np.asarray(list(self.degree_hist.keys()))
        ydata = np.asarray(list(self.degree_hist.values()))
        
    
        popt, pcov = curve_fit(func, xdata, ydata, bounds=([-np.inf, 0, -np.inf], [+np.inf, 4, np.inf]))
        
        power_law_coeffs = {'popt': popt,'pcov': pcov}
        
        self.power_law_coeffs = power_law_coeffs
        
        
        ## in degrees
        xdata = np.asarray(list(self.in_degree_hist.keys()))
        ydata = np.asarray(list(self.in_degree_hist.values()))
        
    
        popt, pcov = curve_fit(func, xdata, ydata, bounds=([-np.inf, 0, -np.inf], [+np.inf, 4, np.inf]))
        
        power_law_coeffs = {'popt': popt,'pcov': pcov}
        
        self.in_power_law_coeffs = power_law_coeffs
        
        ## out degrees
        xdata = np.asarray(list(self.out_degree_hist.keys()))
        ydata = np.asarray(list(self.out_degree_hist.values()))
        
    
        popt, pcov = curve_fit(func, xdata, ydata, bounds=([-np.inf, 0, -np.inf], [+np.inf, 4, np.inf]))
        
        power_law_coeffs = {'popt': popt,'pcov': pcov}
        
        self.out_power_law_coeffs = power_law_coeffs
        
        # fig, axes = plt.subplots(1,1, figsize = (7,7))
        
        # plt.plot(xdata, ydata, '.', label='data')
        
        # xSpace = np.linspace(min(xdata), max(xdata), 100)
        
        # plt.plot(xSpace, func(xSpace, *popt), '-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
        
        # plt.xlabel('x - node degree')
        # plt.ylabel('y - frequency')
        # plt.legend()
        # plt.show()
        
        
    def compute_small_world_coeff(self):
        """Compute small-world coeffs sigma and omega."""
        s= nx.sigma(self.G.to_undirected(), niter=100, nrand=10, seed=None)[source]
        o= nx.sigma(self.G.to_undirected(), niter=100, nrand=10, seed=None)[source]
        
        
        self.small_wolrd_coeffs = {'sigma':s, 'omage':o}
        
    def analyticsPanel(self):
        """Create panel of graph, degree distribution, and network characteristics."""
        #fig, axs = plt.subplots(1,1, figsize = (14,7))
        
        #TODO make this the same line of code to be used as above
        def func(x, a, b, c):
            return a * np.exp(-b * x) + c
        
        
        ## plotting options
        options = {
        'node_color': 'lightsteelblue',
        'node_size': 250,
        'width': 1,
        'arrowstyle': '-|>',
        'arrowsize': 10,
        }
        
        # Set margins for the axes so that nodes aren't clipped
       # axs[0].gca()
        #axs[0].margins(0.20)
        #axs[0].axis("off")
    
        if (self.graph_type == "random") : 
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize = (15,7))
            
        if (self.graph_type == "scale-free"):
            fig, axes = plt.subplots(nrows=2, ncols=3, figsize = (15,7))
        
        
        ax = axes.flatten()

        ### draw graph        
        #nx.draw_networkx(self.G, ax=ax[0], pos=nx.kamada_kawai_layout(self.G), arrows=True, **options)
        # with labels
        nx.draw_networkx(self.G, ax=ax[0], pos=nx.kamada_kawai_layout(self.G), arrows=True, **options, with_labels=True,font_size =10, font_weight='regular')
         
        
        ax[0].set_axis_off()
        ax[0].set_title("Graph")
        
        
        ### add text about graph propertires
        
        ax[1].set_axis_off()
        ax[1].set_title("Graph Analysis")
        ax[1].plot(1,1)
        
        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    

        ax[1].text(0.05, 0.95, self.__str__() , transform=ax[1].transAxes, fontsize=10, verticalalignment='top', bbox=props)
        
        
        
        
        
        if (self.graph_type == "scale-free"):
            #### emtpy pane
            
            ax[2].set_axis_off()
            
    
            ####################### plot node degree
            axis = 3
            ax[axis].set_title("Undirected Node Degree ")
            
            #TODO: alter this to a numpoy array and sort by value  so that one
            xdata = np.asarray(list(self.degree_hist.keys()))
            ydata = np.asarray(list(self.degree_hist.values()))
            
           
            
            ax[axis].plot(xdata, ydata, '.', label='data', markersize=10)
            
            popt = self.power_law_coeffs['popt'] 
            
            xSpace = np.linspace(min(xdata), max(xdata), 100)
            
            ax[axis].plot(xSpace, func(xSpace, *popt), '-', label='fit:  y=%5.2f e^{-%5.2f} + %5.2f' % tuple(popt))
            
            ax[axis].set(xlabel="node degree", ylabel="frequency")
            ax[axis].legend()
            
            ###############  plot in node degree
            axis = 4
            ax[axis].set_title("In-Node Degree ")
            #TODO: alter this to a numpoy array and sort by value  so that one
            xdata = np.asarray(list(self.in_degree_hist.keys()))
            ydata = np.asarray(list(self.in_degree_hist.values()))
            

            ax[axis].plot(xdata, ydata, '.', markersize=10)
            
            popt = self.in_power_law_coeffs['popt'] 
            
            xSpace = np.linspace(min(xdata), max(xdata), 100)
            
            ax[axis].plot(xSpace, func(xSpace, *popt), '-', label='fit:  y=%5.2f e^{-%5.2f} + %5.2f' % tuple(popt))
            
            ax[axis].set(xlabel="node degree", ylabel="frequency")
            ax[axis].legend()

            


            ################## plot in node degree
            axis = 5
            ax[axis].set_title("Out-Node Degree ")
            #TODO: alter this to a numpoy array and sort by value  so that one
            xdata = np.asarray(list(self.out_degree_hist.keys()))
            ydata = np.asarray(list(self.out_degree_hist.values()))
            

            ax[axis].plot(xdata, ydata, '.', markersize=10)
            
            popt = self.out_power_law_coeffs['popt'] 
            
            xSpace = np.linspace(min(xdata), max(xdata), 100)
            
            ax[axis].plot(xSpace, func(xSpace, *popt), '-', label='fit:  y=%5.2f e^{-%5.2f} + %5.2f' % tuple(popt))
            
            ax[axis].set(xlabel="node degree", ylabel="frequency")
            ax[axis].legend()
           
      
        if (self.graph_type == "random"):
            
            ####################### plot node degree
            axis = 2
            ax[axis].set_title("Undirected Node Degree ")
            
            #TODO: alter this to a numpoy array and sort by value  so that one
            xdata = np.asarray(list(self.degree_hist.keys()))
            ydata = np.asarray(list(self.degree_hist.values()))
            
            
            ax[axis].plot(xdata, ydata, '.', label='data', markersize=10)
            
    
            ax[axis].set(xlabel="node degree", ylabel="frequency")
           # ax[axis].legend()
  
        
    def calcTotalEdges(self):
        """Assign total number of edges.
        
        Parameters
        ----------
        self.G : nx.graph.Graph
        graph to analyze
         
        Returns
        -------
        totalEdges : int
        total number of edges in graph
        """
        totalEdges = np.shape(np.array(self.G.edges))[0]
         
        self.totalEdges = totalEdges
            
   
    
        
        
        
    