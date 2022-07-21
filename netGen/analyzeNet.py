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



class netInspector():
    """Implements a vararity of network analytics.
    
    It was intended to be used for directed graphs.
    """

    def __init__(self, G: nx.classes.graph.Graph):
        """Initialize netInspector object with a given Graph G.
        
        Parameters
        ----------
        G : networkx.classes.graph.Graph
            Graph to analyze
        """
        self.G = G
        self.degree_sequence = None
        self.degree_hist = None
        self.node_degree_avg = None
        self.average_clustering = None
        self.average_shortest_path = None
        # self.small_wolrd_coeffs = None
                
        
        

    def __str__(self):
        """
        Print network properties.

        Returns
        -------
        None.

        """
        output = ""
        
        if (self.degree_sequence is not None):
            output += f"Degree sequence {self.degree_sequence} \n\n"
        if (self.degree_hist is not None):
             output +=  "== Degree histogram == \n"
             output += "degree #nodes \n"
             for d in self.degree_hist:
                 output += f"{d:4} {self.degree_hist[d]:6} \n"
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
        
        return output
    
    def eval_all(self):
        """Evaluate all network characteristics."""
        self.degree_seq_hist()
        self.avg_node_degree()
        self.compute_clustering()
        #self.compute_small_world_coeff()
        self.compute_shortest_path()
        self.power_law_fit()
    
  

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
        """Calculate power-law exponent by linear regression."""
        
        def func(x, a, b, c):
            return a * np.exp(-b * x) + c
        
        
        #TODO: alter this to a numpoy array and sort by value  so that one
        xdata = np.asarray(list(self.degree_hist.keys()))
        ydata = np.asarray(list(self.degree_hist.values()))
        
    
        popt, pcov = curve_fit(func, xdata, ydata, bounds=([-np.inf, 1.9, -np.inf], [+np.inf, 3.1, np.inf]))
        
        
        print(popt)
        
        
        fig, axes = plt.subplots(1,1, figsize = (7,7))
        
       
        
        plt.plot(xdata, ydata, '.', label='data')
        
        xSpace = np.linspace(min(xdata), max(xdata), 100)
        
        plt.plot(xSpace, func(xSpace, *popt), '-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
        
        plt.xlabel('x - node degree')
        plt.ylabel('y - frequency')
        plt.legend()
        plt.show()
        
        
    
    
    
    
        
        
        
    