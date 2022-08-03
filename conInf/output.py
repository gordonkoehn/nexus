#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Outputs plots and metrics of the spycon_result and spycon_test.

Created on Tue Aug  2 10:06:48 2022

@author: gordonkoehn
"""

# import common stuff
import sys
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import networkx as nx

# import inference method
sys.path.append('../tools/spycon/src')
from sci_sccg import Smoothed_CCG
### get conncectivity test 
from spycon_tests import load_test, ConnectivityTest

sys.path.append('../tools')
import tools.adEx_util
import conInf.analyser


def visualization_english(Smoothed_CCG, times1: np.ndarray, times2: np.ndarray,
                  t_start: float, t_stop: float) -> (np.ndarray):
    """Compute Cross-Correlation Histrogram.

    # Adding Cross-Correlation methods from "Methods_Viz" from Christian's code
    """
    kernel = Smoothed_CCG.partially_hollow_gauss_kernel()
    counts_ccg, counts_ccg_convolved, times_ccg = Smoothed_CCG.compute_ccg(times1, times2, kernel, t_start, t_stop)
    
    return counts_ccg, counts_ccg_convolved, times_ccg 


def plot_ccg(coninf : Smoothed_CCG, spycon_test : ConnectivityTest, idx : int):
    """Plot Cross-Corrogram..
    
    Parameters
    ----------
    coninf : SpikeConnectivityInference
        Implementation of CCG method
    
    spycon_test : ConnectivityTest
    
    idx : int
        arb. index of ccg pair of neurons 
        (only pairs of edges where a true edge exists)
        
    Returns
    -------
    None - but, plot
       
    """
    ## get edges and ids
    # get rows/indices of marked_edges that contain connections
    edges = np.where(np.logical_and(spycon_test.marked_edges[:,2] != 0, np.logical_not(np.isnan(spycon_test.marked_edges[:,2]))))[0]
    # select arbitrary edge by order in marked edges
    idx = 4
    # get pre- and post-synaptic neuron to do the CCH for
    id1, id2 = spycon_test.marked_edges[edges[idx],:2]
    
    ## run corr correlation 
    times1, times2 = spycon_test.times[spycon_test.ids == id1], spycon_test.times[spycon_test.ids == id2]
    counts_ccg, counts_ccg_convolved, times_ccg = visualization_english(coninf, times1, times2, 0, 3600)
    
    # plot
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.axis('off')
    ax.fill_between([coninf.default_params['syn_window'][0] * 1e3, coninf.default_params['syn_window'][1] * 1e3], 0, np.amax(counts_ccg) + 20, color='C0', alpha=.5)
    ax.bar(times_ccg * 1e3, counts_ccg, width=coninf.default_params['binsize'] * 1e3, color='k', label='Data CCG')
    ax.plot(times_ccg * 1e3, counts_ccg_convolved, 'C0', label='Smoothed CCG', lw=2)
    ax.vlines([0], 0, np.amax(counts_ccg) + 20, lw=2, ls='--', color='gray')
    ax.hlines(40,-12,-8, 'r')
    ax.text(-10, 47, '5 ms', color='r')
    #ax.legend()
    ax.set_xlim([-15,15])
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Spike count')
    ax.set_ylim([0,np.amax(counts_ccg) + 10])
    ax.set_title('smoothed CCG')
    

def plot_all_ccgs(coninf : Smoothed_CCG, spycon_test : ConnectivityTest):
    """Plot all Cross-Corrograms.
    
    The CCG are only calulated for the cases where there are actual true connections.
    
    Parameters
    ----------
    coninf : SpikeConnectivityInference
        Implementation of CCG method
    
    spycon_test : ConnectivityTest
    
    idx : int
        arb. index of ccg pair of neurons 
        (only pairs of edges where a true edge exists)
        
    Returns
    -------
    None - but, plot
    """
    edges = np.where(np.logical_and(spycon_test.marked_edges[:,2] != 0, np.logical_not(np.isnan(spycon_test.marked_edges[:,2]))))[0]
  
    no_edges = len(edges)
    #no_edges = 23
    
    rows = int(no_edges/10)+1
    columns = 10
    
    fig, axs = plt.subplots(rows,columns, figsize=(10, 20), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = .5, wspace=.05)
    #fig.patch.set_facecolor('#E0E0E0')
    
    axs = axs.ravel()
    
    for i in range(rows*columns):
        axs[i].axis('off')
    
    for i in range(no_edges):    
        ## get edges and ids
        # get rows/indices of marked_edges that contain connections
        edges = np.where(np.logical_and(spycon_test.marked_edges[:,2] != 0, np.logical_not(np.isnan(spycon_test.marked_edges[:,2]))))[0]
        # select arbitrary edge by order in marked edges
        idx = i
        # get pre- and post-synaptic neuron to do the CCH for
        id1, id2 = spycon_test.marked_edges[edges[idx],:2]
    
        ## run corr correlation 
        times1, times2 = spycon_test.times[spycon_test.ids == id1], spycon_test.times[spycon_test.ids == id2]
        counts_ccg, counts_ccg_convolved, times_ccg = visualization_english(coninf, times1, times2, 0, 3600)
    
        # plot
        axs[i].axis('off')
    
        
        axs[i].fill_between([coninf.default_params['syn_window'][0] * 1e3, coninf.default_params['syn_window'][1] * 1e3], 0, np.amax(counts_ccg) + 20, color='C0', alpha=.5)
        axs[i].bar(times_ccg * 1e3, counts_ccg, width=coninf.default_params['binsize'] * 1e3, color='k', label='Data CCG')
        axs[i].plot(times_ccg * 1e3, counts_ccg_convolved, 'C0', label='Smoothed CCG', lw=2)
        axs[i].vlines([0], 0, np.amax(counts_ccg) + 20, lw=2, ls='--', color='gray')
        #axs[i].hlines(40,-12,-8, 'r')
        #axs[i].text(-10, 47, '5 ms', color='r')
        #ax.legend()
        axs[i].set_xlim([-15,15])
        axs[i].set_xlabel('Time [ms]')
        axs[i].set_ylabel('Spike count')
        axs[i].set_ylim([0,np.amax(counts_ccg) + 10])
        axs[i].set_title(f'{int(id1)} -->{int(id2)}', fontsize=8)
        
        #print(np.amax(counts_ccg) + 10)
        
        
    plt.savefig('all_CCGs.pdf')  
        #plt.ion()
        
        
def plot_ROC(test_metrics : pd.DataFrame):
    """Plot Receiver Operator Curve as defined by Christian.
    
    Parameters
    ----------
    test_metrics : pd.DataFrame
       test metrics of generating the ROC
       
       Definitiaon of pd.DataFrame
       
       metrics_dict['fpr'], metrics_dict['tpr'], metrics_dict['thresholds'] = metrics.roc_curve(y_true, y_score)
       metrics_dict['auc'] = metrics.roc_auc_score(y_true, y_score)
       metrics_dict['aps'] = metrics.average_precision_score(y_true, y_score)
       metrics_dict['prc_precision'], metrics_dict['prc_recall'], metrics_dict['prc_thresholds'] = metrics.precision_recall_curve(y_true, y_score)
       pred_con_mat = spycon_result.create_connectivity_matrix(conn_type='binary')
       y_pred = pred_con_mat[gt_edge_idx[0], gt_edge_idx[1]]
       metrics_dict['f1'] = metrics.f1_score(y_true, y_pred)
       metrics_dict['precision'] = metrics.precision_score(y_true, y_pred)
       metrics_dict['recall'] = metrics.recall_score(y_true, y_pred)
       metrics_dict['accuracy'] = metrics.accuracy_score(y_true, y_pred)
       metrics_dict['mcc'] = metrics.matthews_corrcoef(y_true, y_pred)
       
    Returns
    -------
    plot
    """
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize = (5,5))
    
    fpr, tpr, auc = tuple(test_metrics[['fpr', 'tpr', 'auc']].to_numpy()[0])
    axes.plot(fpr, tpr, color='blue')
    axes.plot([0,1],[0,1], color='gray', linestyle='--')
    
    # get best threshold
    bestthreshold = conInf.analyser.getBestThreshold(test_metrics)
    axes.plot(bestthreshold['fpr'],bestthreshold['tpr'], color='orange', marker='.', markersize = 10)
    #axes.text(.6,.07,'optimal\nthreshold = %.3f' %bestthreshold['threshold'])
    
    axes.text(.6,.0,'AUC = %.3f' %auc)
    axes.set(xlabel="False positive rate", ylabel="True positive rate")
    axes.set_title('Receiver Operating Curve')
  