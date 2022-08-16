#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 15:54:48 2022

@author: gordonkoehn
"""

import pandas as pd

from matplotlib import pyplot as plt

import scipy
from scipy import stats

###############################################################################
###############################################################################
if __name__ == '__main__':
    
    df = pd.read_csv ('Stage1_ASynchrony.csv', header=2)
    
    
    df = df[df['MostActive']==True]
    
    df_sync = df[df['asynchronous'] == False]
    df_async = df[df['asynchronous'] == True]
    
    
    #### per neuron group
    fig7, ax7 = plt.subplots(figsize = (5,5))
    
    #data = np.concatenate((rates_Hz_ex, rates_Hz_in))
    green_diamond = dict(markerfacecolor='g', marker='D')
    data = [df_sync['m_freq'], df_async['m_freq']]
    #ax7.set_title('Mean Firing Frequency')
    bp = ax7.boxplot(data, flierprops=green_diamond)
    ax7.set_xticklabels(("synchronous", "asynchronous"), size=10)
    ax7.set_xlabel('Activity Type')
    ax7.set_ylabel('mean firing frequency [Hz]')
    ax7.grid()
    
    #### per neuron group
    fig7, ax7 = plt.subplots(figsize = (5,5))
    
    #data = np.concatenate((rates_Hz_ex, rates_Hz_in))
    green_diamond = dict(markerfacecolor='g', marker='D')
    data = [df_sync['m_cv'], df_async['m_cv']]
    #ax7.set_title('Mean Coefficient Of Variation')
    bp = ax7.boxplot(data, flierprops=green_diamond)
    ax7.set_xticklabels(("synchronous", "asynchronous"), size=10)
    ax7.set_xlabel('Activity Type')
    ax7.set_ylabel('Mean Coefficient Of Variation')
    ax7.grid()
    
    #### per neuron group
    fig7, ax7 = plt.subplots(figsize = (5,5))
    
    #data = np.concatenate((rates_Hz_ex, rates_Hz_in))
    green_diamond = dict(markerfacecolor='g', marker='D')
    data = [df_sync['m_corr'], df_async['m_corr']]
    #ax7.set_title('Mean Coefficient Of Variation')
    bp = ax7.boxplot(data, flierprops=green_diamond)
    ax7.set_xticklabels(("synchronous", "asynchronous"), size=10)
    ax7.set_xlabel('Activity Type')
    ax7.set_ylabel('Mean Pairwise Correlation')
    ax7.grid()
    
    
    
    ###### AUC
    tTest = scipy.stats.ttest_ind(df_sync['AUC'], df_async['AUC'],equal_var=False )
    
    
    #### per neuron group
    fig7, ax7 = plt.subplots(figsize = (5,5))
    
    #data = np.concatenate((rates_Hz_ex, rates_Hz_in))
    green_diamond = dict(markerfacecolor='g', marker='D')
    data = [df_sync['AUC'], df_async['AUC']]
    #ax7.set_title('Mean Coefficient Of Variation')
    bp = ax7.boxplot(data, flierprops=green_diamond)
    ax7.set_xticklabels(("synchronous", "asynchronous"), size=10)
    ax7.set_xlabel('Activity Type')
    ax7.set_ylabel('ROC: Area Under Curve')
    ax7.grid()
    
    
    
    
    
    
    
    #### per neuron group
    fig8, ax8 = plt.subplots(figsize = (5,5))
    
    #data = np.concatenate((rates_Hz_ex, rates_Hz_in))
    green_diamond = dict(markerfacecolor='g', marker='D')
    data = [df_sync['AUC'], df_async['AUC']]
    #ax8.set_title('Mean Coefficient Of Variation')
    bp = ax8.violinplot(data, showmeans=True)

    ax8.set_xticks([1, 2])
    ax8.set_xticklabels(("synchronous", "asynchronous"), size=10)
    
    ax8.set_xlabel('Activity Type')
    ax8.set_ylabel('ROC: Area Under Curve')
    ax8.grid()
    
    