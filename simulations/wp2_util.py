#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 14:14:39 2022

@author: th
"""


import numpy as np
import sys

sys.path.append('/home/th/bsse_home/work_directory/wp1/python')

import neural_assembly
# import wp2_util
from elephant.statistics import isi, cv
import matplotlib.pyplot as plt

from itertools import combinations, permutations

# wp2 analysis util script. 
from sklearn import metrics

import pandas as pd




def get_neotrain(network, grid_path):
    example = network['save_name']
    N_size = network['save_name'].split('_')[0]
    
    uniq_tmps = network['valid_idx']
    example_data = np.load(grid_path + '/' + N_size + '/'+  example + '.npy', allow_pickle=True).item()
    result_file = example_data
    
    
    
    spktimes = dict()
    spktimes['ex']= result_file['ex_time'] # in ms.
    spktimes['in'] = result_file['in_time']
    
    spktimes['ex_tmp']= result_file['ex_idx']
    spktimes['in_tmp']= result_file['in_idx']
    
    
    # concatenate
    
    concat_spk = np.concatenate((spktimes['ex'], spktimes['in']))
    concat_tmp = np.concatenate((spktimes['ex_tmp'], spktimes['in_tmp']))
    sort_idx = np.argsort(concat_spk)
    
    concat_spk = concat_spk[sort_idx]
    concat_tmp = concat_tmp[sort_idx]
    
    
    # take out first  (to clear external drives)
    rmv = 1000
    kick_idx = np.where(concat_spk<rmv)[0]
    
    concat_spk = np.delete(concat_spk,kick_idx)
    concat_tmp = np.delete(concat_tmp,kick_idx)
    
        
    # subset those belonging to uniq_tmp
    
    bool_idx = np.isin(concat_tmp, uniq_tmps)
    
    concat_spk = concat_spk[bool_idx]
    concat_tmp = concat_tmp[bool_idx]
    
    
        
    na_train = neural_assembly.Spiketrain(concat_spk, concat_tmp)
    na_train.set_trains()
    neo_train = na_train.to_neotrain()
    
    
    return na_train, concat_spk, concat_tmp



def pass_check(result):
    verdict = dict()
    
    path = result
    
    dt = 0.1 #ms
    
    
    result_file = np.load(path, allow_pickle=True).item()
    
    spktimes = dict()
    spktimes['ex']= result_file['ex_time'] # in ms.
    spktimes['in'] = result_file['in_time']
    
    spktimes['ex_tmp']= result_file['ex_idx']
    spktimes['in_tmp']= result_file['in_idx']
    
    
    # concatenate
    
    concat_spk = np.concatenate((spktimes['ex'], spktimes['in']))
    concat_tmp = np.concatenate((spktimes['ex_tmp'], spktimes['in_tmp']))
    sort_idx = np.argsort(concat_spk)
    
    concat_spk = concat_spk[sort_idx]
    concat_tmp = concat_tmp[sort_idx]
    
    
    # take out first  (to clear external drives)
    rmv = 1000
    kick_idx = np.where(concat_spk<rmv)[0]
    
    concat_spk = np.delete(concat_spk,kick_idx)
    concat_tmp = np.delete(concat_tmp,kick_idx)
    
    
    sim_t = result_file['params']['sim_time']
    if(len(concat_spk)==0 or np.max(concat_spk)<(sim_t-1)*1000):
        print(path + ' didn''t generate lasting activity')
        return
    
    
    
    
    # use neural assmebly
    
    na_train = neural_assembly.Spiketrain(concat_spk, concat_tmp)
    na_train.set_trains()
    neo_train = na_train.to_neotrain()
    
    # check firing rates 
    fr_vec = []
    for entry in neo_train:
        
        fr_vec.append(len(entry)/(sim_t-(rmv/1000)))
        
    # kick out indices those are smaller than 
    
    #uniq_tmps
    uniq_tmps = np.unique(concat_tmp)
    
    
    dynamic_thres = 1/(sim_t-rmv/1000)
    kick_idx = uniq_tmps[np.where(np.array(fr_vec)<=dynamic_thres)[0]]
    
    kick_bool_idx = ~np.isin(concat_tmp, kick_idx)
    
    concat_spk = concat_spk[kick_bool_idx]
    concat_tmp = concat_tmp[kick_bool_idx]
    
    na_train = neural_assembly.Spiketrain(concat_spk, concat_tmp)
    na_train.set_trains()
    neo_train = na_train.to_neotrain()
    
    fr_vec = []
    for entry in neo_train:
        sim_t = result_file['params']['sim_time']
        fr_vec.append(len(entry)/(sim_t-(rmv/1000)))
        
    params = result_file['params']
    save_name = '_'.join([str(params['N']), str(params['a']),str(params['b']), str(params['sim_time'])])
    # if(np.mean(fr_vec)>30):
    #     print('too strong activity passing... ' +  save_name)
    #     return
    
    
    
    # count_mat = na_train.to_countmat(5) # 5ms bin size
    # cov_mat, prec_mat = na_train.compute_covmat(5)
    
    
    # final checking of valid indices
    uniq_tmps = np.unique(concat_tmp)
    
    # quantify CV_ISIs and correlation 
    
    CV=[]
    for train in neo_train:
        CV.append(cv(isi(train)))
    
    # destexhe2009, CV_mean >1 for irregular spikes
    # verdict['CV'] = CV
    verdict['CV_mean'] = np.mean(CV)
    verdict['CV_std'] = np.std(CV)
    
    # corr mean <0.1 for Asynchronous trains
    corr_mat = na_train.compute_corr(5)
    
    # verdict['corr_mat'] = corr_mat
    verdict['corr_mean']= np.mean(np.triu(corr_mat, k=1))
    verdict['corr_std'] = np.std(np.triu(corr_mat, k=1))
    
    verdict['mean_fr']= np.mean(fr_vec)
    verdict['valid_idx']=uniq_tmps 
    
    
    
    # label this network act. for its synchronicity
    if( np.logical_and(verdict['CV_mean']>1 , verdict['corr_mean']<0.1)):
        verdict['label'] = 'async'
        
    else:
        verdict['label']='sync'
        
    
    
    # verdict['sim_data']=result_file
    verdict['save_name']=save_name
    
    
    
    return verdict

def get_gt_conn(sim_data, verdict):
    
    gt_conn =dict()
    keys = ['conn_ee', 'conn_ei', 'conn_ii', 'conn_ie']
    for key in keys:
        gt_conn[key]=convert_conn_idx(verdict, sim_data[key])
        
    return gt_conn
    


def get_sampled_conn(gt_conn, sampled):
    
    samp_conn=dict()
    keys = list(gt_conn.keys())
    
    sampled_idx = sampled['all']
    for key in keys:
        entry = gt_conn[key]
        pre_check = np.isin(entry[0,:], sampled_idx)
        post_check = np.isin(entry[1,:], sampled_idx)
        
        check = np.logical_and(pre_check, post_check)
        
        samp_conn[key]=gt_conn[key][:,check]
        
    samp_conn['all']= np.concatenate(list(samp_conn.values()),1)    
   
        
    return samp_conn

    
def make_marked_edges(samp_conn):
    all_conn = samp_conn['all']
    all_idx= np.unique(all_conn)
    
    pairs = np.array(list(permutations(all_idx, 2)))
    
    marked_edges = np.zeros((pairs.shape[0],pairs.shape[1]+1))
    
    marked_edges[:,0]= pairs[:,0]
    marked_edges[:,1]= pairs[:,1]
    
    for ii in range(all_conn.shape[1]):
        idx = np.where((pairs == all_conn[:,ii]).all(axis=1))[0]
        marked_edges[idx,2]=1
    
    
    return marked_edges



def convert_conn_idx(verdict, ee_con):
    
    uniq_tmps = verdict['valid_idx']
    
    # first delete non-valid pairs
    pre_check = np.isin(ee_con[0,:], uniq_tmps)    
    post_check = np.isin(ee_con[1,:], uniq_tmps)
        
    check = pre_check + post_check
    check_vec = check>0
    
    ee_con_fil = ee_con[:,check_vec]
    
    fcol = np.searchsorted(uniq_tmps, ee_con_fil[0,:], side='left')
    scol = np.searchsorted(uniq_tmps, ee_con_fil[1,:], side='left')
    
    ee_con_conv = np.array([fcol, scol])
    
    return ee_con_conv


def eval_performance(result_mat):
    """ 
    adaptation of Donner et al. script 
    """
    
    metrics_dict = {}
    metrics_dict['compute_time']= result_mat['compute_time']
    true_con_mat = result_mat['true_con']
    pred_con_mat = result_mat['thres_con']
    score_con_mat = result_mat['score_con']
    
    # true_con_mat = self.create_connectivity_matrix(spycon_result.nodes)
    # score_con_mat = spycon_result.create_connectivity_matrix(conn_type='stats')
    gt_edge_idx = np.where(np.logical_not(np.isnan(true_con_mat)))
    y_true = np.zeros(len(gt_edge_idx[0]))
    y_true[np.nonzero(true_con_mat[gt_edge_idx[0], gt_edge_idx[1]])] = 1
    y_score = score_con_mat[gt_edge_idx[0], gt_edge_idx[1]]
    metrics_dict['fpr'], metrics_dict['tpr'], metrics_dict['thresholds'] = metrics.roc_curve(y_true, y_score)
    metrics_dict['auc'] = metrics.roc_auc_score(y_true, y_score)
    metrics_dict['aps'] = metrics.average_precision_score(y_true, y_score)
    metrics_dict['prc_precision'], metrics_dict['prc_recall'], metrics_dict['prc_thresholds'] = metrics.precision_recall_curve(y_true, y_score)
    # pred_con_mat = spycon_result.create_connectivity_matrix(conn_type='binary')
    y_pred = pred_con_mat[gt_edge_idx[0], gt_edge_idx[1]]
    metrics_dict['f1'] = metrics.f1_score(y_true, y_pred)
    metrics_dict['precision'] = metrics.precision_score(y_true, y_pred)
    metrics_dict['recall'] = metrics.recall_score(y_true, y_pred)
    metrics_dict['accuracy'] = metrics.accuracy_score(y_true, y_pred)
    metrics_dict['mcc'] = metrics.matthews_corrcoef(y_true, y_pred)
    metrics_df = pd.DataFrame([metrics_dict.values()], index=[0], columns=metrics_dict.keys())
    return metrics_df



def create_connectivity_matrix(nodes, marked_edges):
    pairs = np.array(list(combinations(nodes, 2)))
    pairs = np.vstack([pairs, pairs[:, ::-1]])
    con_matrix = np.empty((len(nodes), len(nodes)))
    con_matrix[:,:] = np.nan
    edges_to_consider = np.where(np.logical_and(np.isin(marked_edges[:,0], nodes),
                                                      np.isin(marked_edges[:,1], nodes)))[0]
    idx1 = np.searchsorted(nodes, marked_edges[edges_to_consider,0])
    idx2 = np.searchsorted(nodes, marked_edges[edges_to_consider,1])
    con_matrix[idx1, idx2] = marked_edges[edges_to_consider,2]

    return con_matrix