#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 11:43:52 2022

@author: gordonkoehn

This script was created to tinker with the sci_sccg.py script in my attempt to digest its contents.

Here I am trying to understand the partially_hollow_gauss_kernel().
"""
# import inference method
import sys
sys.path.append('../tools/spycon/src')
from spycon_inference import SpikeConnectivityInference
import elephant
import neo
import quantities
import numpy
from scipy.stats import poisson
import multiprocessing
from itertools import repeat
import matplotlib
import matplotlib.pyplot as plt


class Smoothed_CCG_explore(SpikeConnectivityInference):

    def __init__(self, params: dict = {}):
        """ Implements the CCG method from

        English, Daniel Fine, et al. "Pyramidal cell-interneuron circuit architecture and dynamics in hippocampal networks." Neuron 96.2 (2017): 505-520.

        :param params:
            'binsize': Time step (in seconds) that is used for time-discretization. (Default=.4e-3)
            'hf': Half fraction of the Gaussian kernel at zero lag (0<=hf<=1). (Default=0.6)
            'gauss_std': Standard deviation of Gaussian kernel (in seconds). (Default=0.01)
            'syn_window': Time window, which the CCG is check for peaks (in seconds). (Default=(.8e-3,2.8e-3))
            'ccg_tau': The maximum lag which the CCG is calculated for (in seconds). (Default=(20e-3))
            'alpha': Value, that is used for thresholding p-values (0<alpha<1). (Default=.01)
        :type params: dict
        """
        super().__init__(params)
        self.method = 'sccg'
        self.default_params = {'binsize': 1e-3,
                               'hf': .6,
                               'gauss_std': 0.01,
                               'syn_window': (0,5e-3),
                               'ccg_tau': 20e-3,
                               'alpha': .01}

    def partially_hollow_gauss_kernel(self) -> numpy.ndarray:
        """ Computes the partially hollow Gaussian kernel.
    
        :return: 1D array with kernel.
        :rtype: numpy.ndarray
        """
        # get parameters
        binsize = self.params.get('binsize', self.default_params['binsize'])
        hollow_fraction = self.params.get('hf', self.default_params['hf'])
        std = self.params.get('gauss_std', self.default_params['gauss_std'])
        # define the timewindow over which the kernel is defined... just hardcoded as 5 times the std
        kernel_tau = 5. * std
        # get the discrete timepoints to eval the kernel fn for
        delta = numpy.arange(-kernel_tau, kernel_tau + binsize, binsize)
        # plain gauss centered around zero
        kernel = numpy.exp(- .5 * (delta / std) ** 2.)
        # get index of computational zero
        zero_idx = numpy.where(numpy.isclose(delta, 0))[0]
        # set kernel for computational zero to a lower value...by hollow fraction ...0.6
        kernel[zero_idx] = hollow_fraction * kernel[zero_idx]
        # normalize
        kernel /= numpy.sum(kernel)
        return kernel, delta
    
    
if __name__ == '__main__':
        # define inference method
        coninf = Smoothed_CCG_explore() # English2017  
        kernel , delta = coninf.partially_hollow_gauss_kernel()
        print(delta)
        print(numpy.isclose(delta, 0))
       
        Fig=plt.figure(figsize=(20,28))
        plt.subplots_adjust(hspace=0.7, wspace=0.4)

        matplotlib.rc('xtick', labelsize=30) 
        matplotlib.rc('ytick', labelsize=30) 


        figa=Fig.add_subplot()
    
        plt.title('Raster Plot', fontsize=30)
        plt.scatter(delta, kernel, color='red',s=50,label="FS")
       
        plt.legend(loc='best', fontsize=30)
        plt.xlabel('Time [s]', fontsize=30)
        plt.ylabel('Neuron Index', fontsize=30)
        Fig.show()