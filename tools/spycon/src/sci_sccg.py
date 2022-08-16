#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script was provided by Christian Donner.

@author: christiando 
"""


from spycon_inference import SpikeConnectivityInference
import elephant
import neo
import quantities
import numpy
from scipy.stats import poisson
import multiprocessing
from itertools import repeat

class Smoothed_CCG(SpikeConnectivityInference):

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

    def _infer_connectivity(self, times: numpy.ndarray, ids: numpy.ndarray, pairs: numpy.ndarray) -> (numpy.ndarray,):
        """ CCG connectivity inference.

        :param times: Spike times in seconds.
        :type times: numpy.ndarray
        :param ids: Unit ids corresponding to spike times.
        :type ids: numpy.ndarray
        :param pairs: Array of [pre, post] pair node IDs, which inference will be done for.
        :type pairs: numpy.ndarray

        :return: Returns
            1) nodes:   [number_of_nodes], with the node labels.
            2) weights: [number_of_edges], with a graded strength of connections.
            3) stats:   [number_of_edges, 3], containing a fully connected graph, where the first columns are outgoing
                        nodes, the second the incoming node, and the third row contains the statistic, which was used to
                        decide, whether it is an edge or not. A higher value indicates, that an edge is more probable.
            4) threshold: a float that considers and edge to be a connection, if stats > threshold.
        :rtype: tuple
        """
        alpha = self.params.get('alpha', self.default_params['alpha'])
        nodes = numpy.unique(ids)
        weights = []
        stats = []
        num_connections_to_test = pairs.shape[0]
        conn_count = 0
        binsize = self.params.get('binsize', self.default_params['binsize'])
        syn_window = self.params.get('syn_window', self.default_params['syn_window'])
        bonf_corr = numpy.round((syn_window[1] - syn_window[0]) / binsize)
        print_step = numpy.amin([1000,numpy.round(num_connections_to_test / 10.)])
        pairs_already_computed = numpy.empty((0,2))
        for pair in pairs:
            id1, id2 = pair
            if not any(numpy.prod(pairs_already_computed == [id2, id1], axis=1)):
                if conn_count % print_step == 0:
                    print('Test connection %d of %d (%d %%)' %(conn_count, num_connections_to_test,
                                                               100*conn_count/num_connections_to_test))
                logpval1, weight1, logpval2, weight2, pair = self.test_connection_pair(times, ids, pair)
                weights.append(weight1)
                stats.append(numpy.array([id1, id2, logpval1]))
                pairs_already_computed = numpy.vstack([pairs_already_computed, numpy.array([id1, id2])])
                if any(numpy.prod(pairs == [id2, id1], axis=1)):
                    weights.append(weight2)
                    stats.append(numpy.array([id2, id1, logpval2]))
                    conn_count += 2
                    pairs_already_computed = numpy.vstack([pairs_already_computed, numpy.array([id2, id1])])
                else:
                    conn_count += 1
        print('Test connection %d of %d (%d %%)' %(conn_count, num_connections_to_test,
                                                               100*conn_count/num_connections_to_test))

        weights = numpy.vstack(weights)
        stats = numpy.vstack(stats)
        # change pvalues such that predicted edges have positive value
        stats[:,2] =  - stats[:,2]
        threshold = - numpy.log(alpha/bonf_corr)
        return nodes, weights, stats, threshold

    def _infer_connectivity_parallel(self, times: numpy.ndarray, ids: numpy.ndarray,
                                     pairs: numpy.ndarray, num_cores: int) -> (numpy.ndarray,):
        """ CCG connectivity inference. Parallel version.

        :param times: Spike times in seconds.
        :type times: numpy.ndarray
        :param ids: Unit ids corresponding to spike times.
        :type ids: numpy.ndarray
        :param pairs: Array of [pre, post] pair node IDs, which inference will be done for.
        :type pairs: numpy.ndarray

        :return: Returns
            1) nodes:   [number_of_nodes], with the node labels.
            2) weights: [number_of_edges], with a graded strength of connections.
            3) stats:   [number_of_edges, 3], containing a fully connected graph, where the first columns are outgoing
                        nodes, the second the incoming node, and the third row contains the statistic, which was used to
                        decide, whether it is an edge or not. A higher value indicates, that an edge is more probable.
            4) threshold: a float that considers and edge to be a connection, if stats > threshold.
        :rtype: tuple
        """
        alpha = self.params.get('alpha', self.default_params['alpha'])
        nodes = numpy.unique(ids)
        weights = []
        stats = []
        binsize = self.params.get('binsize', self.default_params['binsize'])
        syn_window = self.params.get('syn_window', self.default_params['syn_window'])
        bonf_corr = numpy.round((syn_window[1] - syn_window[0]) / binsize)
        pairs_already_computed = numpy.empty((0,2))
        pairs_to_compute = []
        for pair in pairs:
            id1, id2 = pair
            if not any(numpy.prod(pairs_already_computed == [id2, id1], axis=1)):
                pairs_to_compute.append(pair)
                pairs_already_computed = numpy.vstack([pairs_already_computed, numpy.array([id1, id2])])
                if any(numpy.prod(pairs == [id2, id1], axis=1)):
                    pairs_already_computed = numpy.vstack([pairs_already_computed, numpy.array([id2, id1])])

        job_arguments = zip(repeat(times), repeat(ids), pairs_to_compute)
        pool = multiprocessing.Pool(processes=num_cores)
        results = pool.starmap(self.test_connection_pair, job_arguments)
        pool.close()

        for result in results:
            logpval1, weight1, logpval2, weight2, pair = result
            id1, id2 = pair
            weights.append(weight1)
            stats.append(numpy.array([id1, id2, logpval1]))
            if any(numpy.prod(pairs == [id2, id1], axis=1)):
                weights.append(weight2)
                stats.append(numpy.array([id2, id1, logpval2]))

        weights = numpy.vstack(weights)
        stats = numpy.vstack(stats)
        # change pvalues such that predicted edges have positive value
        stats[:,2] =  - stats[:,2]
        threshold = - numpy.log(alpha/bonf_corr)
        return nodes, weights, stats, threshold


    def test_connection_pair(self, times: numpy.ndarray, ids: numpy.ndarray, pair: tuple) -> (float):
        """ Test connections in both directions.

        :param times: Spike times in seconds.
        :type times: numpy.ndarray
        :param ids: Unit ids corresponding to spike times.
        :type ids: numpy.ndarray
        :param pair: Node pair.
        :type pair: tuple
        :return: pval and weight for edge id1->id2, followed by pval and weight for id2->id1
        :rtype: (float)
        """
        id1, id2 = pair
        binsize = self.params.get('binsize', self.default_params['binsize'])
        kernel = self.partially_hollow_gauss_kernel()
        t_start, t_stop = numpy.amin(times) - binsize, numpy.amax(times) + binsize
        times1 = times[ids == id1]
        times2 = times[ids == id2]
        counts_ccg, counts_ccg_convolved, times_ccg = self.compute_ccg(times1, times2, kernel, t_start, t_stop)

        pval1, weight1 = self._test_connection(counts_ccg, counts_ccg_convolved, times_ccg, len(times1))
        pval2, weight2 = self._test_connection(counts_ccg[::-1], counts_ccg_convolved[::-1], times_ccg, len(times2))
        return pval1, weight1, pval2, weight2, pair

    def _test_connection(self, counts_ccg: numpy.ndarray, counts_ccg_convolved: numpy.ndarray, times_ccg: numpy.ndarray,
                         num_presyn_spikes: int):
        """ Test a connection, given a CCG and a convolved CCG.

        :param counts_ccg:
        :type counts_ccg:
        :param counts_ccg_convolved:
        :type counts_ccg_convolved:
        :param times_ccg:
        :type times_ccg:
        :param num_presyn_spikes:
        :type num_presyn_spikes:
        :return:
        :rtype:
        """
        syn_window = self.params.get('syn_window', self.default_params['syn_window'])
        pos_time_points = numpy.logical_and(times_ccg >= syn_window[0], times_ccg < syn_window[1])
        lmbda_slow = numpy.amax(counts_ccg_convolved[pos_time_points])
        num_spikes_max = numpy.amax(counts_ccg[pos_time_points])
        num_spikes_min = numpy.amin(counts_ccg[pos_time_points])
        #
        #pfast_max = 1. - poisson.cdf(num_spikes_max - 1., mu=lmbda_slow) - .5 * poisson.pmf(num_spikes_max, mu=lmbda_slow)
        #pfast_min = 1. - poisson.cdf(num_spikes_min - 1., mu=lmbda_slow) - .5 * poisson.pmf(num_spikes_min, mu=lmbda_slow)
        logpfast_max = numpy.logaddexp(poisson.logcdf(num_spikes_max-1, mu=lmbda_slow), poisson.logpmf(num_spikes_max, mu=lmbda_slow) - numpy.log(2))
        if numpy.abs(logpfast_max) > 1e-4:
            logpfast_max = numpy.log(-numpy.expm1(logpfast_max))
        else:
            if logpfast_max > -1e-20:
                logpfast_max = -1e-20
            #print(logpfast_max)
            logpfast_max = numpy.log(-logpfast_max)
        logpfast_min = numpy.logaddexp(poisson.logcdf(num_spikes_min-1, mu=lmbda_slow), poisson.logpmf(num_spikes_min, mu=lmbda_slow) - numpy.log(2))
        #if numpy.abs(logpfast_max) > 1e-4:
        #    logpfast_max = numpy.log(-numpy.expm1(x))
        #else:
        #    logpfast_max = numpy.log(-x)
        # num_pos_time_points = numpy.sum(pos_time_points)
        # anticausal_idx = numpy.where(times_ccg <= 0)[0][-num_pos_time_points:]
        # lmbda_anticausal = numpy.amax(counts_ccg_convolved[anticausal_idx])
        # pcausal = 1. - poisson.cdf(num_spikes - 1, mu=lmbda_anticausal) - .5 * poisson.pmf(num_spikes,
        #                                                                                    mu=lmbda_anticausal)
        # pval = max(pfast, pcausal, 0)
        pval_max = logpfast_max#max(pfast_max,0)
        pval_min = logpfast_min#max(1. - pfast_min,0)
        pval = numpy.amin([pval_max, pval_min])
        weight = numpy.sum(counts_ccg[pos_time_points] - counts_ccg_convolved[pos_time_points]) / num_presyn_spikes
        return pval, weight

    def partially_hollow_gauss_kernel(self) -> numpy.ndarray:
        """ Computes the partially hollow Gaussian kernel.

        :return: 1D array with kernel.
        :rtype: numpy.ndarray
        """
        binsize = self.params.get('binsize', self.default_params['binsize'])
        hollow_fraction = self.params.get('hf', self.default_params['hf'])
        std = self.params.get('gauss_std', self.default_params['gauss_std'])
        kernel_tau = 5. * std
        delta = numpy.arange(-kernel_tau, kernel_tau + binsize, binsize)
        kernel = numpy.exp(- .5 * (delta / std) ** 2.)
        zero_idx = numpy.where(numpy.isclose(delta, 0))[0]
        kernel[zero_idx] = hollow_fraction * kernel[zero_idx]
        kernel /= numpy.sum(kernel)
        return kernel

    def compute_ccg(self, times1: numpy.ndarray, times2: numpy.ndarray,
                    kernel: numpy.ndarray, t_start: float, t_stop: float) -> (numpy.ndarray):
        """ Computes the crosscorrelogramm.

        :param times1:
        :type times1:
        :param times2:
        :type times2:
        :param kernel:
        :type kernel:
        :param t_start:
        :type t_start:
        :param t_stop:
        :type t_stop:
        :return:
        :rtype:
        """
        binsize = self.params.get('binsize', self.default_params['binsize'])
        neo_spk_train1 = neo.SpikeTrain(times1, units=quantities.second, t_start=t_start, t_stop=t_stop)
        neo_spk_train2 = neo.SpikeTrain(times2, units=quantities.second, t_start=t_start, t_stop=t_stop)
        st1 = elephant.conversion.BinnedSpikeTrain(neo_spk_train1, bin_size=binsize * quantities.second, tolerance=None)
        st2 = elephant.conversion.BinnedSpikeTrain(neo_spk_train2, bin_size=binsize * quantities.second, tolerance=None)
        ccg_tau = self.params.get('ccg_tau', self.default_params['ccg_tau'])
        ccg_bins = int(numpy.ceil(ccg_tau / binsize))
        ccg_bins_eff = numpy.amax([int(numpy.ceil(len(kernel) / 2)), ccg_bins])
        ccg = elephant.spike_train_correlation.cross_correlation_histogram(st1, st2,
                                                                           window=[-ccg_bins_eff, ccg_bins_eff],
                                                                           border_correction=False, binary=False,
                                                                           kernel=None, method='memory')
        counts_ccg = ccg[0][ccg_bins_eff - ccg_bins:ccg_bins_eff + ccg_bins + 1, 0].magnitude.T[0]
        times_ccg = ccg[0].times.magnitude[ccg_bins_eff - ccg_bins:ccg_bins_eff + ccg_bins + 1]
        ccg_convolved = elephant.spike_train_correlation.cross_correlation_histogram(st1, st2,
                                                                                     window=[-ccg_bins_eff,
                                                                                             ccg_bins_eff],
                                                                                     border_correction=False,
                                                                                     binary=False,
                                                                                     kernel=kernel, method='memory')
        counts_ccg_convolved = ccg_convolved[0][ccg_bins_eff - ccg_bins:ccg_bins_eff + ccg_bins + 1, 0].magnitude.T[0]
        return counts_ccg, counts_ccg_convolved, times_ccg