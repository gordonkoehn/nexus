# nexus
Investigating the performance of the cross-correlation method by English et al. [2017][1] of inferring functional connectivity in adaptive-exponential
integrate and fire (aEIF) neuron model by Brette et al. [2007][2] on small-scale neuronal networks of different activity patterns (synchronous & regular / asynchonous & regular) and topologies (random / scale-free.) 

## Acknowledgements

This code was written as part of a semester project at ETH Zürich / Bio Engeneering Laboratory for my MSc Biotechnolgoy degree. The project was under Kim Taehoons caring supervision. 

### Acknowledgements of Code
This project built upon the works other members of the reseach group at ETH, who generously provided their brainworks/code:
- Kim Taehoon 
    - /simulations/wp2*.py  -> originals & variations of provided scripts
    - /conInf -> structure of analyiss derived from his projects
- Christian donner 
    - /tools/spyCon -> package written for functional connectivity inferrence 

## Structure of Projects

core.py - is the main script to generate networks, run neuronal networks ontop, classify their activity and infer their functional connectivity. 
core_runner.py - allows to force asynchonous/synchonous or physical behaiour by restarting simulations of core

### modules
- classifySim : clasisfies the activity of neuronal simulations from /simulations/simData - (some scripts capable of parallel processing)
- conInf : contains scripts exploring the spyCon package and implements functional connectitiy inference with the usage of the spyCon package
- netGen :  generates networks using networkX of random and scale-free topologies and implemtns methods to anaylsis the topology
- simulations :  implements simulations of neuronal networks with the brain2 simulator and saves the results in /simulations/simData
- tools : /spycon package (provided by Christian donner) + helper scripts

## Abstract of Project 
In the pursuit to comprehend the enigmatic nature of our brain, understand-
ing operational principles of neural circuits is the main goal. The defining
characteristic of any circuit is its connectivity. Yet, investigating the physical
neuron to neuron connections in the living brain on a large scale is presently
still infeasible. Thus, other methods to learn about the connectivity of neu-
ronal circuits are being explored, as for instance via the neural activity. The
dawn of high-density, multi-electrode implants gives hope to record large-scale
neuronal activity on the single neuron level in next decades.
Given neuronal activity recordings, the functional connectivity of a network
may be inferred from statistical correlation. The term functional connectiv-
ity separates the connectivity found by correlation from the physical, called
structural connectivity. It may already contain the operational principles of
our brain we seek to find.
Here, we evaluate the performance of a widely used functional connectivity
inference method by English et al. [2017][1] on small scale networks. We gen-
erate neural activity in silico on a known random network structure, to then
evaluate the performance of the algorithm against it.
The model used to simulate neurons is the prominent adaptive-exponential
integrate and fire (aEIF) model by Brette et al. [2007][2], allowing to capture the
fundamental exponential and adapting behaviour of the action potential.
The performance of the connectivity inference algorithm is evaluated at the
extrema of synchrony of sensible network activity. Therefore an extensive
parametric study in the adaption and conductance space of the used neuron
model was conducted, successfully identifying regimes of a- and synchronous
activity.
Particular cases of very synchronous network activity lead to a poor perfor-
mance of the inference algorithm, yet this study fails to make quantitative
statements aimed for.
Further, an attempt to explore network activity and performance of the algo-
rithm at more neuro-physiological network topologies, namely scale-free net-
works is presented.


## Full Report
Soon to be on g15n.net/ETH/nexus


## Known Issues

### brian gets slow / cython catche full
After running many simulaitons (>10000), brian2 got very slow for me. Python itself was fine. 
The issue turned out to be cython catch filed from brian2 accumulating so that no catch is free, which aparently obstructe normal funciton of brian.
To fix run:

<python>
from brian2.__init__ import clear_cache) 
brian2.__init__.clear_cache('cython')
</python>

or delete all fined in ~/.cython/brian_extensions/ .. files are named *gnu.so or *.lock

if list of files it to long to delete by "rm *.lock" run 

<bash>
find . -name "*.lock" -print0 | xargs -0 rm
</bash>

## References
[1] English, D. F., McKenzie, S., Evans, T., Kim, K., Yoon, E., & Buzsáki, G. (2017). Pyramidal Cell-Interneuron Circuit Architecture and Dynamics in Hippocampal Networks. Neuron, 96(2), 505-520.e7. https://doi.org/10.1016/j.neuron.2017.09.033

[1] Brette, R., & Gerstner, W. (2005). Adaptive exponential integrate-and-fire model as an effective description of neuronal activity. Journal of Neurophysiology, 94(5), 3637–3642. https://doi.org/10.1152/jn.00686.2005
