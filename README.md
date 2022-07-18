# nexus
Investigating methods of inferring functional connectivity in integrate-and-fire//adapting exponential (adEX) neuronal network models.




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
