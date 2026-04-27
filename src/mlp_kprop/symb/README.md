### Symbolic cumulant propagation
This directory implements a symbolic version of cumulant propagation.
Instead of computing cumulant estimates at each layer, 
we instead construct the tensor network corresponding to running cumulant propagation through the entire network,
then evaluate that network.
Because the optimal contraction path may not be layerwise, this can be asymptotically faster than standard cumulant propagation (though, in practice, the constants are too large to be practical).

Note that this currently implements the symbolic version of the deprecated `kprop_ds.py` algorithm instead of `kprop_harmonic.py`.