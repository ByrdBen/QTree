# qtree
Branch analysis scripts for several qubit architectures. 
This module was produced in order to better understand the process of MIST in fluxonium qubits, 
however, it can more generally produce mappings between bare/dressed states of decoupled/coupled
Hamiltonians, and organize them in whichever way is deemed most useful. It is currently arranged 
in a manner most conducive to studying dispersive readout of various qubit types with various couplings,
but it is flexible enough that given bare/coupled Hamiltonians, we can perform branch analysis in a 
manner nominally identical to what is done in Dumas et al., here: https://arxiv.org/pdf/2402.06615 .

The main scripts are all contained within the directory qtree. There are some example notebooks
illustrating a sample workflow on a local PC, and condor_scripts which I have used to perform this analysis
on a high-performance-computing-cluster. For start-up instructions, I suggest looking at example_workflow.ipynb,
and playing with parameters/output data.

To install the package, `pip install qtree` will suffice. The requirements are fairly clean, since the
majority of the analysis is essentially a wrapper around SCQubits and QuTiP, the only other packages we
need are the usual suspects for numerically oriented scientific work like numpy, scipy and the like.

This package is under active development, contributions and suggestions are both welcome. Feel free to
reach out at byrdquantum@gmail.com with any questions, comments, concerns, ideas, anything.
