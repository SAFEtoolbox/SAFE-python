
"""
This script applies Variance-Based Sensitivity Analysis to the Ishigami-Homma
function. This function is commonly used to test approximation procedures of
variance-based indices because its output variance, first-order and
total-order indices (or 'main' and 'total' effects) can be analytically
computed. Therefore this script mainly aims at analyzing the accuracy
and convergence of the function 'VBSA.vbsa_indices'.

This script was prepared by Fanny Sarrazin, 2019
fanny.sarrazin@ufz.de
"""

#%% Step 1: (import python modules)

from __future__ import division, absolute_import, print_function

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

import safepython.VBSA as VB # module to perform VBSA
import safepython.plot_functions as pf # module to visualize the results
from safepython.model_execution import model_execution # module to execute the model
from safepython.sampling import AAT_sampling  # module to perform the input sampling

from safepython.ishigami_homma import ishigami_homma_function

#%% Step 2 (setup the model)

# Number of uncertain parameters subject to SA:
M = 3

# Parameter ranges:
xmin = -np.pi
xmax = np.pi
# Parameter distributions:
distr_fun = st.uniform # uniform distribution
# The shape parameters of the uniform distribution are the lower limit and the
# difference between lower and upper limits:
distr_par = [xmin, xmax - xmin]

# Define output:
fun_test = ishigami_homma_function

#%% Step 3 (sampling and model execution)

samp_strat = 'lhs' # Latin Hypercube
N = 3000  #  Number of samples
X = AAT_sampling(samp_strat, M, distr_fun, distr_par, 2*N)
XA, XB, XC = VB.vbsa_resampling(X)

# Run the model and compute selected model output at sampled parameter
# sets:
YA = model_execution(fun_test, XA,)# shape (N, )
YB = model_execution(fun_test, XB) # shape (N, )
YC = model_execution(fun_test, XC) # shape (N*M, )

#%% Step 4 (Compute first-order and total-order variance-based indices)

# Compute the exact values of the output variance (V) and of the first-order
# (Si_ex) and total-order (STi_ex) variance-based sensitivity indices (this is
# possible in thisvery specific case because V, Si_ex and STi_ex can be
# computed analytically):
_, V, Si_ex, STi_ex = ishigami_homma_function(np.random.random((M,)))

# Compute main (first-order) and total effects:
Si, STi = VB.vbsa_indices(YA, YB, YC, M)

# Plot main and total effects and compare the values estimated by the function
# 'VBSA.vbsa_indices' with the exact values:
S = np.zeros((2, M))
S[0, :] = Si
S[1, :] = Si_ex
ST = np.zeros((2, M))
ST[0, :] = STi
ST[1, :] = STi_ex
plt.figure() # plot both in one plot:
plt.subplot(121)
pf.boxplot2(S, Y_Label='main effects')
plt.subplot(122)
pf.boxplot2(ST, Y_Label='total effects', legend=['estimated', 'exact'])
plt.show()

# Analyze convergence of sensitivity indices:
NN = np.linspace(N/5, N, 5).astype(int)
Sic, STic = VB.vbsa_convergence(YA, YB, YC, M, NN)
plt.figure()
plt.subplot(121)
pf.plot_convergence(Sic, NN*(M+2), SExact=Si_ex, X_Label='model evals', Y_Label='main effects')
plt.subplot(122)
pf.plot_convergence(STic, NN*(M+2), SExact=STi_ex, X_Label='model evals', Y_Label='total effects')
plt.show()
