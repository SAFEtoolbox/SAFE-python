"""
This script provides an application example of the Fourier Amplitude
Sensitivity Test (FAST). FAST uses the Fourier decomposition of the model output
to approximate the variance-based first-order sensitivity indices.
See help of 'FAST.FAST_indices.m' for more details and references.

In this workflow, FAST is applied to the rainfall-runoff Hymod model
(see help of 'HyMod.hymod_sim.m' for more details)
applied to the Leaf catchment in Mississipi, USA
(see header of file LeafCatch.txt for more details).
The inputs subject to SA are the 5 model parameters, and the scalar
output for SA is a performance metric.

FAST estimates are compared to those obtained by the 'conventional'
resampling approach used in Variance-Based SA
[see help of 'VBSA.vbsa_indices.m'].

This script prepared by  Fanny Sarrazin, 2019
fanny.sarrazin@ufz.de
"""

#%% Step 1: (import python modules)

from __future__ import division, absolute_import, print_function

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

from safepython import FAST # module to perform FAST
import safepython.VBSA as VB # module to perform VBSA
import safepython.plot_functions as pf # module to visualize the results
from safepython.model_execution import model_execution # module to execute the model
from safepython.sampling import AAT_sampling # functions to perform the input sampling

from safepython import HyMod

#%% Step 2: (setup the Hymod model)

# Specify the directory where the data are stored (CHANGE TO YOUR OWN DIRECTORY)
mydir = r'Y:\Home\sarrazin\SAFE\SAFE_Python\SAFE-python-0.1.1\examples\data'
# Load data:
data = np.genfromtxt(mydir +'\LeafCatch.txt', comments='%')
rain = data[0:365, 0] # 1-year simulation
evap = data[0:365, 1]
flow = data[0:365, 2]
warmup = 30 # Model warmup period (days)

# Number of uncertain parameters subject to SA:
M = 5

# Parameter ranges (from Kollat et al.(2012)):
xmin = [0, 0, 0, 0, 0.1]
xmax = [400, 2, 1, 0.1, 1]

# Parameter distributions
distr_fun = st.uniform # uniform distribution
# The shape parameters for the uniform distribution are the lower limit and the
# difference between lower and upper limits:
distr_par = [np.nan] * M
for i in range(M):
    distr_par[i] = [xmin[i], xmax[i] - xmin[i]]

# Name of parameters (will be used to customize plots):
X_Labels = ['Sm', 'beta', 'alfa', 'Rs', 'Rf']

# Define output:
fun_test = HyMod.hymod_nse

#%% Step 3 (Approximate first-order sensitivity indices by FAST)

# FAST sampling:
X, s = FAST.FAST_sampling(distr_fun, distr_par, M)

# Run the model and compute model output at sampled parameter sets:
Y = model_execution(fun_test, X, rain, evap, flow, warmup)

# Estimate indices:
Si_fast, V, A, B, Vi = FAST.FAST_indices(Y, M)

#%% Step 4 (Convergence analysis)
# The 'FAST.FAST_sampling' function used above automatically set the sample size
# to the minimum possible given the number M of inputs (see help of
# FAST.FAST_sampling to learn about this). In our case this is:
N_fast = len(Y)

# We can now assess whether FAST estimates would change if using a larger
# number of samples:
NNfast = np.linspace(N_fast, N_fast+5000, 6).astype(int)
Si_fast_conv = np.nan * np.zeros((len(NNfast), M))
Si_fast_conv[0, :] = Si_fast

for n in range(1, len(NNfast)):
    Xn, sn = FAST.FAST_sampling(distr_fun, distr_par, M, NNfast[n])
    Yn = model_execution(fun_test, Xn, rain, evap, flow, warmup)
    Si_fast_conv[n, :], _, _, _, _ = FAST.FAST_indices(Yn, M)

# Plot results:
plt.figure()
pf.plot_convergence(Si_fast_conv, NNfast, X_Label='no of model evaluations',
                    Y_Label='1st-order sensitivity', labelinput=X_Labels)
plt.show()

#%% Step 5 (Comparison with VBSA)

# Here we compare FAST estimates with those obtained by the 'conventional'
# resampling approach used in Variance-Based SA
# (see help of 'VBSA.vbsa_indices.m').

# Set the base sample size for VBSA in such a way that the total number of
# model evaluations be the same as FAST:
Nvbsa = int(np.ceil(np.max(NNfast)/(M+2)))

# VBSA sampling:
samp_strat = 'lhs'
X = AAT_sampling(samp_strat, M, distr_fun, distr_par, 2*Nvbsa)
XA, XB, XC = VB.vbsa_resampling(X)

# Run the model and compute selected model output at sampled parameter
# sets:
YA = model_execution(fun_test, XA, rain, evap, flow, warmup) # shape (N, )
YB = model_execution(fun_test, XB, rain, evap, flow, warmup) # shape (N, )
YC = model_execution(fun_test, XC, rain, evap, flow, warmup) # shape (N*M, )

# Use output samples to estimate the indices at different sub-sample sizes:
NNvbsa = np.array([int(i) for i in np.floor(NNfast/(M+2))])
Si_vbsa_conv, _ = VB.vbsa_convergence(YA, YB, YC, M, NNvbsa)

# Compare
xm = min(NNfast[0], NNvbsa[0]*(M+2))
xM = max(NNfast[-1], NNvbsa[-1]*(M+2))
plt.figure()
plt.subplot(211)
pf.plot_convergence(Si_fast_conv, NNfast, X_Label='no of model evaluations',
                    Y_Label='1st-order sensitivity (FAST)', labelinput=X_Labels)
plt.xlim(xm, xM)
plt.subplot(212)
pf.plot_convergence(Si_vbsa_conv, NNvbsa*(M+2), X_Label='no of model evaluations',
                    Y_Label='1st-order sensitivity (VBSA)', labelinput=X_Labels)
plt.xlim(xm, xM)
plt.show()