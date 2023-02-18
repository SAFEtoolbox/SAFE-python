"""
This script provides an application example of the Fourier Amplitude
Sensitivity Test (FAST). FAST uses the Fourier decomposition of the model output
to approximates the variance-based first-order sensitivity indices.
See help of 'FAST.FAST_indices.m' for more details and references.

In this workflow, FAST is applied to the Sobol g-function
with varying number of inputs and different parameterizations for the
function parameters not subject to SA
[see help of 'sobol_g.sobol_g_function.m'].

FAST estimates are compared to those obtained by the 'conventional'
resampling approach used in Variance-Based SA
[see help of 'VBSA.vbsa_indices.m'].

This script can be used to reproduce Figure 2 in:
Saltelli and Bolado (1998), An alternative way to compute Fourier
amplitude sensitivity test (FAST), Computational Statistics & Data
Analysis, 26, 445-460.

This script was prepared by Fanny Sarrazin, 2019
fanny.sarrazin@ufz.de
"""

#%% Step 1: (import python modules)

from __future__ import division, absolute_import, print_function

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

# Import SAFE modules
from safepython import FAST # module to perform FAST
import safepython.VBSA as VB # module to perform VBSA
from safepython.model_execution import model_execution # module to execute the model
from safepython.sampling import AAT_sampling # functions to perform the input sampling

from safepython.sobol_g import sobol_g_function

#%% Step 2 (setup the model)

# Define output:
fun_test = sobol_g_function

# Parameter distributions:
distr_fun = st.uniform # uniform distribution
# Parameter ranges:
distr_par = [0, 1]

MM = np.arange(5, 12) # options for the number of inputs subject to SA
aa = np.array([0, 1, 9, 99]) # options for the (fixed) parameters

#%% Step 3 (calculate sensitivity indices)

ms = 16 # marker size for plots
pltfont = {'fontname': 'DejaVu Sans', 'fontsize': 15} # font for plots

plt.figure()

for param in range(4):
    SM = np.nan * np.zeros((len(MM), 3))
    Nfast = np.zeros((len(MM), ), dtype=int)
    Nvbsa = np.zeros((len(MM), ), dtype=int)

    for m in range(len(MM)):

        M = MM[m] # number of inputs subject to SA
        a = np.ones((M, ))*aa[param] # fixed parameters

        # Analytical indices:
        _, V_ex, Si_ex = sobol_g_function(np.random.random((M, )), a)

        # FAST:
        X, s = FAST.FAST_sampling(distr_fun, distr_par, M)
        # Run the model and compute selected model output at sampled parameter
        # sets:
        Y = model_execution(fun_test, X, a) # shape (Ns,1)
        Si_fast, _, _, _, _ = FAST.FAST_indices(Y, M)

        # VBSA:
        samp_strat = 'lhs'
        Nfast[m] = len(Y)
        # Option 1: set the base sample size for VBSA in such a way that
        # the total number of model evaluations is the same as FAST:
        Nvbsa[m] = int(np.ceil(Nfast[m]/(M+2)))
        # Option 2: set the base sample size to 4096 independently by the
        # sample size used for FAST (as done in Saltelli and Bolado, 1998)
        # [THIS OPTION MAY BE TIME-CONSUMING!]
        # Nvbsa[m] = 4096
        X = AAT_sampling(samp_strat, M, distr_fun, distr_par, 2*Nvbsa[m])
        XA, XB, XC = VB.vbsa_resampling(X)
        YA = model_execution(fun_test, XA, a) # shape (N,1)
        YB = model_execution(fun_test, XB ,a) # shape (N,1)
        YC = model_execution(fun_test, XC, a) # shape (N*M,1)
        Si_vbsa = VB.vbsa_indices(YA, YB, YC, M)
        SM[m, :] = np.array([np.mean(Si_ex), np.mean(Si_vbsa), np.mean(Si_fast)])

    # Plot the results:
    plt.subplot(2, 2, param+1)
    plt.plot(MM, SM[:, 0], 'sk', markersize=ms,
             markerfacecolor=[126/256, 126/256, 126/256])
    plt.plot(MM, SM[:, 1], '^k', markersize=ms, markerfacecolor="None",
             markeredgewidth=2)
    plt.plot(MM, SM[:, 2], 'ok', markersize=ms, markerfacecolor="None",
             markeredgewidth=2)
    # customize picture:
    plt.xticks(MM, **pltfont)
    plt.xlim((MM[0]-1, MM[-1]+1))
    plt.ylim((min(np.min(SM)*1.2, 0), max(1.1, np.max(SM)*1.2)))

    plt.xlabel('Number of inputs M', **pltfont)
    plt.ylabel('1st-order sensitivity', **pltfont)
    plt.legend(['Analytical', 'VBSA', 'FAST'])
    plt.title('a(i)= %d' % (aa[param]), **pltfont)

plt.show()

# Plot number of model evaluations against number of inputs:
plt.figure()
plt.plot(MM, Nvbsa*(MM+2), '^k', markersize=ms, markerfacecolor="None",
         markeredgewidth=2)
plt.plot(MM, Nfast, 'ok', markersize=ms, markerfacecolor="None",
         markeredgewidth=2)

plt.xticks(MM, **pltfont)
plt.xlabel('Number of inputs M', **pltfont)
plt.ylabel('Number of model evaluations N', **pltfont)
plt.legend(['VBSA', 'FAST'])
plt.grid(axis='both')
plt.show()
