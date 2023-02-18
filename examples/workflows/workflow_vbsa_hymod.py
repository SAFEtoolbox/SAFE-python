"""
This script provides an application example of Variance Based Sensitivity
Analysis (VBSA)

METHODS

We use two well established variance-based sensitivity indices:
- the first-order sensitivity index (or 'main effects')
- the total-order sensitivity index (or 'total effects')
Interaction effect is also assessed as the difference between total and main
effect.
(see help of 'VBSA.vbsa_indices' for more details and references)

MODEL AND STUDY AREA

The model under study is the rainfall-runoff model Hymod
(see help of function hymod_sim.m for more details)
applied to the Leaf catchment in Mississipi, USA
(see header of file LeafCatch.txt for more details).
The inputs subject to SA are the 5 model parameters, and the scalar
output for SA is (one or multiple) performance metric.


INDEX

Steps:
1. Set current working directory and import python modules
2. Load data, set-up the Hymod model and define input ranges
3. Compute first-order (main effects) and total-order (total effects)
   variance-based indices.
4. Example of how to repeat computions after adding up new input/output samples.
5. Example of how to compute indices when dealing with multiple outputs.
6. Example of how to identify influential and non-influential inputs using a
   'dummy' input (see help of 'VBSA.vbsa_indices' for more details and
   references on the use of the dummy input).

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
from safepython.sampling import AAT_sampling, AAT_sampling_extend  # module to
# perform the input sampling
from safepython.util import aggregate_boot # function to aggregate results across bootstrap
# resamples

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

#%% Step 3 (Compute first-order and total-order variance-based indices)

# Sample parameter space using the resampling strategy proposed by
# (Saltelli, 2008; for reference and more details, see help of functions
# VBSA.vbsa_resampling and VBSA.vbsa_indices)
samp_strat = 'lhs' # Latin Hypercube
N = 3000 #  Number of samples
X = AAT_sampling(samp_strat, M, distr_fun, distr_par, 2*N)
XA, XB, XC = VB.vbsa_resampling(X)

# Run the model and compute selected model output at sampled parameter
# sets:
YA = model_execution(fun_test, XA, rain, evap, flow, warmup) # shape (N, )
YB = model_execution(fun_test, XB, rain, evap, flow, warmup) # shape (N, )
YC = model_execution(fun_test, XC, rain, evap, flow, warmup) # shape (N*M, )

# Compute main (first-order) and total effects:
Si, STi = VB.vbsa_indices(YA, YB, YC, M)

# Plot results:
plt.figure()
plt.subplot(131)
pf.boxplot1(Si, X_Labels=X_Labels, Y_Label='main effects')
plt.subplot(132)
pf.boxplot1(STi, X_Labels=X_Labels, Y_Label='total effects')
plt.subplot(133)
pf.boxplot1(STi-Si, X_Labels=X_Labels, Y_Label='interactions')
plt.show()

# Plot main and total effects in one plot:
plt.figure()
pf.boxplot2(np.stack((Si, STi)), X_Labels=X_Labels,
            legend=['main effects', 'total effects'])
plt.show()

# Check the model output distribution (if multi-modal or highly skewed, the
# variance-based approach may not be adequate):
Y = np.concatenate((YA, YC))
plt.figure()
pf.plot_cdf(Y, Y_Label='NSE')
plt.show()
plt.figure()
fi, yi = pf.plot_pdf(Y, Y_Label='NSE')
plt.show()

# Use bootstrapping to derive confidence bounds:
Nboot = 1000
# Compute sensitivity indices for Nboot bootstrap resamples:
Si, STi = VB.vbsa_indices(YA, YB, YC, M, Nboot=Nboot)
# Si and STi have shape (Nboot, M)
# Compute mean and confidence intervals of the sensitivity indices across the
# bootstrap resamples:
Si_m, Si_lb, Si_ub = aggregate_boot(Si) # shape (M,)
STi_m, STi_lb, STi_ub = aggregate_boot(STi) # shape (M,)
Inti_m, Inti_lb, Inti_ub = aggregate_boot(STi-Si) # shape (M,)

# Plot bootstrapping results:
plt.figure() # plot main, total and interaction effects separately
plt.subplot(131)
pf.boxplot1(Si_m, S_lb=Si_lb, S_ub=Si_ub, X_Labels=X_Labels, Y_Label='main effects')
plt.subplot(132)
pf.boxplot1(STi_m, S_lb=STi_lb, S_ub=STi_ub, X_Labels=X_Labels, Y_Label='total effects')
plt.subplot(133)
pf.boxplot1(Inti_m, S_lb=Inti_lb, S_ub=Inti_ub, X_Labels=X_Labels, Y_Label='interactions')
plt.show()

# Plot main and total effects in one plot:
plt.figure()
pf.boxplot2(np.stack((Si_m, STi_m)), S_lb=np.stack((Si_lb, STi_lb)),
            S_ub=np.stack((Si_ub, STi_ub)), X_Labels=X_Labels,
            legend=['main effects', 'total effects'])
plt.show()

# Analyze convergence of sensitivity indices:
NN = np.linspace(N/10, N, 10).astype(int)
Sic, STic = VB.vbsa_convergence(YA, YB, YC, M, NN)
# Plot convergence results:
plt.figure()
plt.subplot(121)
pf.plot_convergence(Sic, NN*(M+2), X_Label='no of model evaluations',
                    Y_Label='main effects', labelinput=X_Labels)
plt.subplot(122)
pf.plot_convergence(STic, NN*(M+2), X_Label='no of model evaluations',
                    Y_Label='total effects', labelinput=X_Labels)
plt.show()

# Analyze convergence using bootstrapping to derive confidence intervals:
Sic, STic = VB.vbsa_convergence(YA, YB, YC, M, NN, Nboot)
# Compute mean and confidence intervals of the sensitivity indices across the
# bootstrap resamples:
Sic_m, Sic_lb, Sic_ub = aggregate_boot(Sic) # shape (R,M)
STic_m, STic_lb, STic_ub = aggregate_boot(STic) # shape (R,M)

# Plot convergence results:
plt.figure()
plt.subplot(121)
pf.plot_convergence(Sic_m, NN*(M+2), Sic_lb, Sic_ub,
                    X_Label='no of model evaluations',
                    Y_Label='main effects', labelinput=X_Labels)
plt.subplot(122)
pf.plot_convergence(STic_m, NN*(M+2), STic_lb, STic_ub,
                    X_Label='no of model evaluations',
                    Y_Label='total effects', labelinput=X_Labels)
plt.show()

#%% Step 4: Adding up new samples
Nnew = 4000 # increase of base sample size
# (that means: Nnew*(M+2) new samples that will need to be evaluated)
Xext = AAT_sampling_extend(X, distr_fun, distr_par, 2*(N+Nnew)) # extended sample
# (it includes the already evaluated samples 'X' and the new ones)
Xnew = Xext[2*N:2*(N+Nnew), :] # extract the new input samples that need to be
# evaluated

# Resampling strategy:
[XAnew, XBnew, XCnew] = VB.vbsa_resampling(Xnew)
# Evaluate model against new samples:
YAnew = model_execution(fun_test, XAnew, rain, evap, flow, warmup)
# should have shape (Nnew, 1)
YBnew = model_execution(fun_test, XBnew, rain, evap, flow, warmup)
# should have shape (Nnew, 1)
YCnew = model_execution(fun_test, XCnew, rain, evap, flow, warmup)
# should have shape (Nnew*M, 1)

# Put new and old results toghether:
YA2 = np.concatenate((YA, YAnew))  # should have shape (N+Nnew, 1)
YB2 = np.concatenate((YB, YBnew))  # should have shape (N+Nnew,1)
YC2 = np.concatenate((np.reshape(YC, (M, N)), np.reshape(YCnew, (M, Nnew))),
                     axis=1)# should have size (M, N+Nnew)
YC2 = YC2.flatten() # should have size ((N+Nnew)*M, )

# Recompute indices:
Nboot = 1000
Si2, STi2 = VB.vbsa_indices(YA2, YB2, YC2, M, Nboot)
# Compute mean and confidence intervals of the sensitivity indices across the
# bootstrap resamples:
Si2_m, Si2_lb, Si2_ub = aggregate_boot(Si2) # shape (M,)
STi2_m, STi2_lb, STi2_ub = aggregate_boot(STi2) # shape (M,)

# Plot sensitivity indices calculated with the initial sample and the extended
# sample
plt.figure()
plt.subplot(121)
pf.boxplot2(np.stack((Si_m, STi_m)), S_lb=np.stack((Si_lb, STi_lb)),
            S_ub=np.stack((Si_ub, STi_ub)), X_Labels=X_Labels)
plt.title('%d' % (N*(M+2)) + ' model eval.')
plt.subplot(122)
pf.boxplot2(np.stack((Si2_m, STi2_m)), S_lb=np.stack((Si2_lb, STi2_lb)),
            S_ub=np.stack((Si2_ub, STi2_ub)), X_Labels=X_Labels,
            legend=['main effects', 'total effects'])
plt.title('%d' % ((N+Nnew)*(M+2)) + ' model eval.')
plt.show()

#%% Step 5 (Case of multiple outputs)
# (In this example: RMSE and AME)

# Run the model and compute selected model output at sampled parameter
# sets:
fun_test = HyMod.hymod_MulObj
YA = model_execution(fun_test, XA, rain, evap, flow, warmup) # shape (N, )
YB = model_execution(fun_test, XB, rain, evap, flow, warmup) # shape (N, )
YC = model_execution(fun_test, XC, rain, evap, flow, warmup) # shape (N*M, )

# Select the j-th model output and compute sensitivity indices:
Nboot = 1000
j = 0 # RMSE
Si1, STi1 = VB.vbsa_indices(YA[:, j], YB[:, j], YC[:, j], M, Nboot)
j = 1 # BIAS
Si2, STi2 = VB.vbsa_indices(YA[:, j], YB[:, j], YC[:, j], M, Nboot)

# Compute mean and confidence intervals of the sensitivity indices across the
# bootstrap resamples:
Si1_m, Si1_lb, Si1_ub = aggregate_boot(Si1) # shape (M, )
STi1_m, STi1_lb, STi1_ub = aggregate_boot(STi1) # shape (M, )
Si2_m, Si2_lb, Si2_ub = aggregate_boot(Si2) # shape (M, )
STi2_m, STi2_lb, STi2_ub = aggregate_boot(STi2) # shape (M, )

# Compare boxplots:
plt.figure()
plt.subplot(121)
pf.boxplot2(np.stack((Si1_m, STi1_m)), S_lb=np.stack((Si1_lb, STi1_lb)),
            S_ub=np.stack((Si1_ub, STi1_ub)), X_Labels=X_Labels)
plt.title('RMSE')
plt.subplot(122)
pf.boxplot2(np.stack((Si2_m, STi2_m)), S_lb=np.stack((Si2_lb, STi2_lb)),
            S_ub=np.stack((Si2_ub, STi2_ub)), X_Labels=X_Labels,
            legend=['main effects', 'total effects'])
plt.title('BIAS')
plt.show()

# Use stacked bar to put all outputs on one plot:
plt.figure()
plt.subplot(121)
pf.stackedbar(np.stack((Si1_m, Si2_m)), Y_Label='main effects',
              horiz_tick_label=['RMSE', 'BIAS'])
plt.subplot(122)
pf.stackedbar(np.stack((STi1_m, STi2_m)), labelinput=X_Labels,
              Y_Label='total effects', horiz_tick_label=['RMSE', 'BIAS'])
plt.show()

# We note that the bootstrap confidence intervals for the sensitivity indices 
# obtained for RMSE are very large and there tend to be negative values, which
# means that the sample size used is not large enough. This convergence issue
# may be explained by the fact that the distribution of RMSE is not normal.
# Plot distribution for RMSE
plt.figure()
pf.plot_cdf(YA[:, 0], Y_Label='RMSE')
plt.show()
plt.figure()
fi, yi = pf.plot_pdf(YA[:, 0], Y_Label='RMSE')
plt.show()
# Plot distribution for BIAS
plt.figure()
pf.plot_cdf(YA[:, 1], Y_Label='BIAS')
plt.show()
plt.figure()
fi, yi = pf.plot_pdf(YA[:, 1], Y_Label='BIAS')
plt.show()

#%% Step 6 (Identification of influential and non-influential inputs adding an
# articial 'dummy' input to the list of the model inputs. The sensitivity
# indices for the dummy parameter estimate the approximation error of the
# sensitivity indices. For reference and more details, see help of the function
# VBSA.vbsa_indices)

# Name of parameters (will be used to customize plots) including the dummy input:
X_Labels_dummy = ['Sm', 'beta', 'alfa', 'Rs', 'Rf', 'dummy']

# Compute main (first-order) and total effects using bootstrapping for the model
# inputs and the dummy input:
#j = 0 # RMSE
j = 1 # BIAS
Nboot = 1000
Si, STi, Sdummy, STdummy = VB.vbsa_indices(YA[:, j], YB[:, j], YC[:, j],
                                           M, Nboot=Nboot, dummy=True)
# STdummy is the sensitivity index (total effect) for the dummy input
# Compute mean and confidence intervals of the sensitivity indices across the
# bootstrap resamples:
Si_m, Si_lb, Si_ub = aggregate_boot(np.column_stack((Si, Sdummy))) # shape (M+1,)
STi_m, STi_lb, STi_ub = aggregate_boot(np.column_stack((STi, STdummy))) # shape (M+1,)

# Plot bootstrapping results:
plt.figure() # plot main and total separately
plt.subplot(121)
pf.boxplot1(Si_m, S_lb=Si_lb, S_ub=Si_ub, X_Labels=X_Labels_dummy, Y_Label='main effects')
plt.subplot(122)
pf.boxplot1(STi_m, S_lb=STi_lb, S_ub=STi_ub, X_Labels=X_Labels_dummy, Y_Label='total effects')
plt.show()

#  Analyze convergence:
NN = np.linspace(N/10, N, 10).astype(int)
Sic, STic, Sdummyc, STdummyc = VB.vbsa_convergence(YA[:, j], YB[:, j], YC[:, j],
                                                   M, NN, Nboot, dummy=True)
# Compute mean and confidence intervals of the sensitivity indices across the
# bootstrap resamples:
Sic_all = []
STic_all = []
for i in range(len(NN)):
    Sic_all = Sic_all + [np.column_stack((Sic[i], Sdummyc[i]))]
    STic_all = STic_all + [np.column_stack((STic[i], STdummyc[i]))]
Sic_m, Sic_lb, Sic_ub = aggregate_boot(Sic_all) # shape (R,M+1)
STic_m, STic_lb, STic_ub = aggregate_boot(STic_all) # shape (R,M+1)

# Plot convergence results:
plt.figure()
plt.subplot(121)
pf.plot_convergence(Sic_m, NN*(M+2), Sic_lb, Sic_ub, X_Label='no of model evaluations',
                    Y_Label='main effects', labelinput=X_Labels_dummy)
plt.subplot(122)
pf.plot_convergence(STic_m, NN*(M+2), STic_lb, STic_ub, X_Label='no of model evaluations',
                    Y_Label='total effects', labelinput=X_Labels_dummy)
plt.show()
