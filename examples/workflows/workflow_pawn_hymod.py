"""
This script provides an application example of the PAWN sensitivity analysis
approach (Pianosi and Wagener, 2015,2018)

MODEL AND STUDY AREA

The model under study is the rainfall-runoff model Hymod
(see help of function hymod_sim.m for more details)
applied to the Leaf catchment in Mississipi, USA
(see header of file LeafCatch.txt for more details).
The inputs subject to SA are the 5 model parameters, and the scalar
output for SA is a statistic of the simulated time series
(e.g. the maximum flow over the simulation horizon)

INDEX

Steps:
1. Set current working directory and import python modules
2. Load data and set-up the Hymod model
3. Sample inputs space
4. Run the model against input samples
5. Apply PAWN
6. Example of how to repeat computions after adding up new input/output samples
7. Example of how to identify influential and non-influential inputs using a
   'dummy' input (see help of 'PAWN.pawn_indices' for more details and
   references on the use of the dummy input).
8. Example of advanced usage of PAWN for Regional-Response Global Sensitivity
   Analysis

REFERENCES

Pianosi, F. and Wagener, T. (2018), Distribution-based sensitivity
analysis from a generic input-output sample, Env. Mod. & Soft., 108, 197-207.

Pianosi, F. and Wagener, T. (2015), A simple and efficient method
for global sensitivity analysis based on cumulative distribution
functions, Env. Mod. & Soft., 67, 1-11.

This script prepared by  Fanny Sarrazin, 2019
fanny.sarrazin@ufz.de
"""

#%% Step 1: (import python modules)

from __future__ import division, absolute_import, print_function

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

from safepython import PAWN
import safepython.plot_functions as pf # module to visualize the results
from safepython.model_execution import model_execution # module to execute the model
from safepython.sampling import AAT_sampling, AAT_sampling_extend # module to perform the input sampling
from safepython.util import aggregate_boot  # function to aggregate the bootstrap results

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

# Parameter distributions:
distr_fun = st.uniform # uniform distribution
# The shape parameters of the uniform distribution are the lower limit and the
# difference between lower and upper limits:
distr_par = [np.nan] * M
for i in range(M):
    distr_par[i] = [xmin[i], xmax[i] - xmin[i]]

# Name of parameters (will be used to customize plots):
X_Labels = ['Sm', 'beta', 'alfa', 'Rs', 'Rf']

# Define output:
fun_test = HyMod.hymod_max

#%% Step 3 (sample inputs space)
samp_strat = 'lhs' # Latin Hypercube
N = 2000  #  Number of samples
X = AAT_sampling(samp_strat, M, distr_fun, distr_par, N)

#%% Step 4 (run the model)
Y = model_execution(fun_test, X, rain, evap, warmup)

#%% Step 5 (Apply PAWN)

n = 10 # number of conditioning intervals

# Compute and plot conditional and unconditional CDFs:
YF, FU, FC, xc = PAWN.pawn_plot_cdf(X, Y, n)
plt.show()
# Add colorbar:
YF, FU, FC, xc = PAWN.pawn_plot_cdf(X, Y, n, cbar=True, n_col=3)
plt.show()
# Add label to the colorbar:
YF, FU, FC, xc = PAWN.pawn_plot_cdf(X, Y, n, cbar=True, n_col=3, labelinput=X_Labels)
# You can also adjust the spacing between the subplots to create a nice figure
# using the function subplots_adjust:
#plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
#                    hspace=0.5, wspace=0.4)
plt.show()

# Compute and plot KS statistics for each conditioning interval:
KS = PAWN.pawn_plot_ks(YF, FU, FC, xc)
plt.show()
# Customize plot:
KS = PAWN.pawn_plot_ks(YF, FU, FC, xc, n_col=3, X_Labels=X_Labels)
# You can also adjust the spacing between the subplots to create a nice figure
# using the function subplots_adjust:
#plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
#                    hspace=0.5, wspace=0.4)
plt.show()

# Compute PAWN sensitivity indices:
KS_median, KS_mean, KS_max = PAWN.pawn_indices(X, Y, n)
# Plot results:
plt.figure()
plt.subplot(131)
pf.boxplot1(KS_median, X_Labels=X_Labels, Y_Label='KS (median)')
plt.subplot(132)
pf.boxplot1(KS_mean, X_Labels=X_Labels, Y_Label='KS (mean)')
plt.subplot(133)
pf.boxplot1(KS_max, X_Labels=X_Labels, Y_Label='Ks (max)')
plt.show()

# Use bootstrapping to derive confidence bounds:
Nboot = 1000
# Compute sensitivity indices for Nboot bootstrap resamples
# (Warning: the following line may take some time to run, as the computation of
# CDFs is costly):
KS_median, KS_mean, KS_max = PAWN.pawn_indices(X, Y, n, Nboot=Nboot)
# KS_median and KS_mean and KS_max have shape (Nboot, M)
# Compute mean and confidence intervals of the sensitivity indices across the
# bootstrap resamples:
KS_median_m, KS_median_lb, KS_median_ub = aggregate_boot(KS_median) # shape (M,)
KS_mean_m, KS_mean_lb, KS_mean_ub = aggregate_boot(KS_mean) # shape (M,)
KS_max_m, KS_max_lb, KS_max_ub = aggregate_boot(KS_max) # shape (M,)

# Plot bootstrapping results:
plt.figure()
plt.subplot(131)
pf.boxplot1(KS_median_m, S_lb=KS_median_lb, S_ub=KS_median_ub,
            X_Labels=X_Labels, Y_Label='KS (median)')
plt.subplot(132)
pf.boxplot1(KS_mean_m, S_lb=KS_mean_lb, S_ub=KS_mean_ub,
            X_Labels=X_Labels, Y_Label='KS (mean)')
plt.subplot(133)
pf.boxplot1(KS_max_m, S_lb=KS_max_lb, S_ub=KS_max_ub,
            X_Labels=X_Labels, Y_Label='Ks (max)')
plt.show()

# Analyze convergence of sensitivity indices:
NN = np.linspace(N/5, N, 5).astype(int)
# Warning: the following line may take some time to run, as the computation of
# CDFs is costly:
KS_median_c, KS_mean_c, KS_max_c = PAWN.pawn_convergence(X, Y, n, NN) # shape (R,M)

# Plot convergence results:
plt.figure()
plt.subplot(311)
pf.plot_convergence(KS_median_c, NN, X_Label='no of model evaluations',
                    Y_Label='KS (median)', labelinput=X_Labels)
plt.subplot(312)
pf.plot_convergence(KS_median_c, NN, X_Label='no of model evaluations',
                    Y_Label='KS (mean)', labelinput=X_Labels)
plt.subplot(313)
pf.plot_convergence(KS_max_c, NN, X_Label='no of model evaluations',
                    Y_Label='KS (max)', labelinput=X_Labels)
plt.show()

# Analyze convergence using bootstrapping to derive confidence intervals
#( Warning: the following line may take some time to run, as the computation of
# CDFs is costly):
KS_median_c, KS_mean_c, KS_max_c = PAWN.pawn_convergence(X, Y, n, NN, Nboot=1000)
# Compute mean and confidence intervals of the sensitivity indices across the
# bootstrap resamples:
KS_median_c_m, KS_median_c_lb, KS_median_c_ub = aggregate_boot(KS_median_c) # shape (R,M)
KS_mean_c_m, KS_mean_c_lb, KS_mean_c_ub = aggregate_boot(KS_mean_c) # shape (R,M)
KS_max_c_m, KS_max_c_lb, KS_max_c_ub = aggregate_boot(KS_max_c) # shape (R,M)

# Plot convergence results:
plt.figure()
plt.subplot(311)
pf.plot_convergence(KS_median_c_m, NN, KS_median_c_lb, KS_median_c_ub,
                    X_Label='no of model evaluations',
                    Y_Label='KS (median)', labelinput=X_Labels)
plt.subplot(312)
pf.plot_convergence(KS_mean_c_m, NN, KS_mean_c_lb, KS_mean_c_ub,
                    X_Label='no of model evaluations',
                    Y_Label='KS (mean)', labelinput=X_Labels)
plt.subplot(313)
pf.plot_convergence(KS_max_c_m, NN, KS_max_c_lb, KS_max_c_ub,
                    X_Label='no of model evaluations',
                    Y_Label='KS (max)', labelinput=X_Labels)
plt.show()

#%% Step 6 (Adding up new samples)
Next = 3000 # new sample size
Xext = AAT_sampling_extend(X, distr_fun, distr_par, Next)
# Run the model over the new samples:
Ynew = model_execution(fun_test, Xext[N:N+Next, :], rain, evap, warmup)
Yext = np.concatenate((Y, Ynew))  # should have shape (Next, 1)

# Recompute indices:
Nboot = 1000
KS_median2, KS_mean2, KS_max2 = PAWN.pawn_indices(Xext, Yext, n, Nboot=Nboot)
# Compute mean and confidence intervals of the sensitivity indices across the
# bootstrap resamples:
KS_median2_m, KS_median2_lb, KS_median2_ub = aggregate_boot(KS_median2) # shape (M,)
KS_mean2_m, KS_mean2_lb, KS_mean2_ub = aggregate_boot(KS_mean2) # shape (M,)
KS_max2_m, KS_max2_lb, KS_max2_ub = aggregate_boot(KS_max2) # shape (M,)

# Plot sensitivity indices calculated with the initial sample and the extended
# sample (for instance for KS_max):
plt.figure()
plt.subplot(121)
pf.boxplot1(KS_max_m, S_lb=KS_max_lb, S_ub=KS_max_ub,
            X_Labels=X_Labels, Y_Label='KS (median)')
plt.subplot(122)
pf.boxplot1(KS_max2_m, S_lb=KS_max2_lb, S_ub=KS_max2_ub,
            X_Labels=X_Labels, Y_Label='KS (median)')
plt.show()

#%% Step 7 (Identification of influential and non-influential inputs adding an
# articial 'dummy' input to the list of the model inputs. The sensitivity
# indices for the dummy parameter estimate the approximation error of the
# sensitivity indices. For reference and more details, see help of the function
# PAWN.pawn_indices

# Name of parameters (will be used to customize plots) including the dummy input:
X_Labels_dummy = ['Sm', 'beta', 'alfa', 'Rs', 'Rf', 'dummy']

# Sensitivity indices using bootstrapping for the model inputs and the dummy
# input:
  # Use bootstrapping to derive confidence bounds:
Nboot = 1000
# Compute sensitivity indices for Nboot bootstrap resamples. We analyse KS_max
# only (and not KS_median and KS_mean) for screening purposes.
# (Warning: the following line may take some time to run, as the computation of
# CDFs is costly):
_, _, KS_max, KS_dummy = PAWN.pawn_indices(X, Y, n, Nboot=Nboot, dummy=True)
# KS_max has shape (Nboot, M), KS_dummy has shape (Nboot, )

# Compute mean and confidence intervals of the sensitivity indices across the
# bootstrap resamples:
KS_m, KS_lb, KS_ub = aggregate_boot(np.column_stack((KS_max, KS_dummy)))

# Plot bootstrapping results:
plt.figure() # plot main and total separately
pf.boxplot1(KS_m, S_lb=KS_lb, S_ub=KS_ub, X_Labels=X_Labels_dummy, Y_Label='KS')
plt.show()

# Analyze convergence using bootstrapping to derive confidence intervals
#( Warning: the following line may take some time to run, as the computation of
# CDFs is costly):
NN = np.linspace(N/5, N, 5).astype(int)
_, _, KS_max_c, KS_dummy_c = PAWN.pawn_convergence(X, Y, n, NN, Nboot=1000, dummy=True)

# Compute mean and confidence intervals of the sensitivity indices across the
# bootstrap resamples:
KSc_all = []
for i in range(len(NN)):
    KSc_all = KSc_all + [np.column_stack((KS_max_c[i], KS_dummy_c[i]))]
KSc_m, KSc_lb, KSc_ub = aggregate_boot(KSc_all) # shape (R,M+1)

# Plot convergence results:
plt.figure()
pf.plot_convergence(KSc_m, NN, KSc_lb, KSc_ub,
                    X_Label='no of model evaluations',
                    Y_Label='KS', labelinput=X_Labels_dummy)
plt.show()

#%% Step 8 (ADVANCED USAGE for Regional-Response Global Sensitivity Analysis):
# (Apply PAWN to a sub-region of the output range)

# Compute the PAWN index over a sub-range of the output distribution, for
# instance only output values above a given threshold:
thres = [30]
Nboot = 1000
# print to screen the size of the sample that will be used to calculate the
# PAWN indices:
print(np.sum(Y>thres))
KS_median_cond, KS_mean_cond, KS_max_cond = \
PAWN.pawn_indices(X, Y, n, Nboot=Nboot, output_condition=PAWN.above, par=thres)
# other implemented output conditions: PAWN.allrange (default) and PAWN.below

# Compute mean and confidence intervals of the sensitivity indices across the
# bootstrap resamples:
KS_median_cond_m, KS_median_cond_lb, KS_median_cond_ub = aggregate_boot(KS_median_cond) # shape (M,)
KS_mean_cond_m, KS_mean_cond_lb, KS_mean_cond_ub = aggregate_boot(KS_mean_cond) # shape (M,)
KS_max_cond_m, KS_max_cond_lb, KS_max_cond_ub = aggregate_boot(KS_max_cond) # shape (M,)

# Plot bootstrapping results:
plt.figure()
plt.subplot(131)
pf.boxplot1(KS_median_cond_m, S_lb=KS_median_cond_lb, S_ub=KS_median_cond_ub,
            X_Labels=X_Labels, Y_Label='KS (median)')
plt.subplot(132)
pf.boxplot1(KS_mean_cond_m, S_lb=KS_mean_cond_lb, S_ub=KS_mean_cond_ub,
            X_Labels=X_Labels, Y_Label='KS (mean)')
plt.subplot(133)
pf.boxplot1(KS_max_cond_m, S_lb=KS_max_cond_lb, S_ub=KS_max_cond_ub,
            X_Labels=X_Labels, Y_Label='KS (max)')
plt.show()
