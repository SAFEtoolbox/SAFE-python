"""
This script provides an application example of the PAWN sensitivity analysis
approach (Pianosi and Wagener, 2015,2018).
Useful to learn about how to use PAWN when one of the parameter takes
discrete values.

MODEL AND STUDY AREA

The model under study is the HBV rainfall-runoff model,
the inputs subject to SA are the 13 model parameters,
and the outputs for SA are a set of performance metrics.
See help of function 'hbv_snow_objfun.m' for more details
and references about the HBV model and the objective functions.

The HBV model has a parameter that takes discrete values. This workflow 
demonstrates how discrete inputs are handled to calculate PAWN sensitivity 
indices.

The case study area is is the Nezinscot River at Turner center, Maine,
USA (USGS 01055500, see http://waterdata.usgs.gov/nwis/nwismap)

REFERENCES

Pianosi, F. and Wagener, T. (2018), Distribution-based sensitivity
analysis from a generic input-output sample, Env. Mod. & Soft., 108, 197-207.

Pianosi, F. and Wagener, T. (2015), A simple and efficient method
for global sensitivity analysis based on cumulative distribution
functions, Env. Mod. & Soft., 67, 1-11.

This script prepared by Fanny Sarrazin, 2019
fanny.sarrazin@ufz.de
"""

#%% Step 1: (import python modules)

from __future__ import division, absolute_import, print_function

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

# Import SAFE modules:
from safepython import PAWN
import safepython.plot_functions as pf # module to visualize the results
from safepython.model_execution import model_execution # module to execute the model
from safepython.sampling import AAT_sampling # module to perform the input sampling
from safepython.util import aggregate_boot # function to aggregate the bootstrap results

from safepython import HBV

#%% Step 2: (setup the HBV model)

# Specify the directory where the data are stored (CHANGE TO YOUR OWN DIRECTORY)
mydir = r'Y:\Home\sarrazin\SAFE\SAFE_Python\SAFE-python-0.1.1\examples\data'
# Load data:
data = np.genfromtxt(mydir + r'\01055500.txt', comments='%')

Case = 1 # Case=1: interflow is dominant / Case = 2: percolation is dominant
warmup = 365 # Model warmup period (days)

# Prepare forcing data (extract simulation period):
date = [[1948, 10, 1], [1953, 9, 30]] #  5-year simulation
t_start = np.where(np.logical_and(np.logical_and(
    data[:, 0] == date[0][0], data[:, 1] == date[0][1]),
                                  data[:, 2] == date[0][2]))[0]
t_end = np.where(np.logical_and(np.logical_and(
    data[:, 0] == date[1][0], data[:, 1] == date[1][1]),
                                data[:, 2] == date[1][2]))[0]
tt = np.arange(t_start, t_end+1, 1)
prec = data[tt, 3]
ept = data[tt, 4]
flow = data[tt, 5]
temp = np.mean(data[tt, 6:8], axis=1)

# Uncertain parameters:
X_Labels = ['TS', 'CFMAX', 'CFR', 'CWH', 'BETA', 'LP', 'FC', 'PERC', 'K0',
            'K1', 'K2', 'UZL', 'MAXBAS']
M = len(X_Labels) # number of uncertain parameters

# Parameter ranges (from Kollat et al.(2012)):
xmin = [-3, 0, 0, 0, 0, 0.3, 1, 0, 0.05, 0.01, 0.05, 0, 1]
xmax = [3, 20, 1, 0.8, 7, 1, 2000, 100, 2, 1, 0.1, 100, 6]

# Parameter distributions:
distr_fun = [st.uniform] * M # uniform distribution
distr_fun[-1] = st.randint # discrete uniform distribution
# The shape parameters of the uniform distribution are the lower limit and the
# difference between lower and upper limits:
distr_par = [np.nan] * M
for i in range(M-1):
    distr_par[i] = [xmin[i], xmax[i] - xmin[i]]
# The shape parameters of the discrete uniform distribution are the lower limit
# and the upper limit+1:
distr_par[-1] = [xmin[-1], xmax[-1] + 1]

# Define output:
fun_test = HBV.hbv_snow_objfun

#%% Step 3 (sample inputs space and run the model)

samp_strat = 'lhs' # Latin Hypercube
N = 3000  #  Number of samples
X = AAT_sampling(samp_strat, M, distr_fun, distr_par, N)
# Run the model:
Y = model_execution(fun_test, X, prec, temp, ept, flow, warmup, Case)

#%% Step 4 (Apply PAWN)

# Set the number of conditioning intervals:
# option 1 (same value for all inputs):
n = 10
# option 2 (varying value across inputs)
# n = [10]*M
# n[-1] = 6 # a different number of conditioning intervals is chosen input X13
# (parameter MAXBAS)
# Note that the PAWN functions handle discrete inputs/inputs that have values 
# that are repeated, so that all occurences of a given value belong to the same 
# conditioning interval

# Choose one among multiple outputs for subsequent analysis:
Yi = Y[:, 1]

# Check how the sample is split for parameter MAXBAS that takes discrete values:
YY, xc, NC, n_eff, Xk, XX = PAWN.pawn_split_sample(X, Yi, n)
print(n_eff) # number of conditioning intervals for each input (for MAXBAS, the
# number of intervals was changed to match the number of discrete values)
print(NC[1]) # number of data points in each conditioning interval conditioning
# intervals for input Ts
print(NC[-1])# number of data points in each conditioning interval conditioning
# intervals for input MAXBAS

# Important remarks:
# - When the prescribed number of conditioning intervals for parameter MAXBAS
# is > 6, a warning message is displayed: as MAXBAS takes discrete values
# (6 possible values) the actual number of conditioning intervals used for that
# input when applying PAWN is 6.
# - The number of data points in each conditioning intervals may vary across
# inputs and may also vary across the conditioning intervals of parameter MAXBAS.
# This means that the sensitivity indices (Kolmogorov Smirnov or KS statistic)
# may be estimated using different numbers of data points.

# Compute and plot conditional and unconditional CDFs
YF, FU, FC, xc = PAWN.pawn_plot_cdf(X, Yi, n)
plt.show()
# Add colorbar:
YF, FU, FC, xc = PAWN.pawn_plot_cdf(X, Yi, n, cbar=True, n_col=3, labelinput=X_Labels)
# You can also adjust the spacing between the subplots to create a nice figure
# using the function subplots_adjust:
#plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
#                    hspace=0.5, wspace=0.4)
plt.show()

# Compute and plot KS statistics for each conditioning interval
# (notice the number of KS values for each input and in particular for MAXBAS):
KS = PAWN.pawn_plot_ks(YF, FU, FC, xc)
plt.show()
# Customize plot:
KS = PAWN.pawn_plot_ks(YF, FU, FC, xc, X_Labels=X_Labels)
# You can also adjust the spacing between the subplots to create a nice figure
# using the function subplots_adjust:
#plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
#                    hspace=0.5, wspace=0.4
plt.show()

# Compute PAWN sensitivity indices:
KS_median, KS_mean, KS_max = PAWN.pawn_indices(X, Yi, n)
# Plot results 8for instance for KS_max):
plt.figure()
pf.boxplot1(KS_max, X_Labels=X_Labels, Y_Label='Ks (max)')
plt.show()

# Use bootstrapping to derive confidence bounds:
Nboot = 1000
# Compute sensitivity indices for Nboot bootstrap resamples
# (Warning: the following line may take some time to run, as the computation of
# CDFs is costly):
KS_median, KS_mean, KS_max = PAWN.pawn_indices(X, Yi, n, Nboot=Nboot)
# KS_median and KS_mean and KS_max have shape (Nboot, M)
# Compute mean and confidence intervals of the sensitivity indices across the
# bootstrap resamples:
KS_median_m, KS_median_lb, KS_median_ub = aggregate_boot(KS_median) # shape (M,)
KS_mean_m, KS_mean_lb, KS_mean_ub = aggregate_boot(KS_mean) # shape (M,)
KS_max_m, KS_max_lb, KS_max_ub = aggregate_boot(KS_max) # shape (M,)

# Plot bootstrapping results (for instance for KS_max):
plt.figure()
pf.boxplot1(KS_max_m, S_lb=KS_max_lb, S_ub=KS_max_ub,
            X_Labels=X_Labels, Y_Label='Ks (max)')
plt.show()
