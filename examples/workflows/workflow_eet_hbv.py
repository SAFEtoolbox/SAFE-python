"""
This script provides an application example of the
Elementary Effects Test to the HBV rainfall-runoff model.
Useful to learn about how to use the EET when one of the parameter takes
discrete values.

METHOD

see description in 'workflow_eet_hymod.py'

MODEL AND STUDY AREA

The model under study is the HBV rainfall-runoff model,
the inputs subject to SA are the 13 model parameters,
and the outputs for SA are a set of performance metrics.
See help of function 'hbv_snow_objfun.m' for more details
and references about the HBV model and the objective functions.

The case study area is is the Nezinscot River at Turner center, Maine,
USA (USGS 01055500, see http://waterdata.usgs.gov/nwis/nwismap)

This script prepared by Fanny Sarrazin, 2019
fanny.sarrazin@ufz.de
"""
#%% Step 1: (import python modules)

from __future__ import division, absolute_import, print_function

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

import safepython.EET as EET # module to perform the EET
import safepython.plot_functions as pf # module to visualize the results
from safepython.model_execution import model_execution # module to execute the model
from safepython.sampling import OAT_sampling, Morris_sampling # module to perform the input
# sampling
from safepython.util import aggregate_boot # function to aggregate the bootstrap results

from safepython import HBV

#%% Step 2: (setup the Hymod model)

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

#%% Step 3 (sample inputs space)

r = 100 # Number of Elementary Effects
# [notice that the final number of model evaluations will be equal to
# r*(M+1)]

# option 1: use the sampling method originally proposed by Morris (1991):
#L = 6  # number of levels in the uniform grid
#design_type  = 'trajectory' # (not used here but required later)
#X = Morris_sampling(r, xmin, xmax, L) # shape (r*(M+1),M)

# option 2: Latin Hypercube sampling strategy
samp_strat = 'lhs' # Latin Hypercube
design_type = 'radial'
# other options for design type:
# design_type  = 'trajectory'
X = OAT_sampling(r, M, distr_fun, distr_par, samp_strat, design_type)

#%% Step 4 (run the model)
Y = model_execution(fun_test, X, prec, temp, ept, flow, warmup, Case) # shape (r*(M+1),1)

#%% Step 5 (Computation of the Elementary effects)

# Choose one among multiple outputs for subsequent analysis:
Yi = Y[:, 1]
#Compute Elementary Effects:
mi, sigma,_ = EET.EET_indices(r, xmin, xmax, X, Yi, design_type)

# Plot results in the plane (mean(EE),std(EE)):
EET.EET_plot(mi, sigma, X_Labels)
plt.show()

# Use bootstrapping to derive confidence bounds:
Nboot = 1000
# Compute sensitivity indices for Nboot bootstrap resamples:
mi, sigma, EE = EET.EET_indices(r, xmin, xmax, X, Yi, design_type, Nboot=Nboot)
# mi and sigma have shape (Nboot, M)
# Compute mean and confidence intervals of the sensitivity indices across the
# bootstrap resamples:
mi_m, mi_lb, mi_ub = aggregate_boot(mi) # shape (M,)
sigma_m, sigma_lb, sigma_ub = aggregate_boot(sigma) # shape (M,)

# Plot bootstrapping results in the plane (mean(EE),std(EE)):
EET.EET_plot(mi_m, sigma_m, X_Labels, mi_lb, mi_ub, sigma_lb, sigma_ub)
plt.show()

# Convergence analysis:
rr = np.linspace(r/5, r, 5).astype(int) # Sample sizes at which the indices will
# be estimated
mic, sigmac = EET.EET_convergence(EE, rr) # mic and sigmac are lists in which
# the i-th element correspond to the sensitivity indices at the i-th sample size
# Plot the sensitivity measure (mean of elementary effects) as a function
# of the number of model evaluations:
plt.figure()
pf.plot_convergence(mic, rr*(M+1), X_Label='no of model evaluations',
                    Y_Label='mean of EEs', labelinput=X_Labels)
plt.show()

# Convergence analysis using bootstrapping:
# Compute sensitivity indices for Nboot bootstrap resamples:
mic, sigmac = EET.EET_convergence(EE, rr, Nboot) # mic and sigmac are lists in
# which the i-th element correspond to the sensitivity indices at the i-th sample size
# Compute mean and confidence intervals of the sensitivity indices across the
# bootstrap resamples:
mic_m, mic_lb, mic_ub = aggregate_boot(mic) # shape (R,M)
sigmac_m, sigmac_lb, sigmac_ub = aggregate_boot(sigmac) # shape (R,M)
# Plot the sensitivity measure (mean of elementary effects) as a function
# of the number of model evaluations:
plt.figure()
pf.plot_convergence(mic_m, rr*(M+1), mic_lb, mic_ub,
                    X_Label='no of model evaluations', Y_Label='mean of EEs',
                    labelinput=X_Labels)
plt.show()
