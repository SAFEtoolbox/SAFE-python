"""
Created on DATE

This script provides an application example of VBSA where the original
uncertain model inputs are grouped into a lower number of uncertain groups.
As an example, we use the HBV rainfall-runoff model combined with a snow
melting routine, and apply SA to investigate propagation of parameter
uncertainty. The total number of parameters of this model is 13, however
we here group them into 3 groups:
    - parameters referring to the snowmelt routine,
    - parameters of the soil moisture accounting component
    - and parameters of the flow routing component.

See also 'workflow_vbsa_hymod.py' about VBSA
and 'workflow_eet_hbv.py' about HBV model and study area.

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

from safepython import HBV

#%% Step 2 (setup the HBV model and define model output)

# Define option for model structure:
# Case = 1 > interflow is dominant
# Case = 2 > percolation is dominant
Case = 1

# Setup warmup period:
warmup = 30 # (days)

# Specify the directory where the data are stored (CHANGE TO YOUR OWN DIRECTORY)
mydir = r'Y:\Home\sarrazin\SAFE\SAFE_Python\SAFE-python-0.1.1\examples\data'
# Load data:
data = np.genfromtxt(mydir + r'\01055500.txt', comments='%')

# Prepare forcing data (extract simulation period):
date = [[1948, 10, 1], [1950, 9, 30]] #  2-year simulation
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

# Define output:
fun_test = HBV.hbv_snow_objfun

#%% Step 3 (define input ranges)

# Snow routine parameters:
# 0. Ts     = threshold temperature [C]
# 1. CFMAX  = degree day factor [mm/C]
# 2. CFR    = refreezing factor [-]
# 3. CWH    = Water holding capacity of snow [-]

# HBV parameters (soil moisture account):
# 4. BETA   = Exponential parameter in soil routine [-]
# 5. LP     = evapotranspiration limit [-]
# 6. FC     = field capacity [mm]

# HBV parameters (flow routing):
# 7. PERC   = maximum flux from Upper to Lower Zone [mm/Dt]
# 8. K0     = Near surface flow coefficient (ratio) [1/Dt]
# 9. K1    = Upper Zone outflow coefficient (ratio) [1/Dt]
# 10. K2    = Lower Zone outflow coefficient (ratio) [1/Dt]
# 11. UZL   = Near surface flow threshold [mm]
# 12. MAXBAS= Flow routing coefficient [Dt]

# Define parameter ranges (from Kollat et al., 2012):
xmin = [-3, 0, 0, 0, 0, 0.3, 1, 0, 0.05, 0.01, 0.05, 0, 1]
xmax = [3, 20, 1, 0.8, 7, 1, 2000, 100, 2, 1, 0.1, 100, 6]

# Parameter distributions:
M = len(xmin) # number of uncertain parameters
distr_fun = [st.uniform] * M # uniform distribution
distr_fun[-1] = st.randint # discrete uniform distribution
# (parameter MAXBAS is an integer)
# The shape parameters of the uniform distribution are the lower limit and the
# difference between lower and upper limits:
distr_par = [np.nan] * M
for i in range(M-1):
    distr_par[i] = [xmin[i], xmax[i] - xmin[i]]
# The shape parameters of the discrete uniform distribution are the lower limit
# and the upper limit+1:
distr_par[-1] = [xmin[-1], xmax[-1] + 1]

#%% Step 4 - define groups

# Define grouping:
groups = [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9, 10, 11, 12]]

# Choose names of groups (needed for plots):
G_labels = ['snow', 'soil', 'routing']

Mg = len(G_labels) # number of uncertain groups

#%% Step 5 (build catalogue)

# Choose length of each catalogue:
Ng = [10000, 10000, 100000]

# Build the catalogue (via sampling):
samp_strat = 'rsu'
CAT = [np.nan] * Mg
for i in range(Mg):
    distr_fun_i = [distr_fun[j] for j in groups[i]]
    distr_par_i = [distr_par[j] for j in groups[i]]
    CAT[i] = AAT_sampling(samp_strat, len(groups[i]), distr_fun_i,
                          distr_par_i, Ng[i])

#%% Step 6 (perform sampling)

# Choose sampling strategy:
samp_strat = 'lhs'

# Choose base sample size:
N = 3000

# Set distribution of the catalouge index variables to discrete uniform:
distr_funG = [st.randint] * Mg
# Set distribution parameter to number of elements in the catalogue:
distr_parG = [np.nan] * Mg
for i in range(Mg):
    distr_parG[i] = [0, Ng[i]]

# Perform sampling (over indices of catalogue):
X = AAT_sampling(samp_strat, Mg, distr_funG, distr_parG, 2*N)

# Create additional samples through resampling strategy:
XA, XB, XC = VB.vbsa_resampling(X)
XA = XA.astype(int)
XB = XB.astype(int)
XC = XC.astype(int)

# Transform back to original inputs by reading the catalogue:
XAp = np.nan * np.zeros((N, M))
XBp = np.nan * np.zeros((N, M))
XCp = np.nan * np.zeros((N*Mg, M))

for i in range(Mg):
    CAT_i = CAT[i]
    XAp[:, np.array(groups[i])] = CAT_i[XA[:, i], :]
    XBp[:, np.array(groups[i])] = CAT_i[XB[:, i], :]
    XCp[:, np.array(groups[i])] = CAT_i[XC[:, i], :]

#%% Step 7 (run the model)

# Warning: it may take some time to run the following lines:
YA = model_execution(fun_test, XAp, prec, temp, ept, flow, warmup, Case) # size (N,1)
YB = model_execution(fun_test, XBp, prec, temp, ept, flow, warmup, Case) # size (N,1)
YC = model_execution(fun_test, XCp, prec, temp, ept, flow, warmup, Case) # size (N*Mg,1)

#%% Step 8 (compute sensitivity indices)

# Choose which model output to analyze:
j = 1 # 0:AME, 1:NSE, 2:BIAS, 3:TRMSE, 4:SFDCE, 5:RMSE,
# see help of 'HBV.hbv_snow_objfun' for more information

# Compute main (first-order) and total effects:
Si, STi = VB.vbsa_indices(YA[:, j], YB[:, j], YC[:, j], Mg)

# Plot results (plot main and total separately):
plt.figure()
plt.subplot(121)
pf.boxplot1(Si, G_labels, 'main effects')
plt.subplot(122)
pf.boxplot1(STi, G_labels, 'total effects')
plt.show()

# Plot results (both in one plot):
plt.figure()
pf.boxplot2(np.stack((Si, STi)), G_labels, legend=['main effects', 'total effects'])
plt.show()
