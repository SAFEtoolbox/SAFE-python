"""
This script provides an application example of time-varying
sensitivity analysis.
The user can choose one of three GSA methods (FAST, VBSA or EET)
and this script will guide through the application of that method
to a time-varying model output.
In this example, the time-varying model output is the model prediction
(estimated flow) at each time step of the simulation.

MODEL AND STUDY AREA

The model under study is the rainfall-runoff model Hymod
(see help of function hymod_sim.m for more details)
applied to the Leaf catchment in Mississipi, USA
(see header of file LeafCatch.txt for more details).
The inputs subject to SA are the 5 model parameters,
and the time-varying model output is the model prediction
(estimated flow) at each time step of the simulation.
For an example of how to use this type of analysis,
see for instance Reusser and Zehe (2011)

REFERENCES

Reusser, D. E., Zehe, E., 2011. Inferring model structural deficits
by analyzing temporal dynamics of model performance and parameter
sensitivity. Water Resources Research 47 (7).

This script was prepared by Fanny Sarrazin, 2019
fanny.sarrazin@ufz.de
"""

#%% Step 1: (import python modules)

from __future__ import division, absolute_import, print_function

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

# Import SAFE modules
import safepython.VBSA as VB # module to perform VBSA
from safepython import FAST # module to perform FAST
from safepython import EET # module to perform EET
from safepython.model_execution import model_execution # module to execute the model
from safepython.sampling import AAT_sampling, OAT_sampling # functions to perform the input sampling

from safepython import HyMod

#%% Step 2: (Setup the Hymod model and define input variability space)

# Specify the directory where the data are stored (CHANGE TO YOUR OWN DIRECTORY)
mydir = r'Y:\Home\sarrazin\SAFE\SAFE_Python\SAFE-python-0.1.1\examples\data'
# Load data:
data = np.genfromtxt(mydir +'\LeafCatch.txt', comments='%')
T = 2*365
rain = data[0:T, 0] # 1-year simulation
evap = data[0:T, 1]
flow = data[0:T, 2]

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

#%% Step 3: (Define the model output and GSA method)

# Choose the function for model simulation:
simfun = HyMod.hymod_sim
warmup = 30 # Model warmup period (days)
# (sensitivity indices will not be computed for the warmup period)

# Choose GSA method
#GSA_met = 'FAST'
GSA_met = 'EET'
#GSA_met = 'VBSA'

#%% Step 4: (Define choices for sampling)

r = 20 # number of EEs (needed for EET only)

N = 3001 # sample size (needed for all other methods)

# Notes:
# - Case VBSA: N is the size of the base sample: the total number of model
# evaluations will be N*(M+2) [more on this: see help VBSA.vbsa_resampling]
# - Case FAST: N must be odd [more on this: see help FAST.FAST_sampling]
# - Case EET: N is not used, total number of model evaluations depend on 'r'
# and precisely it will be r*(M+1) [more on this: see help sampling.OAT_sampling]

samp_strat = 'lhs' # Sampling strategy (needed for EET and VBSA)

#%% Step 5: (Sample inputs space and evaluate the model)

if GSA_met == 'FAST':

    X, _ = FAST.FAST_sampling(distr_fun, distr_par, M, N)
    Y = model_execution(simfun, X, rain, evap) # shape (N,T)

elif GSA_met == 'VBSA':

    X = AAT_sampling(samp_strat, M, distr_fun, distr_par, 2*N)
    XA, XB, XC = VB.vbsa_resampling(X)
    YA = model_execution(simfun, XA, rain, evap) # shape (N,T)
    YB = model_execution(simfun, XB, rain, evap) # shape (N,T)
    YC = model_execution(simfun, XC, rain, evap) # shape (N*M,T)
    Y = np.concatenate((YA, YB, YC), axis=0) # shape (N*(2+M),T)
    # ... only needed for the next plot!

elif GSA_met == 'EET':

    design_type = 'radial'
    X = OAT_sampling(r, M , distr_fun, distr_par, samp_strat, design_type)
    # More options are actually available for EET sampling,
    # see workflow_eet_hymod.py
    Y = model_execution(simfun, X, rain, evap) # shape (r*(M+1),T)

else:
    raise ValueError('No method called '+ GSA_met)

# Plot results:
plt.figure()
plt.plot(np.transpose(Y), 'b');
plt.xlabel('time')
plt.ylabel('flow')
plt.xlim((0, T-1))
plt.ylim((0, np.max(Y)))

#%% Step 6 (Compute time-varying Sensitivity Indices)

import time
start_time = time.time()
if GSA_met == 'FAST':

    Si = np.nan * np.zeros((T, M))
    for t in range(warmup, T):
        Si[t, :], _, _, _, _ = FAST.FAST_indices(Y[:, t], M)
    S_plot = np.transpose(Si)

elif GSA_met == 'VBSA':

    Si  = np.nan * np.zeros((T, M))
    STi = np.nan * np.zeros((T, M))
    for t in range(warmup, T):
        Si[t, :], STi[t, :] = VB.vbsa_indices(YA[:, t], YB[:, t], YC[:, t], M)

    # select sensitivity index to be plotted in the next figure:
    S_plot = np.transpose(Si)
    # S_plot = np.transpose(STi)

elif GSA_met == 'EET':

    mi = np.nan * np.zeros((T, M))
    sigma = np.nan * np.zeros((T, M))
    for t in range(warmup, T):
        mi[t, :], sigma[t, :], _ = EET.EET_indices(r, xmin, xmax, X, Y[:, t],
                                              design_type)

    # select sensitivity index to be plotted in the next figure:
    S_plot = np.transpose(mi)
    # S_plot = np.transpose(sigma)

time.time() - start_time
# Plot results:
plt.figure()
c = plt.pcolormesh(S_plot[:, warmup:T], cmap='YlOrRd')

plt.xlabel('time')
plt.ylabel('inputs')
plt.title('Sensitivity indices - GSA meth: ' + GSA_met)
plt.yticks(np.arange(0.5, M+0.5), X_Labels)
cb = plt.colorbar(c)
cb.set_label('sensitivity index')
plt.clim(-0.1, max(1.1, np.max(S_plot[:, warmup:T])))

# Add flow:
Nplot = Y.shape
Nplot = Nplot[0] # scaling factor
plt.plot(flow[warmup:T]/max(flow[warmup:T])*M, 'k', linewidth=2)
plt.show()

