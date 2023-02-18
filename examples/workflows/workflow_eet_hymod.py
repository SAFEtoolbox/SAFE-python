"""
This script provides a basic application example of the Elementary Effects Test.
Useful to get started with the EET.

METHOD

This script provides an example of application of the Elementary Effects
Test (EET) or 'method of Morris' (Morris, 1991; Saltelli et al., 2008).

The EET is a One-At-the-Time method for Global Sensitivity Analysis.
It computes two indices for each input:

i) the mean (mi) of the EEs, which measures the total effect of an input
over the output;

ii) the standard deviation (sigma) of the EEs, which measures the degree
of interactions with the other inputs and the degree of non-linearity of the
model response.

Both sensitivity indices are relative measures, i.e. their value does not
have any specific meaning per se but they can only be used in pair-wise
comparison (e.g. if input x(1) has higher mean EEs than input x(3), then
x(1) is more influential than x(3)).

For an application example in the environmental domain, see for instance
Nguyen and de Kok (2007).

MODEL AND STUDY AREA

The model under study is the rainfall-runoff model Hymod (see help of function
HyMod.hymod_sim for more details)
applied to the Leaf catchment in Mississipi, USA (Sorooshian et al., 1983).
The inputs subject to SA are the 5 model parameters, and the scalar output for
SA is a metric of model performance

INDEX

Steps:
1. Set current working directory and import python modules
2. Load data and set-up the Hymod model
3. Sample inputs space
4. Run the model against input samples
5. Compute the elementary effects
6. Example of how to repeat computions after adding up new
   input/output samples.

REFERENCES

Morris, M.D. (1991), Factorial sampling plans for preliminary
computational experiments, Technometrics, 33(2).

Nguyen, T.G. and de Kok, J.L. (2007). Systematic testing of an integrated
systems model for coastal zone management using sensitivity and
uncertainty analyses. Env. Mod. & Soft., 22, 1572-1587.

Saltelli, A., et al. (2008) Global Sensitivity Analysis, The Primer,
Wiley.

Sorooshian, S., Gupta, V., Fulton, J. (1983). Evaluation of maximum
likelihood parameter estimation techniques for conceptual rainfall-runoff
models: Influence of calibration data variability and length on model
credibility. Water Resour. Res., 19, 251-259.

This script prepared by  Fanny Sarrazin, 2019
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
from safepython.sampling import OAT_sampling, Morris_sampling, OAT_sampling_extend # functions
# to perform the input sampling
from safepython.util import aggregate_boot # function to aggregate the bootstrap results

from safepython import HyMod

#%% Step 2: (setup the Hymod model)

# Specify the directory where the data are stored (CHANGE TO YOUR OWN DIRECTORY)
mydir = r'Y:\Home\sarrazin\SAFE\SAFE_Python\SAFE-python-0.1.1\examples\data'
# Load data:
data = np.genfromtxt(mydir +'\LeafCatch.txt', comments='%')
rain = data[0:365, 0] # 2-year simulation
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
# The shape parameters of the uniform distribution are the lower limit and the
# difference between lower and upper limits:
distr_par = [np.nan] * M
for i in range(M):
    distr_par[i] = [xmin[i], xmax[i] - xmin[i]]

# Name of parameters (will be used to customize plots):
X_Labels = ['Sm', 'beta', 'alfa', 'Rs', 'Rf']

# Define output:
fun_test = HyMod.hymod_nse

#%% Step 3 (sample inputs space)
r = 100 # Number of Elementary Effects
# [notice that the final number of model evaluations will be equal to
# r*(M+1)]

# Option 1: use the sampling method originally proposed by Morris (1991):
#L = 6  # number of levels in the uniform grid
#design_type = 'trajectory' # (not used here but required later)
#X = Morris_sampling(r, xmin, xmax, L) # shape (r*(M+1),M)

# Option 2: Latin Hypercube sampling strategy
samp_strat = 'lhs' # Latin Hypercube
design_type = 'radial'
# other options for design type:
# design_type  = 'trajectory'
X = OAT_sampling(r, M, distr_fun, distr_par, samp_strat, design_type)

#%% Step 4 (run the model)
Y = model_execution(fun_test, X, rain, evap, flow, warmup) # shape (r*(M+1),1)

#%% Step 5 (Computation of the Elementary effects)

mi, sigma, _ = EET.EET_indices(r, xmin, xmax, X, Y, design_type)

# Plot results in the plane (mean(EE), std(EE)):
EET.EET_plot(mi, sigma, X_Labels)
plt.show()

# Use bootstrapping to derive confidence bounds:
Nboot = 1000
# Compute sensitivity indices for Nboot bootstrap resamples:
mi, sigma, EE = EET.EET_indices(r, xmin, xmax, X, Y, design_type, Nboot=Nboot)
# mi and sigma have shape (Nboot, M)
# Compute mean and confidence intervals of the sensitivity indices across the
# bootstrap resamples:
mi_m, mi_lb, mi_ub = aggregate_boot(mi) # shape (M,)
sigma_m, sigma_lb, sigma_ub = aggregate_boot(sigma) # shape (M,)

# Plot bootstrapping results in the plane (mean(EE),std(EE)):
EET.EET_plot(mi_m, sigma_m, X_Labels, mi_lb, mi_ub, sigma_lb, sigma_ub)
plt.show()

# Repeat computations using a decreasing number of samples so as to assess
# if convergence was reached within the available dataset:
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

# Repeat convergence analysis using bootstrapping:
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

#%% Step 6 (Adding up new samples)

# Add new parameter samples:
r2 = 200 # New sample size
X2, Xnew = OAT_sampling_extend(X, r2, distr_fun, distr_par, design_type)
# extended sample (it includes the already evaluated sample 'X' and the new one)

# Evaluate model against the new sample:
Ynew = model_execution(fun_test, Xnew, rain, evap, flow, warmup) # shape((r2-r)*(M+1),1)

# Put new and old results together:
Y2 = np.concatenate((Y, Ynew)) # shape (r2*(M+1),1)

# Recompute indices:
Nboot = 1000
mi2, sigma2, EE2 = EET.EET_indices(r2, xmin, xmax, X2, Y2, design_type, Nboot)
# Compute mean and confidence intervals of the sensitivity indices across the
# bootstrap resamples:
mi2_m, mi2_lb, mi2_ub = aggregate_boot(mi2) # shape (M,)
sigma2_m, sigma2_lb, sigma2_ub = aggregate_boot(sigma2) # shape (M,)
# Plot new bootstrapping results in the plane (mean(EE),std(EE)):
EET.EET_plot(mi2_m, sigma2_m, X_Labels, mi2_lb, mi2_ub, sigma2_lb, sigma2_ub)
plt.show()

# Repeat convergence analysis:
Nboot = 1000
rr2 = np.linspace(r2/5, r2, 5).astype(int)
mic2, sigmac2 = EET.EET_convergence(EE2, rr2, Nboot)
# Compute mean and confidence intervals of the sensitivity indices across the
# bootstrap resamples:
mic2_m, mic2_lb, mic2_ub = aggregate_boot(mic2) # shape (M,)
sigmac2_m, sigmac2_lb, sigmac2_ub = aggregate_boot(sigmac2) # shape (M,)
# Plot the sensitivity measure (mean of elementary effects) as a function
# of model evaluations:
plt.figure()
pf.plot_convergence(mic2_m, rr2*(M+1), mic2_lb, mic2_ub,
                    X_Label='no of model evaluations', Y_Label='mean of EEs',
                    labelinput=X_Labels)
plt.show()
