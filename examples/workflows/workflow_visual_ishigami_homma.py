"""
This script provides an application example of how to use several
visualization tools (scatter plots, 2D coloured scatter plots, parallel
coordinate plots) to learn about sensitivity.
The application example is the Hishigami-Homma function, which is a
standard benchmark function in the Sensitivity Analysis literature.
(see help of function 'ishigami_homma.ishigami_homma_function' for more details
and references).

This script was prepared by Fanny Sarrazin, 2019
fanny.sarrazin@ufz.de
"""

#%% Step 1: (import python modules)

from __future__ import division, absolute_import, print_function

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

from safepython.model_execution import model_execution
import safepython.plot_functions as pf # module to visualize the results
from safepython.sampling import AAT_sampling

from safepython.ishigami_homma import ishigami_homma_function

#%% Step 2 (setup the model)

# Number of uncertain parameters subject to SA:
M = 3

# Parameter ranges
xmin = -np.pi
xmax = np.pi
# Parameter distributions
distr_fun = st.uniform # uniform distribution
# The shape parameters of the uniform distribution are the lower limit and the
# difference between lower and upper limits:
distr_par = [xmin, xmax - xmin]

# Define output:
fun_test = ishigami_homma_function

#%% Step 3 (sampling and model evaluation)
samp_strat = 'lhs' # Latin Hypercube
N = 3000  #  Number of samples
X = AAT_sampling(samp_strat, M, distr_fun, distr_par, N)
Y = model_execution(fun_test, X)

#%% Step 4 (Scatter plots)
# Use scatter plots of inputs againts output to visually assess
# direct effects:
plt.figure(); pf.scatter_plots(X, Y)
plt.show()
# Use coloured scatter plots of one input against another on to assess
# interactions:
i1 = 0
i2 = 1
plt.figure(); pf.scatter_plots_col(X, Y, i1, i2) # plot x[i1] against x[i2]
plt.show()
# Change i2:
i2 = 2
plt.figure(); pf.scatter_plots_col(X, Y, i1, i2)
plt.show()
# Put all possible combinations of i1,i2 into one figure:
pf.scatter_plots_interaction(X, Y)
plt.show()
# Customize titles:
pf.scatter_plots_interaction(X, Y, X_Labels=['x(1)', 'x(2)', 'x(3)'])
plt.show()

#%% Step 5 (Parallel Coordinate Plot)

# Use Parallel Coordinate Plot to find patterns of input combinations
# mapping into specific output condition
idx = Y > 30 # output condition to be highlighted
plt.figure(); pf.parcoor(X, X_Labels=['x(1)', 'x(2)', 'x(3)'], idx=idx)
plt.show()