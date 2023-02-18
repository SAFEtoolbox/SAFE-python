"""
This script provides an examples of how to use the SAFE Toolbox in
combination with a model that does not run under matlab.

STEPS

1) In Matlab: use SAFE sampling functions to sample the input space.
Then save the input samples in a text file that will be passed on to the
external model.

2) Outside Matlab [not shown in this workflow]: run the model against
each input sample and save the corresponding output into a text file.

3) In Matlab: load the text file and use SAFE post-processing functions
to compute sensitivity indices.

This script prepared by Fanny Sarrazin, 2019
fanny.sarrazin@ufz.de
"""

#%% Step 1: (import python modules)

from __future__ import division, absolute_import, print_function

import numpy as np
import scipy.stats as st

from safepython.sampling import AAT_sampling, OAT_sampling # module to perform
# the input sampling

# %% Step 2 (Create input sample)

# a) Define the input feasible space

# Number of inputs:
M = 5
# Parameter ranges::
xmin = [0, 0, 0, 0, 0.1]
xmax = [400, 2, 1, 0.1, 1]

# Parameter distributions:
distr_fun = st.uniform # distraibution (here we choose the same for all inputs)
# The shape parameters of the uniform distribution are the lower limit and the
# difference between lower and upper limits:
distr_par = [np.nan] * M
for i in range(M):
    distr_par[i] = [xmin[i], xmax[i] - xmin[i]]

# b) Perform sampling

# For instance, One-At-the-Time sampling (see workflow about EET to learn
# more about available options for OAT):
r = 30 # number of sampling points
samp_strat = 'lhs' # Latin Hypercube
design_type = 'radial'# Type of design
X = OAT_sampling(r, M, distr_fun, distr_par, samp_strat,design_type)

# Or, All-At-the-Time sampling (again, see workflow about RSA for more
# options):
samp_strat = 'lhs' # Latin Hypercube
N = 3000 # Number of samples
X = AAT_sampling(samp_strat, M, distr_fun, distr_par, N)

# c) Save to file:

# Specify name for the input file:
input_file = '\Input_samples.txt'

# Specify the directory where the input file will be saved:
my_dir = r'Y:\Home\sarrazin\SAFE\SAFE_Python'

# Choose the format type (same format for all inputs):
formattype = '%3.3f'
# Or choose a different format for each input:
# formattype = ['%3.1f', '%1.2f', '%1.2f', '%1.3f', '%1.3f']
# Or, let the function choose the 'more compact' format (but in this case
# double-check that numbers in the file have the required precision):
# formattype = '%g'

# Save to text file:
np.savetxt(my_dir + input_file, X, fmt=formattype)

# You can also add a header to the file
header = 'This file was created by XXX'
np.savetxt(my_dir + input_file, X, fmt=formattype, header=header)

#%% Step 3

# Run the model outside matlab and generate the output file

#%% Step 4 (Perform Sensitivity Analysis)

# Specify the directory where the output file will be saved:
my_dir = r'Y:\Home\sarrazin\SAFE\SAFE_Python'

# Output file name:
output_file = '\Output_samples.txt'

# Load data from file:
Y = np.genfromtxt(my_dir + output_file, comments='#')

# From now on, just use all the functions in RSA, EET, VBSA, etc.



