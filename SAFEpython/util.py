"""
    Module that contains utility functions

    This module is part of the SAFE Toolbox by F. Pianosi, F. Sarrazin and
    T. Wagener at Bristol University (2015).
    SAFE is provided without any warranty and for non-commercial use only.
    For more details, see the Licence file included in the root directory
    of this distribution.
    For any comment and feedback, or to discuss a Licence agreement for
    commercial use, please contact: francesca.pianosi@bristol.ac.uk
    For details on how to cite SAFE in your publication, please see:
    https://www.safetoolbox.info

    Package version: SAFEpython_v0.0.0
"""
from __future__ import division, absolute_import, print_function

from warnings import warn
import numpy as np
from numpy.matlib import repmat
from numba import jit # the function jit allows to compile the code and reduced
# the running time

@jit
def empiricalcdf(x, xi):

    """ Compute the empirical CDF of the sample 'x' and evaluate it
    at datapoints 'xi'.

    This function is called internally in RSA_thres.compute_indices and
    RSA_groups.compute_indices to calculate the input CDFs.

    Usage:
        Fi = util.empiricalcdf(x, xi)

    Input:
     x = samples to build the empirical CDF F(x)- numpy.ndarray(N,1) or (N, )
    xi = values where to evaluate the CDF       - numpy.ndarray(Ni,1) or (Ni, )

    Output:
    Fi = CDF values at 'xi'                     - numpy.ndarray (Ni, )

    Use:
    F = empiricalcdf(x, x)
    to obtain the CDF values at the same datapoints used for its construction.

    Example:

    import numpy as np
    import matplotlib.pyplot as plt
    from SAFEpython.util import empiricalcdf
    x = np.random.random((10,))
    F = empiricalcdf(x, x)
    xi = np.arange(np.min(x), np.max(x), 0.001)
    Fi = empiricalcdf(x, xi)
    plt.figure()
    plt.plot(xi, Fi, 'k', x, F, 'or');

    This function is part of the SAFE Toolbox by F. Pianosi, F. Sarrazin
    and T. Wagener at Bristol University (2015).
    SAFE is provided without any warranty and for non-commercial use only.
    For more details, see the Licence file included in the root directory
    of this distribution.
    For any comment and feedback, or to discuss a Licence agreement for
    commercial use, please contact: francesca.pianosi@bristol.ac.uk
    For details on how to cite SAFE in your publication, please see:
    https://www.safetoolbox.info """

    ###########################################################################
    # Check inputs
    ###########################################################################
    if not isinstance(x, np.ndarray):
        raise ValueError('"x" must be a numpy.array.')
    if x.dtype.kind != 'f' and x.dtype.kind != 'i' and x.dtype.kind != 'u':
        raise RuntimeError('"x" must contain floats or integers.')

    if not isinstance(xi, np.ndarray):
        raise ValueError('"xi" must be a numpy.array.')
    if xi.dtype.kind != 'f' and xi.dtype.kind != 'i' and xi.dtype.kind != 'u':
        raise ValueError('"xi" must contain floats or integers.')

    x = x.flatten() # shape (N, )
    xi = xi.flatten() # shape (Ni, )

    ###########################################################################
    # Estimate empirical CDF values at 'x':
    ###########################################################################

    N = len(x)
    F = np.linspace(1, N, N)/N

    # Remove any multiple occurance of 'x'
    # and set F(x) to the upper value (recall that F(x) is the percentage of
    # samples whose value is lower than *or equal to* x!)
    # We save the indices of the last occurence of each element in the vector 'x',
    # when 'x' is sorted in ascending order. Since the function 'np.unique' returns
    # the first occurence of each element, we first sort x in descending order
    # before applying the function 'np.unique'

    # x = sorted(x, reverse=True)
    x = np.flip(np.sort(x), axis=0)
    x, iu = np.unique(x, return_index=True)
    iu = N-1 - iu # Correct the indices so that they refer to the vector x sorted
    # in ascending order

    F = F[iu]
    N = len(F)

    # Interpolate the empirical CDF at 'xi':
    Fi = np.ones((len(xi),))

    for j in range(N-1, -1, -1):
        Fi[xi[:] <= x[j]] = F[j]

    Fi[xi < x[0]] = 0

    return Fi

def NSE(y_sim, y_obs):

    """Computes the Nash-Sutcliffe Efficiency (NSE) coefficient.

    Usage:
        nse = util.NSE(Y_sim, y_obs)

    Input:
    y_sim = time series of modelled variable              - numpy.ndarray (N, )
            (N > 1 different time series can be        or - numpy.ndarray (N,T)
            evaluated at once)
    y_obs = time series of observed variable              - numpy.ndarray (T, )
                                                       or - numpy.ndarray (1,T)

    Output:
      nse = vector of NSE coefficients                    - numpy.ndarray (N, )

    References:

    Nash, J. E. and J. V. Sutcliffe (1970),
    River flow forecasting through conceptual models part I
    A discussion of principles, Journal of Hydrology, 10 (3), 282-290.

    This function is part of the SAFE Toolbox by F. Pianosi, F. Sarrazin
    and T. Wagener at Bristol University (2015).
    SAFE is provided without any warranty and for non-commercial use only.
    For more details, see the Licence file included in the root directory
    of this distribution.
    For any comment and feedback, or to discuss a Licence agreement for
    commercial use, please contact: francesca.pianosi@bristol.ac.uk
    For details on how to cite SAFE in your publication, please see:
    https://www.safetoolbox.info"""

    Nsim = y_sim.shape
    if len(Nsim) > 1:
        N = Nsim[0]
        T = Nsim[1]
    elif len(Nsim) == 1:
        T = Nsim[0]
        N = 1
        y_sim_tmp = np.nan * np.ones((1, T))
        y_sim_tmp[0, :] = y_sim
        y_sim = y_sim_tmp

    Nobs = y_obs.shape
    if len(Nobs) > 1:
        if Nobs[0] != 1:
            raise ValueError('"y_obs" be of shape (T, ) or (1,T).')
        if Nobs[1] != T:
            raise ValueError('the number of elements in "y_obs" must be equal'+
                             'to the number of columns in "y_sim"')
    elif len(Nobs) == 1:
        if Nobs[0] != T:
            raise ValueError('the number of elements in "y_obs" must be equal'+
                             'to the number of columns in "y_sim"')
        y_obs_tmp = np.nan * np.ones((1, T))
        y_obs_tmp[0, :] = y_obs
        y_obs = y_obs_tmp

    Err = y_sim - repmat(y_obs, N, 1)
    Err0 = y_obs - np.mean(y_obs)
    nse = 1 - np.sum(Err**2, axis=1) / np.sum(Err0**2, axis=1)

    return nse


def RMSE(y_sim, y_obs):

    """Computes the Root Mean Squared Error

    Usage:
    rmse = util.RMSE(Y_sim, y_obs)

    Input:
    y_sim = time series of modelled variable              - numpy.ndarray (N, )
            (N > 1 different time series can be        or - numpy.ndarray (N,T)
            evaluated at once)
            at once)
    y_obs = time series of observed variable              - numpy.ndarray (T, )
                                                       or - numpy.ndarray (1,T)

    Output:
     rmse = vector of RMSE coefficients                   - numpy.ndarray (N, )

    This function is part of the SAFE Toolbox by F. Pianosi, F. Sarrazin
    and T. Wagener at Bristol University (2015).
    SAFE is provided without any warranty and for non-commercial use only.
    For more details, see the Licence file included in the root directory
    of this distribution.
    For any comment and feedback, or to discuss a Licence agreement for
    commercial use, please contact: francesca.pianosi@bristol.ac.uk
    For details on how to cite SAFE in your publication, please see:
    https://www.safetoolbox.info"""

    Nsim = y_sim.shape
    if len(Nsim) > 1:
        N = Nsim[0]
        T = Nsim[1]
    elif len(Nsim) == 1:
        T = Nsim[0]
        N = 1
        y_sim_tmp = np.nan * np.ones((1, T))
        y_sim_tmp[0, :] = y_sim
        y_sim = y_sim_tmp

    Nobs = y_obs.shape
    if len(Nobs) > 1:
        if Nobs[0] != 1:
            raise ValueError('"y_obs" be of shape (T, ) or (1,T).')
        if Nobs[1] != T:
            raise ValueError('the number of elements in "y_obs" must be' +
                             'equal to the number of columns in "y_sim"')
    elif len(Nobs) == 1:
        if Nobs[0] != T:
            raise ValueError('the number of elements in "y_obs" must be' +
                             'equal to the number of columns in "y_sim"')
        y_obs_tmp = np.nan * np.ones((1, T))
        y_obs_tmp[0, :] = y_obs
        y_obs = y_obs_tmp

    Err = y_sim - repmat(y_obs, N, 1)
    rmse = np.sqrt(np.mean(Err**2, axis=1))

    return rmse

def aggregate_boot(S, alfa=0.05):

    """ This function computes the mean and confidence intervals of the
    sensitivity indices across bootstrap resamples.

    Usage:
        S_m, S_lb, S_ub = util.aggregate_bootstrap(S, alfa=0.05)

    Input:
        S = array of sensitivity indices estimated    - numpy.np.array(Nboot,M)
            for each bootstrap resample at a given
            sample size
            or list of sensitivity indices         or - list (R elements)
            estimated for each bootstrap resample
            (list of R numpy.ndarrays (Nboot, M)
            where S[j] are the estimates of the
            sensitivity indices at the jth sample
            size, Nboot>=1 and M>=1)

    Optional input:
     alfa = significance level for the confidence     - float
            intervals estimated by bootstrapping
            (default: 0.05)

    Output:
      S_m = mean sensitivity indices across bootstrap - numpy.np.array(R,M)
            resamples at the different sample sizes
     S_lb = lower bound of sensitivity indices across - numpy.np.array(R,M)
             bootstrap resamples at the different
             sample sizes
     S_lb = upper bound of sensitivity indices across - numpy.np.array(R,M)
             bootstrap resamples at the different
             sample sizes

    This function is part of the SAFE Toolbox by F. Pianosi, F. Sarrazin
    and T. Wagener at Bristol University (2015).
    SAFE is provided without any warranty and for non-commercial use only.
    For more details, see the Licence file included in the root directory
    of this distribution.
    For any comment and feedback, or to discuss a Licence agreement for
    commercial use, please contact: francesca.pianosi@bristol.ac.uk
    For details on how to cite SAFE in your publication, please see:
    https://www.safetoolbox.info"""

    ###########################################################################
    # Check inputs
    ###########################################################################
    if isinstance(S, np.ndarray):
        if S.dtype.kind != 'f' and S.dtype.kind != 'i' and S.dtype.kind != 'u':
            raise ValueError('Elements in "S" must be int or float.')
        Ns = S.shape
        R = 1 # number of sample sizes
        S = [S] # create list to simply computation
        if len(Ns) != 2:
            raise ValueError('"S" must be of shape (Nboot,M) where Nboot>=1 and M>=1.')

    elif isinstance(S, list):
        if not all(isinstance(i, np.ndarray) for i in S):
            raise ValueError('Elements in "S" must be int or float.')
        Ns = S[0].shape
        R = len(S) # number of sample sizes
        if len(Ns) != 2:
            raise ValueError('"S[i]" must be of shape (Nboot,M) where Nboot>=1 and M>=1.')
    else:
        raise ValueError('"S" must be a list of a numpy.ndarray.')

    M = Ns[1]
    Nboot = Ns[0]

    ###########################################################################
    # Check optional inputs
    ###########################################################################
    if not isinstance(alfa, (float, np.float16, np.float32, np.float64)):
        raise ValueError('"alfa" must be scalar and numeric.')
    if alfa < 0 or alfa > 1:
        raise ValueError('"alfa" must be in (0,1).')

    ###########################################################################
    # Compute statistics across bootstrap resamples
    ###########################################################################
    # Variable initialization
    S_m = np.nan*np.ones((R, M))
    S_lb = np.nan*np.ones((R, M))
    S_ub = np.nan*np.ones((R, M))

    for j in range(R): # loop over sample sizes

        S_m[j, :] = np.nanmean(S[j], axis=0) # bootstrap mean
        idx = ~np.isnan(S[j][:, 1])
        if np.sum(idx) < Nboot:
            warn('Statistics were computed using ' + '%d' % (np.sum(idx))+
                 ' bootstrap resamples instead of '+'%d' % (Nboot))

        S_lb_sorted = np.sort(S[j][idx, :], axis=0)
        S_lb[j, :] = S_lb_sorted[np.max([0, int(round(np.sum(idx)*alfa/2))-1]), :] # lower bound
        S_ub[j, :] = S_lb_sorted[np.max([0, int(round(np.sum(idx)*(1-alfa/2)))-1]), :] # Upper bound

    if R == 1 or M == 1:
        S_m = S_m.flatten() # shape (M, )
        S_lb = S_lb.flatten() # shape (M, )
        S_ub = S_ub.flatten() # shape (M, )

    return S_m, S_lb, S_ub



def split_sample(Z, n=10):

    """ Split a sample in n equiprobable groups based on the sample values
    (each groups contains approximately the same number of data points).

    This function is called internally in:
        - RSA_groups.RSA_indices_groups to split the output sample
        - PAWN.PAWN_split_sample to split the input sample for each of input
          factor sequentially.

    Usage:
         idx, Zk, Zc, n_eff = util.split_sample(Z, n=10)

    Input:
        Z = sample of a model input  or output       - numpy.ndarray(N,)
                                                  or - numpy.ndarray(N,1)

    Optional input:
        n = number of groups to split the sample

    Output:
      idx = respective groups of the samples         - numpy.ndarray(N, )
            You can easily derive the n groups
            {Zi} as:
                Zi = Z[idx == i]  for i = 1, ..., n
       Zk = groups' edges (range of Z in each group) - numpy.ndarray(n_eff+1, )
       Zc = groups' centers (mean value of Z in each - numpy.ndarray(n_eff, )
            group)
    n_eff = number of groups actually used to split  - scalar
            the sample

    NOTES:
    - When Z is discrete and when the number of values taken by Z (nz) is
      lower than the prescribed number of groups (n), a group is created for
      each value of Z (and therefore the number of groups is set to n_eff = nz).
    - The function ensures that values of Z that are repeated several times
      belong to the same group. This may lead to a number of group n_eff lower
      than n and to having a different number of data points across the groups.

    This function is part of the SAFE Toolbox by F. Pianosi, F. Sarrazin
    and T. Wagener at Bristol University (2015).
    SAFE is provided without any warranty and for non-commercial use only.
    For more details, see the Licence file included in the root directory
    of this distribution.
    For any comment and feedback, or to discuss a Licence agreement for
    commercial use, please contact: francesca.pianosi@bristol.ac.uk
    For details on how to cite SAFE in your publication, please see:
    https://www.safetoolbox.info"""

    ###########################################################################
    # Check inputs
    ###########################################################################

    if not isinstance(Z, np.ndarray):
        raise ValueError('"Z" must be a numpy.array.')
    if Z.dtype.kind != 'f' and Z.dtype.kind != 'i' and Z.dtype.kind != 'u':
        raise ValueError('"Z" must contain floats or integers.')

    Nz = Z.shape
    N = Nz[0]
    if len(Nz) == 2:
        if Nz[1] != 1:
            raise ValueError('"Z" must be of size (N, ) or (N,1).')
        Z = Z.flatten()
    elif len(Nz) != 1:
        raise ValueError('"Z" must be of size (N, ) or (N,1).')

    if not isinstance(n, (int, np.int8, np.int16, np.int32, np.int64)):
        raise ValueError('"n" must be scalar and integer.')
    if n <= 0:
        raise ValueError('"n" must be positive.')

    ###########################################################################
    # Create sub-samples
    ###########################################################################
    n_eff = n

    Zu = np.unique(Z) # district values of Z

    if len(Zu) < n: # if number of distinct values less than the specified
                    # number of groups

        n_eff = len(Zu)
        Zc = np.sort(Zu) # groups' centers are the different values of Xi
        Zk = np.concatenate((Zc, np.array([Zc[-1]]))) # groups' edges

    else:
        # Sort values of Z in ascending order:
        Z_sort = np.sort(Z)
        # Define indices for splitting Z into ngroup equiprobable groups
        # (i.e. with the same number of values):
        split = [int(round(j)) for j in np.linspace(0, N, n_eff+1)]
        split[-1] = N-1
        # Determine the edges of Z in each group:
        Zk = Z_sort[split]

        # Check that values that appear several times in Z belong to the same group:
        idx_keep = np.full((n_eff+1, ), True, dtype=bool)
        for k in range(len(Zk)):
            if np.sum(Zk[k+1:n_eff+1] == Zk[k]) > 1:
                if k < len(Zk)-1:
                    idx_keep[k] = False
        Zk = Zk[idx_keep]
        n_eff = len(Zk) - 1

        Zc = np.mean(np.column_stack((Zk[np.arange(0, n_eff)],
                                      Zk[np.arange(1, n_eff+1)])),
                     axis=1) # centers (average value of each group)

    # Determine the respective groups of the sample:
    idx = -1 * np.ones((N, ), dtype='int8')
    for k in range(n_eff):
        if k < n_eff - 1:
            idx[[j >= Zk[k] and j < Zk[k+1] for j in Z]] = k
        else:
            idx[[j >= Zk[k] and j <= Zk[k+1] for j in Z]] = k

    return idx, Zk, Zc, n_eff


def above(y, par):

    """ This function can be used as input argument ("output_condition") when
    applying PAWN.pawn_indices, PAWN.pawn_convergence, PAWN.pawn_plot_ks """

    idx = y >= par[0]

    return idx

def below(y, par):

    """ This function can be used as input argument ("output_condition") when
    applying PAWN.pawn_indices, PAWN.pawn_convergence, PAWN.pawn_plot_ks """

    idx = y <= par[0]

    return idx

def allrange(y, par):

    """ This function can be used as input argument ("output_condition") when
    applying PAWN.pawn_indices, PAWN.pawn_convergence, PAWN.pawn_plot_ks """

    idx = np.full(y.shape, True, dtype=bool)

    return idx