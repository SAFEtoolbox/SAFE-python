"""
    Module to sample the input space

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

import math
from warnings import warn
import numpy as np
from numpy.matlib import repmat
import scipy.stats as st

from SAFEpython.lhcube import lhcube, lhcube_extend

def AAT_sampling(samp_strat, M, distr_fun, distr_par, N, nrep=5):

    """ Generates a sample X composed of N random samples of M uncorrelated
    variables.

    Usage:
        X = sampling.AAT_sampling(samp_strat, M, distr_fun, distr_par, N, nrep=5)

    Input:
    samp_strat = sampling strategy                                     - string
                 Options: 'rsu': random uniform
                          'lhs': latin hypercube
             M = number of variables                                  - integer
     distr_fun = probability distribution function for ech input
                     - function (eg: 'scipy.stats.uniform') if all
                       variables have the same pdf
                     - list of M functions (e.g.:
                       ['scipy.stats.uniform','scipy.stats.norm']) otherwise
     distr_par = parameters of the probability distribution function
                     - list of parameters if all input variables have the same
                     - list of M lists otherwise
             N = number of samples                                    - integer

    Optional input:
          nrep = number of replicate to select the maximin            - integer
                 hypercube(default value: 5)

    Output:
             X = matrix of samples                        - numpy.ndarray (N,M)
                 Each row is a point in the input space.
                 In contrast to OAT_sampling, rows are not sorted in any
                 specific order, and all elements in a row usually
                 differ from the elements in the following row.

    Supported probability distribution function :

         scipy.stats.beta        (Beta)
         scipy.stats.binom       (Binomial)
         scipy.stats.chi2        (Chisquare)
         scipy.stats.dweibull    (Double Weibull)
         scipy.stats.expon       (Exponential)
         scipy.stats.f           (F)
         scipy.stats.gamma       (Gamma)
         scipy.stats.genextreme  (Generalized Extreme Value)
         scipy.stats.genpareto   (Generalized Pareto)
         scipy.stats.geom        (Geometric)
         scipy.stats.hypergeom   (Hypergeometric)
         scipy.stats.lognorm     (Lognormal)
         scipy.stats.nbinom      (Negative Binomial)
         scipy.stats.ncf         (Noncentral F)
         scipy.stats.nct         (Noncentral t)
         scipy.stats.ncx2        (Noncentral Chi-square)
         scipy.stats.norm        (Normal)
         scipy.stats.poisson     (Poisson)
         scipy.stats.randint     (Discrete Uniform)
         scipy.stats.rayleigh    (Rayleigh)
         scipy.stats.t           (T)
         scipy.stats.uniform     (Uniform)
         scipy.stats.weibull_max (Weibull maximum)
         scipy.stats.weibull_min (Weibull minimum)

    Examples:

    import scipy.stats as st
    import matplotlib.pyplot as plt

    from SAFEpython.sampling import AAT_sampling

    # Example 1: 2 inputs, both from Unif[0,3]
    N = 1000
    M = 2
    distr_fun = st.uniform
    distr_par = [0, 3]
    samp_strat = 'lhs'
    X = AAT_sampling(samp_strat, M, distr_fun, distr_par, N)

    # Plot results:
    plt.figure()
    plt.plot(X[:, 0], X[:, 1], '.k')
    plt.xlabel('x_1')
    plt.ylabel('x_2')

    # Example 2: 2 inputs, one from Unif[0,3], one from Unif[1,5]
    distr_fun = st.uniform
    distr_par = [[0, 3], [1, 4]]
    X = AAT_sampling(samp_strat, M, distr_fun, distr_par, N)
    # (use above code to plot results)

    # Example 3: 2 inputs, one from Unif[0,3], one from discrete, uniform in [1,5]
    distr_fun = [st.uniform, st.randint]
    distr_par = [[0, 3], [1, 6]]
    X = AAT_sampling(samp_strat, M, distr_fun, distr_par, N)

    # Example 4: investigate the difference between 'rsu' and 'lhs':
    N = 20
    X1 = AAT_sampling('rsu', 2, st.uniform, [0, 1], N)
    X2 = AAT_sampling('lhs', 2, st.uniform,[0, 1], N)
    plt.figure()
    plt.subplot(121)
    plt.plot(X1[:, 0],X1[:, 1], 'ok')
    plt.title('Random Uniform')
    plt.subplot(122)
    plt.plot(X2[:, 0],X2[:, 1], 'ok')
    plt.title('Latin Hypercube')

    Note: In sampling.AAT_sampling, the function lhcube.lhcube is used to derive
    latin hypercube sampling (L126). Alternatively, the python package pyDOE
    (pyDOE.lhs) could be used to perform latin hypercube sampling.

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

    if not isinstance(samp_strat, str):
        raise ValueError('"samp_strat" must be a string.')

    if not isinstance(M, (int, np.int8, np.int16, np.int32, np.int64)):
        raise ValueError('"M" must be scalar and integer.')
    if M <= 0:
        raise ValueError('"M" must be positive.')

    if callable(distr_fun):
        distr_fun = [distr_fun] * M
    elif isinstance(distr_fun, list):
        if len(distr_fun) != M:
            raise ValueError('If "distr_fun" is a list, it must have M components.')
    else:
        raise ValueError('"distr_fun" must be a list of functions or a function.')

    if isinstance(distr_par, list):
        if all(isinstance(i, float) or isinstance(i, int) for i in distr_par):
            distr_par = [distr_par] * M
        elif not all(isinstance(i, list) for i in distr_par):
            raise ValueError('"distr_par" be a list of M lists of parameters' +
                             'or a list of parameters is all input have the same.')
    else:
        raise ValueError('Wrong data type for input "distr_par".')

    if not isinstance(N, (int, np.int8, np.int16, np.int32, np.int64)):
        raise ValueError('"N" must be scalar and integer.')
    if N <= 0:
        raise ValueError('"N" must be positive.')

    ###########################################################################
    # Uniformly sample the unit square
    ###########################################################################
    if samp_strat == 'rsu':
        X = np.random.random((N, M))
    elif samp_strat == 'lhs':
        X, _ = lhcube(N, M, nrep)
    else:
        raise ValueError("""Sampling_strategy should be either ''rsu'' or ''lhs'""")

    ###########################################################################
    # Map back into the specified distribution by inverting the CDF
    ###########################################################################
    for i in range(M):

        pars = distr_par[i]
        num_par = len(pars)
        name = distr_fun[i]

        if name in [st.chi2, st.expon, st.geom, st.poisson, st.rayleigh,
                    st.t,st.weibull_max, st.weibull_min]:
            if num_par != 1:
                raise ValueError('Input ' + '%d' % (i+1)+ ': Number of PDF' +
                                 'parameters not consistent with PDF type')
            else:
                X[:, i] = name.ppf(X[:, i], pars)

        elif name in [st.beta, st.binom, st.f, st.gamma, st.lognorm, st.nbinom,
                      st.nct, st.ncx2, st.norm, st.uniform, st.randint]:
            if num_par != 2:
                raise ValueError('Input ' + '%d' % (i+1)+ ': Number of PDF' +
                                 'parameters not consistent with PDF type')
            else:
                X[:, i] = name.ppf(X[:, i], pars[0], pars[1])

        elif name in [st.genextreme, st.genpareto, st.hypergeom, st.ncf]:
            if num_par != 3:
                raise ValueError('Input ' + '%d' % (i+1)+ ': Number of PDF' +
                                 'parameters not consistent with PDF type')
            else:
                X[:, i] = name.ppf(X[:, i], pars[0], pars[1], pars[2])
        else:
            raise ValueError('Input ' + '%d' % (i+1)+ ': Unknown PDF type')

    return X

def AAT_sampling_extend(X, distr_fun, distr_par, Next, nrep=10):

    """This function create an expanded sample 'Xext' starting from a sample
    'X' and using latin hypercube and the maximin criterion.

    Usage:
    Xext = sampling.AAT_sampling_extend(X, distr_fun, distr_par, Next, n_rep=10)

    Input:
            X = initial sample                            - numpy.ndarray(N,M)
    distr_fun = probability distribution function for ech input
                     - string (eg: 'unif') if all variables have the same pdf
                     - list of M strings (e.g.: ['unif','norm']) otherwise
    distr_par = parameters of the probability distribution function
                      - list of parameters if all input variables have the same
                      - list of M lists otherwise
         Next = new dimension of the sample (must be > N)             - integer
         nrep = number of replicate to select the maximin             - integer
                hypercube(default value: 10)

    Output:
          Xext = expanded sample                        - numpy.ndarray(Next,M)

    See also sampling.AAT_sampling.

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
    if not isinstance(X, np.ndarray):
        raise ValueError('"X" must be a numpy.array.')
    if X.dtype.kind != 'f' and X.dtype.kind != 'i' and X.dtype.kind != 'u':
        raise ValueError('"X" must contain floats or integers.')

    Nx = X.shape
    if len(Nx) != 2:
        raise ValueError('"X" must be an array of size (N,M).')
    N = Nx[0]
    M = Nx[1]

    if callable(distr_fun):
        distr_fun = [distr_fun] * M
    elif isinstance(distr_fun, list):
        if len(distr_fun) != M:
            raise ValueError('If "distr_fun" is a list, it must have M components.')
    else:
        raise ValueError('"distr_fun" must be a list of functions or a function.')

    if isinstance(distr_par, list):
        if all(isinstance(i, float) or isinstance(i, int) for i in distr_par):
            distr_par = [distr_par] * M
        elif not all(isinstance(i, list) for i in distr_par):
            raise ValueError('"distr_par" be a list of M lists of parameters' +
                             'or a list of parameters is all input have the same.')
    else:
        raise ValueError('Wrong data type for input "distr_par".')

    if not isinstance(Next, (int, np.int8, np.int16, np.int32, np.int64)):
        raise ValueError('"N" must be scalar and integer.')
    if Next <= N:
        raise ValueError('"Next" must be larger than N.')

    ###########################################################################
    # Check optional inputs
    ##########################################################################
    if not isinstance(nrep, (int, np.int8, np.int16, np.int32, np.int64)):
        raise ValueError('"nrep" must be scalar and integer.')
    if nrep <= 0:
        raise ValueError('"nrep" must be positive.')

    ###########################################################################
    # Map back the original sample into the uniform unit square
    ##########################################################################
    U = np.nan * np.ones((N, M)) # initialization

    for i in range(M):

        pars = distr_par[i]
        num_par = len(pars)
        name = distr_fun[i]


        if name in [st.chi2, st.expon, st.geom, st.poisson, st.rayleigh,
                    st.t,st.weibull_max, st.weibull_min]:
            if num_par != 1:
                raise ValueError('Input ' + '%d' % (i+1)+ ': Number of PDF' +
                                 'parameters not consistent with PDF type')
            else:
                U[:, i] = name.cdf(X[:, i], pars)

        elif name in [st.beta, st.binom, st.f, st.gamma, st.lognorm, st.nbinom,
                      st.nct, st.ncx2, st.norm, st.uniform, st.randint]:
            if num_par != 2:
                raise ValueError('Input ' + '%d' % (i+1)+ ': Number of PDF' +
                                 'parameters not consistent with PDF type')
            else:
                U[:, i] = name.cdf(X[:, i], pars[0], pars[1])

        elif name in [st.genextreme, st.genpareto, st.hypergeom, st.ncf]:
            if num_par != 3:
                raise ValueError('Input ' + '%d' % (i+1)+ ': Number of PDF' +
                                 'parameters not consistent with PDF type')
            else:
                U[:, i] = name.cdf(X[:, i], pars[0], pars[1], pars[2])
        else:
            raise ValueError('Input ' + '%d' % (i+1)+ ': Unknown PDF type')

    ###########################################################################
    # Add samples in the unit square
    ###########################################################################
    Uext = lhcube_extend(U, Next, nrep=nrep)

    ###########################################################################
    # Map back into the specified distribution by inverting the CDF
    ###########################################################################
    Xext = np.nan * np.ones((Next, M)) # initialization

    for i in range(M):

        pars = distr_par[i]
        num_par = len(pars)
        name = distr_fun[i]

        if name in [st.chi2, st.expon, st.geom, st.poisson, st.rayleigh,
                    st.t,st.weibull_max, st.weibull_min]:
            if num_par != 1:
                raise ValueError('Input ' + '%d' % (i+1)+ ': Number of PDF' +
                                 'parameters not consistent with PDF type')
            else:
                Xext[:, i] = name.ppf(Uext[:, i], pars)

        elif name in [st.beta, st.binom, st.f, st.gamma, st.lognorm, st.nbinom,
                      st.nct, st.ncx2, st.norm, st.uniform, st.randint]:
            if num_par != 2:
                raise ValueError('Input ' + '%d' % (i+1)+ ': Number of PDF' +
                                 'parameters not consistent with PDF type')
            else:
                Xext[:, i] = name.ppf(Uext[:, i], pars[0], pars[1])

        elif name in [st.genextreme, st.genpareto, st.hypergeom, st.ncf]:
            if num_par != 3:
                raise ValueError('Input ' + '%d' % (i+1)+ ': Number of PDF' +
                                 'parameters not consistent with PDF type')
            else:
                Xext[:, i] = name.ppf(Uext[:, i], pars[0], pars[1], pars[2])
        else:
            raise ValueError('Input ' + '%d' % (i+1)+ ': Unknown PDF type')

    return Xext


def OAT_sampling(r, M, distr_fun, distr_par, samp_strat, des_type, nrep=5):

    """Build a matrix X of input samples to be used for the Elementary Effects
    Test, using a One-At-the-Time sampling strategy as described in
    Campolongo et al. (2011).

    Usage:
    X = sampling.OAT_sampling(r, M, distr_fun, distr_par, samp_strat,
                              des_type, nrep=5)

    Input:
              r = number of elementary effects                        - integer
              M = number of inputs                                    - integer
      distr_fun = probability distribution function for each input
                  - string (eg: 'unif') if all variables have the same pdf
                  - list of M strings (e.g. '[unif','norm']) otherwise
                  See help of AAT_sampling to check supported PDF types
     distr_par = parameters of the probability distribution function
                 - list of parameters if all input variables have the same
                 - list of M lists otherwise

    samp_strat = sampling strategy                                     - string
                 Options: 'rsu': random uniform
                          'lhs': latin hypercube
      des_type = design type                                           - string
                 Options: 'trajectory','radial'

    Optional input:
          nrep = number of replicate to select the maximin            - integer
                 hypercube(default value: 5)

    Output:
             X = array of sampling datapoints where - numpy.ndarray (r*(M+1),M)
                 EEs must be computed.
                 Each row is a point in the input
                 space. Rows are sorted in 'r'
                 blocks, each including 'M+1' rows.
                 Within each block, points (rows)
                 differ in one component at the
                 time. Thus, each block can be used
                 to compute one Elementary Effect
                 (EE_i) for each model input
                 ( = 1,...,M).

    See also sampling.AAT_sampling for supported probability distribution
    function.

    Examples:

    import scipy.stats as st
    import matplotlib.pyplot as plt
    from SAFEpython.sampling import OAT_sampling

    # Example 1: 2 inputs, both from Unif[0,3]
    r = 10
    M = 2
    distr_fun = st.uniform
    distr_par = [0, 3]
    samp_strat = 'lhs'
    des_type = 'radial'
    X = OAT_sampling(r, M, distr_fun, distr_par, samp_strat, des_type)

    # Plot results:
    clrs = plt.cm.jet(np.linspace(0, 1, r))
    plt.figure()
    j = 0
    for k in range(r): # loop over trajectories
        idx = np.arange(j, j+M+1)
        j = j+M+1
        plt.plot(X[idx, 0], X[idx, 1], 'o:k', markerfacecolor=clrs[k])
    plt.xlabel('x1')
    plt.ylabel('x2')

    # Example 2: 2 inputs, one from Unif[0,3], one from Unif[1,5]
    distr_fun = st.uniform
    distr_par = [[0, 3], [1, 4]]
    X = OAT_sampling(r ,M, distr_fun, distr_par, samp_strat, des_type)
    # (use above code to plot results)

    # Example 3: 2 inputs, one from Unif[0,3], one from discrete, uniform in [1,5]
    distr_fun = [st.uniform, st.randint]
    distr_par = [[0, 3], [1, 6]]
    X  = OAT_sampling(r, M, distr_fun, distr_par, samp_strat, des_type)

    References:

    Campolongo F., Saltelli, A. and J. Cariboni (2011), From screening to
    quantitative sensitivity analysis. A unified approach, Computer Physics
    Communications, 182(4), 978-988.

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
    if not isinstance(r, (int, np.int8, np.int16, np.int32, np.int64)):
        raise ValueError('"r" must be scalar and integer.')
    if r <= 0:
        raise ValueError('"r" must be positive.')

    if not isinstance(M, (int, np.int8, np.int16, np.int32, np.int64)):
        raise ValueError('"M" must be scalar and integer.')
    if M <= 0:
        raise ValueError('"M" must be positive.')

    if not isinstance(des_type, str):
        raise ValueError('"des_type" must be a string.')

    # 'distr_fun', 'distr_par', 'samp_strat' will be checked later by the
    # sampling.AAT_sampling function

    ###########################################################################
    # Perform sampling
    ###########################################################################

    # Sample the array of r baseline and auxiliary points 'AB':
    AB = AAT_sampling(samp_strat, M, distr_fun, distr_par, 2*r, nrep)

    X = np.nan * np.ones((r*(M+1), M)) # initialize sampling points
    k = 0

    for i in range(r): # loop over the elementary effects
    # Sample datapoints:
        a = AB[i*2, :] # baseline point, shape (M, )
        b = AB[i*2+1, :] # auxiliary point, shape (M, )
        for j in range(M): # loop over inputs
            if a[j] == b[j]:
                if distr_fun[j] == st.randint: # resample this component if the
                                             # distribution is random uniform
                    while a[j] == b[j]:
                        tmp = AAT_sampling(samp_strat, M, distr_fun, distr_par, 2)
                        b[j] = tmp[1, j]
                        warn('b[i=%d, j=%d] was randomly changed ' % (i, j) +
                             'not to overlap with a[i=%d, j=%d] \n'% (i, j))
                else: # just print a warning message
                    warn('b[i=%d, j=%d] and a[i=%d, j=%d] are the same! \n'% (i, j, i, j))

        X[k, :] = a
        k = k + 1
        x = repmat(a, M, 1) # shape(M, M)
        if des_type == 'radial':
            for j in range(M):
                x[j, j] = b[j]
                X[k, :] = x[j, :]
                if abs(X[k, j] - X[k-1, j] == 0):
                    warn('X(%d, %d) and X(%d, %d) are equal\n' % (k, j, k-1, j))
                k = k + 1

        elif des_type == 'trajectory':
            for j in range(M):
                x[j, 0:j+1] = b[0:j+1]
                X[k, :] = x[j, :]
                if abs(X[k, j] - X[k-1, j] == 0):
                    warn('X(%d, %d) and X(%d, %d) are equal\n' % (k, j, k-1, j))
                k = k + 1
        else:
            raise ValueError('"des_type" must be one among ["radial", "trajectory"]')

    return X


def Morris_sampling(r, xmin, xmax, L):

    """Build a array X of input samples to be used for the Elementary Effects
    Test, using the One-At-the-Time sampling strategy originally proposed by
    Morris (1991). It implicitely assumes that all the inputs be uncorrelated
    and drawn from a continuous, uniform distribution function.

    Usage:
        X = sampling.Morris_sampling(r, xmin, xmax, L)

    Input:
       r = number of elementary effects             - integer
    xmin = lower bounds of input ranges             - list (M elements)
    xmax = upper bounds of input ranges             - list (M elements)
       L = number of levels in the sampling grid    - integer (even)

    Output:
       X = matrix of sampling datapoints where EE   - numpy.ndarray(r*(M+1),M))
           must becomputed. Each row of X is a
           point in the input space. Rows are sorted
           in 'r' blocks, each including 'M+1' rows.
           Within each block, points (rows) differ
           in one component at the time. Thus, each
           block can be used to compute one
           Elementary Effect (EE_i) for each model
           input (i=1,...,M).

    References:

    Morris, M.D. (1991), Factorial sampling plans for preliminary
    computational experiments, Technometrics, 33(2).

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

    if not isinstance(r, (int, np.int8, np.int16, np.int32, np.int64)):
        raise ValueError('"r" must be scalar and integer.')
    if r <= 0:
        raise ValueError('"r" must be positive.')

    if not isinstance(xmin, list):
        raise ValueError('"xmin" must be a list with M elements')
    if not all(isinstance(i, float) or isinstance(i, int) for i in xmin):
        raise ValueError('Elements in "xmin" must be int or float.')
    if not isinstance(xmax, list):
        raise ValueError('"xmin" must be a list with M elements')
    if not all(isinstance(i, float) or isinstance(i, int) for i in xmax):
        raise ValueError('Elements in "xmax" must be int or float.')
    M = len(xmin)
    if len(xmax) != M:
        raise ValueError('"xmin" and "xmax" must have the same number of elements.')
    xmin = np.array(xmin)
    xmax = np.array(xmax)
    Dr = xmax - xmin
    if not all(i >= 0 for i in Dr):
        raise ValueError('all components of "xmax" must be higher than'+
                         'the corresponding ones in "xmin"')

    if not isinstance(L, (int, np.int8, np.int16, np.int32, np.int64)):
        raise ValueError('"L" must be scalar and integer.')
    if L <= 0:
        raise ValueError('"L" must be positive.')
    if L % 2 != 0:
        L = math.ceil(L/2)*2
        warn('"L" must be even!\n Using L = %d instead of user-defined value\n' % L)

    ###########################################################################
    # Perform sampling
    ###########################################################################

    X = np.nan * np.ones((r*(M+1), M)) # sampling points

    for i in range(r): # loop on elementary effects
        # Sample datapoints in the unit square:
        Bstar = Morris_orientation_matrix(M, L) #  shape (M+1,M)
        # Resort to original ranges:
        Bstar = repmat(xmin, M+1, 1) + Bstar*repmat(Dr, M+1, 1)
        X[i*(M+1):(i+1)*(M+1), :] = Bstar

    return X

def Morris_orientation_matrix(k, p):

    """
    Usage:
        Bstar = sampling.Morris_orientation_matrix(k, p)

        k = number of inputs                             - integer
        p = number of levels                             - integer (even)
    Bstar = array of (k+1) datapoints in the             - numpy.ndarray(k+1,k)
            k-dimensional input space
            (to be used for computing one Elementary
            Effect for each of the 'k' inputs)

    # Example in two-dimensional space (k=2):

    import numpy as np
    import matplotlib.pyplot as plt
    from SAFEpython.sampling import Morris_orientation_matrix

    p = 4
    k = 2
    plt.figure()
    Bstar = Morris_orientation_matrix(k, p)
    plt. plot(Bstar[:, 0], Bstar[:, 1], '.r-')
    # if you want to generate more datapoints:
    Bstar = Morris_orientation_matrix(k, p)
    plt. plot(Bstar[:, 0], Bstar[:, 1], 'xb-')
    Bstar = Morris_orientation_matrix(k, p)
    plt. plot(Bstar[:, 0], Bstar[:, 1], 'oc-')
    Bstar = Morris_orientation_matrix(k, p)
    plt. plot(Bstar[:, 0], Bstar[:, 1], 'sm-')
    plt.xticks(np.arange(0, 1+1/(p-1), 1/(p-1)))
    plt.yticks(np.arange(0, 1+1/(p-1), 1/(p-1)))
    plt.xlim((-0.1, 1.1))
    plt.ylim((-0.1, 1.1))
    plt.grid(b=True)
    """

    ###########################################################################
    # Check inputs
    ###########################################################################
    if not isinstance(k, (int, np.int8, np.int16, np.int32, np.int64)):
        raise ValueError('"k" must be scalar and integer.')
    if k <= 0:
        raise ValueError('"k" must be positive.')

    if not isinstance(p, (int, np.int8, np.int16, np.int32, np.int64)):
        raise ValueError('"p" must be scalar and integer.')
    if p <= 0:
        raise ValueError('"p" must be positive.')
    if p % 2 != 0:
        p = math.ceil(p/2)*2
        warn('"p" must be even!\n Using p = %d instead of user-defined value\n' % p)

    ###########################################################################
    # Determine orientation matrix
    ###########################################################################

    Delta = p/(2*(p-1))
    # sampling matrix:
    B = np.tril(np.ones((k+1, k)), -1) # shape (k+1,k)

    # Create diagonal matrix with [-1,1] elements
    tmp = np.random.randint(0, 2, size=(k,)) # random numbers from [0,1]
    tmp[tmp == 0] = -1 # random numbers from [-1,1]
    D = np.diag(tmp)

    # Create base value vector
    In = np.nan * np.ones((1, int(p/2)))
    for i in range(int(p/2)):
        In[0, i] = i/(p-1)

    tmp = np.random.randint(len(In[0, :]), size=(k,))
    x = In[:, tmp]

    # Create random permutation matrix
    P = np.eye(k)
    idx = np.random.choice(k, size=(k,), replace=False)
    P = P[idx, :]

    Jmk = np.ones((k+1, k))
    Jm1 = np.ones((k+1, 1))

    # Create a random orientation of B:
    #Bstar = (Jm1@x + Delta/2*((2*B-Jmk)@D + Jmk))@P # does not work for python 2
    Bstar = np.matmul(np.matmul(Jm1, x) + Delta/2*(np.matmul(2*B-Jmk,D) + Jmk), P)

    return Bstar

def OAT_sampling_extend(X, r_ext, distr_fun, distr_par, des_type, nrep=10):

    """This function create an expanded sample 'Xext' starting from a sample
    'X' to be used for the Elementary Effects Test. One-At-the-Time sampling
    strategy as described in Campolongo et al.(2011) is used. The matrix of
    baseline and auxiliary points is built using latin hypercube and the
    maximin criterion.

    Usage:

    Xext, Xnew = \
     sampling.OAT_sampling_extend(X,r_ext,M,distr_fun,distr_par,des_type)
    Xext, Xnew = \
     sampling.OAT_sampling_extend(X,r_ext,M,distr_fun,distr_par,des_type,n_rep)

      Input:
              X = initial sample build as described - numpy.ndarray (r*(M+1),M)
                 in OAT_sampling.
          r_ext = new number of elementary effects                    - integer
      distr_fun = probability distribution function for each input
                  - string (eg: 'unif') if all variables have the same pdf
                  - list of M strings (e.g. '[unif','norm']) otherwise
                  See help of AAT_sampling to check supported PDF types
     distr_par = parameters of the probability distribution function
                 - list of parameters if all input variables have the same
                 - list of M lists otherwise
      des_type = design type                                           - string
                 Options: 'trajectory','radial'

    Optional input:
          nrep = number of replicate to select the maximin hypercube  - integer
                 (default value: 10)

    Output:
         Xext = expanded sample                 - numpy.ndarray (r_ext*(M+1),M)
         Xnew = newly added samples             - numpy.ndarray (r_new*(M+1),M)
                (where    Xext = [ X; Xnew ]
                          r_ext = r + r_new )

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
    if not isinstance(X, np.ndarray):
        raise ValueError('"X" must be a numpy.array.')
    if X.dtype.kind != 'f' and X.dtype.kind != 'i' and X.dtype.kind != 'u':
        raise ValueError('"X" must contain floats or integers.')

    Nx = X.shape
    if len(Nx) != 2:
        raise ValueError('"X" must be an array of size (N,M).')
    N = Nx[0]
    M = Nx[1]

    if N % (M+1) != 0:
        raise ValueError('"X" must have r*(M+1) lines and M columns')
    r = int(N/(M+1))

    if not isinstance(r_ext, (int, np.int8, np.int16, np.int32, np.int64)):
        raise ValueError('"r_ext" must be scalar and integer.')
    if r_ext <= 0:
        raise ValueError('"r_ext" must be positive.')

    if not isinstance(des_type, str):
        raise ValueError('"des_type" must be a string.')

    # 'distr_fun', 'distr_par', 'samp_strat', 'r_ext' and 'n_rep' will be
    # checked later by the AAT_sampling function

    ###########################################################################
    # Compute the new matrix of baseline and auxiliary points
    ###########################################################################
    ABold = np.nan * np.ones((2*r, M)) # old matrix of baseline and auxiliary points in sample X
    if des_type == 'radial':
        for i in range(r):
            ABold[2*i, :] = X[i*(M+1), :]
            for j in range(M):
                ABold[2*i+1, j] = X[i*(M+1)+j+1, j]

    elif des_type == 'trajectory':
        for i in range(r):
            ABold[2*i, :] = X[i*(M+1), :]
            ABold[2*i+1, :] = X[(i+1)*(M+1)-1, :]
    else:
        raise ValueError('"des_type" must be one among ["radial","trajectory"]')

    ABext = AAT_sampling_extend(ABold, distr_fun, distr_par, 2*r_ext, nrep)
    # new matrix of baseline and auxiliary points for extended sample Xext

    ABnew = ABext[2*r:2*r_ext, :]
    r_new = r_ext-r

    ###########################################################################
    # Add the intermediate points to the extension of the sample
    ###########################################################################
    Xnew = np.nan * np.ones((r_new*(M+1), M)) # initialize sampling points
    k = 0

    for i in range(r_new): # loop over the elementary effects
    # Sample datapoints:
        a = ABnew[i*2, :] # baseline point, shape (M, )
        b = ABnew[i*2+1, :] # auxiliary point, shape (M, )
        for j in range(M): # loop over inputs
            if a[j] == b[j]:
                if distr_fun[j] == 'randint': # resample this component if the
                                             # distribution is random uniform
                    while a[j] == b[j]:
                        tmp = AAT_sampling('lhs', M, distr_fun, distr_par, 2)
                        b[j] = tmp[1, j]
                        warn('b[i=%d, j=%d] was randomly changed ' % (i, j) +
                             'not to overlap with a[i=%d, j=%d] \n'% (i, j))
                else: # just print a warning message
                    warn('b[i=%d, j=%d] and a[i=%d, j=%d] are the same! \n'% (i, j, i, j))

        Xnew[k, :] = a
        k = k + 1
        x = repmat(a, M, 1) # shape(M, M)
        if des_type == 'radial':
            for j in range(M):
                x[j, j] = b[j]
                Xnew[k, :] = x[j, :]
                if abs(Xnew[k, j] - Xnew[k-1, j] == 0):
                    warn('Xnew(%d, %d) and Xnew(%d, %d) are equal\n' % (k, j, k-1, j))
                k = k + 1

        elif des_type == 'trajectory':
            for j in range(M):
                x[j, 0:j+1] = b[0:j+1]
                Xnew[k, :] = x[j, :]
                if abs(Xnew[k, j] - Xnew[k-1, j] == 0):
                    warn('Xnew(%d, %d) and Xnew(%d, %d) are equal\n' % (k, j, k-1, j))
                k = k + 1

    Xext = np.concatenate((X, Xnew), axis=0)

    return Xext, Xnew
