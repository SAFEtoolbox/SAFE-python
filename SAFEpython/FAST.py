"""
    Module to perform the Fourier Amplitude Sensitivity Test (FAST)

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

    References:

    Cukier, R.I., Levine, H.B., and Shuler, K.E. (1978), Nonlinear
    Sensitivity Analyis of Multiparameter Model SYstems, Journal of
    Computational Physics, 16, 1-42.

    Saltelli, A., Tarantola, S. and Chan, K.P.S. (1999), A Quantitative
    Model-Independent Method for Global Sensitivty Analysis of Model Output,
    Technometrics, 41(1), 39-56.
"""

from __future__ import division, absolute_import, print_function

from warnings import warn
import numpy as np
import scipy.stats as st
from numba import jit

def FAST_sampling(distr_fun, distr_par, M, N=[], Nharm=4, omega=[]):

    """Implements sampling for the Fourier Amplitude Sensitivity Test (FAST;
    Cukier et al., 1978) and returns a matrix 'X' of N input samples.
    (See also FAST_sampling_unif.m for details and references about FAST
    sampling strategy)

    Usage:
    X, s = FAST.FAST_sampling(distr_fun, distr_par, M, n=[], Nharm=4, omega=[])

    Generates a sample X composed of N random samples of M uncorrelated
    variables.

    Input:
     distr_fun = probability distribution function for ech input
                     - function (eg: scipy.stats.uniform') if all variables
                       have the same pdf
                     - list of M funtions(e.g.:
                       ['scipy.stats.uniform','scipy.stats.norm']) otherwise
     distr_par = parameters of the probability distribution function
                     - list of parameters if all input variables have the same
                     - list of M lists otherwise
             M = number of inputs                           - integer

     Optional input:
             N = number of samples (default is              - integer
                 2*Nharm*max(omega)+1 which is the minimum     (odd)
                 sampling size according to Cukier et al.
                 (1978))

         Nharm = interference factor, i.e. the number of    - integer
                 higher harmonics to be considered
                 (default is 4)
         omega = angular frequencies associated to inputs   - numpy.ndarray(M,)
                 (default values computed by function
                 'FAST.generate_FAST_frequency.m')

    Output:
        X = array of input samples                         - numpy.ndarray(N,M)
        s = vector of sampled points over the search curve - numpy.ndarray(N,)

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

    if not isinstance(M, int) and isinstance(M, np.int8) and \
    isinstance(M, np.int16) and isinstance(M, np.int32) and \
    isinstance(M, np.int64):
        raise ValueError('"M" must be scalar and integer.')
    if M < 0:
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

    ###########################################################################
    # Recover and check optional inputs
    ###########################################################################
    if not isinstance(Nharm, (int, np.int8, np.int16, np.int32, np.int64)):
        raise ValueError('"Nharm" must be scalar and integer.')
    if Nharm <= 0:
        raise ValueError('"Nharm" must be positive.')

    if len(omega) == 0:
        omega = generate_FAST_frequency(M)
    else:
        if not isinstance(omega, np.ndarray):
            raise ValueError('"omega" must be a numpy.array.')
        if omega.dtype.kind != 'i':
            raise ValueError('"omega" must contain integers.')
        No = omega.shape
        if len(No) > 1:
            raise ValueError('"omega" must be of shape (M, ).')
        if No[0] != M:
            raise ValueError('"omega" must be of shape (M, ).')
        if any([i < 0 for i in omega]):
            raise ValueError('"omega" must contain positive integers.')

    if N == []:
        N = 2*Nharm*np.max(omega)+1
    else:
        if not isinstance(N, (int, np.int8, np.int16, np.int32, np.int64)):
            raise ValueError('"N" must be scalar and integer.')
        if N <= 0:
            raise ValueError('"N" must be positive.')
        if N % 2 == 0:
            raise ValueError('"N" must be odd.')
        if N < 2*Nharm*np.max(omega)+1:
            Nuser = N
            N = 2*Nharm*np.max(omega)+1
            warn('Sample size specified by user (%d) is smaller ' % (Nuser) +
                 'than minimum sample size. Using the latter (%d) instead' % (N))
    ###########################################################################
    # Uniformly sample the unit square using FAST sampling
    ###########################################################################
    X, s = FAST_sampling_unif(M, N, Nharm, omega)

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

    return X, s


def FAST_sampling_unif(M, N=[], Nharm=4, omega=[]):

    """Implements sampling for the Fourier Amplitude Sensitivity Test (FAST;
    Cukier et al., 1978) and returns a matrix 'X' of N input samples.
    Inputs are assumed to be uniformly distributed in the unit hypercube
    [0,1]^M.
    Samples are taken along the search curve defined by transformations

         x_i(s) = G_i( sin( omega_i*s ) )      i=1,...,M         (*)

    where 's' is a scalar variable that varies in (-pi/2,pi/2)

    Usage:
        X, s = FAST.FAST_sampling_unif(M, N=[], Nharm=4, omega=[])

    Input:
        M = number of inputs                               - integer

    Optional input:
        N = number of samples (default is                  - integer
           2*Nharm*max(omega)+1 which is the minimum         (odd)
           sampling size according to Cukier et al. (1978))
    Nharm = interference factor, i.e. the number of higher - integer
            harmonics to be considered (default is 4,
            taken from  Saltelli et al. (1999; page 42))
    omega = angular frequencies associated to inputs       - numpy.ndarray(M, )
            (default values computed by function
            'FAST.generate_FAST_frequency.m')

    Output:
        X = matrix of input samples                        - numpy.ndarray(N,M)
        s = vector of sampled points over the search curve - numpy.ndarray(M, )

    Notes:
    (*)  Here we use the curve proposed by Saltelli et al. (1999):
             x_i = 1/2 + 1/pi * arcsin( sin( omega_i*s ) )

    References:

    Cukier, R.I., Levine, H.B., and Shuler, K.E. (1978), Nonlinear
    Sensitivity Analyis of Multiparameter Model SYstems, Journal of
    Computational Physics, 16, 1-42.

    Saltelli, A., Tarantola, S. and Chan, K.P.S. (1999), A Quantitative
    Model-Independent Method for Global Sensitivty Analysis of Model Output,
    Technometrics, 41(1), 39-56.

    This function is part of the SAFE Toolbox by F. Pianosi, F. Sarrazin
    and T. Wagener at Bristol University (2015).
    SAFE is provided without any warranty and for non-commercial use only.
    For more details, see the Licence file included in the root directory
    of this distribution.
    For any comment and feedback, or to discuss a Licence agreement for
    commercial use, please contact: francesca.pianosi@bristol.ac.uk
    For details on how to cite SAFE in your publication, please see:
    https://www.safetoolbox.info/"""

    ###########################################################################
    # Check inputs
    ###########################################################################

    if not isinstance(M, (int, np.int8, np.int16, np.int32, np.int64)):
        raise ValueError('"M" must be scalar and integer.')

    ###########################################################################
    # Recover and check optional inputs
    ###########################################################################

    if not isinstance(Nharm, (int, np.int8, np.int16, np.int32, np.int64)):
        raise ValueError('"Nharm" must be scalar and integer.')
    if Nharm <= 0:
        raise ValueError('"Nharm" must be positive.')

    if len(omega) == 0:
        omega = generate_FAST_frequency(M)
    else:
        if not isinstance(omega, np.ndarray):
            raise ValueError('"omega" must be a numpy.array.')
        if omega.dtype.kind != 'i':
            raise ValueError('"omega" must contain integers.')
        No = omega.shape
        if len(No) > 1:
            raise ValueError('"omega" must be of shape (M, ).')
        if No[0] != M:
            raise ValueError('"omega" must be of shape (M, ).')
        if any([i < 0 for i in omega]):
            raise ValueError('"omega" must contain positive integers.')

    if N == []:
        N = 2*Nharm*np.max(omega)+1
    else:
        if not isinstance(N, (int, np.int8, np.int16, np.int32, np.int64)):
            raise ValueError('"N" must be scalar and integer.')
        if N <= 0:
            raise ValueError('"N" must be positive.')
        if N % 2 == 0:
            raise ValueError('"N" must be odd.')
        if N < 2*Nharm*np.max(omega)+1:
            Nuser = N
            N = 2*Nharm*np.max(omega)+1
            warn('Sample size specified by user (%d) is smaller ' % (Nuser) +
                 'than minimum sample size. Using the latter (%d) instead' % (N))

    ###########################################################################
    # Perform sampling over the search curve
    ###########################################################################

    s = np.pi/2*(2*np.arange(1, N+1)-N-1)/N

    ###########################################################################
    # Map back sampled points in the input space
    ###########################################################################

    X = np.nan * np.zeros((N, M))
    for n in range(N):
        X[n, :] = 1/2 + 1/np.pi*np.arcsin(np.sin(omega*s[n]))

    return X, s

def generate_FAST_frequency(M):

    """ Generates a sequence of M frequency values (omega) for sampling
    according to the FAST method
    (See also FAST.FAST_sampling_unif for details and references about FAST
    sampling strategy)

    Usage:
        omega = FAST.generate_FAST_frequency(M)

    Input:
        M = number of inputs (values between 4 and 50)    - integer

    Output:
    omega = frequency set free of interferences through   - numpy.ndarray (M, )
            (at least) 4th order

    Note:

    - For M > 4, frequencies are computed based on the recursive algorithm by:

    Cukier et al. (1975) Study of the sensitivity of coupled reaction systems
    to uncertainties in rate coefficients. III. Analysis of the approximations,
    J. Chem. Phys. 63, 1140

    which is free of interferences through the 4th order.

    - For M <= 4, we use values from the literature that guarantee higher order
    interferences free (see comments in the code for specific references)

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

    if not isinstance(M, (int, np.int8, np.int16, np.int32, np.int64)):
        raise ValueError('"M" must be scalar and integer.')
    if M < 2:
        raise ValueError('"M" must be  >=2.')

    ###########################################################################
    # Generate frequency
    ###########################################################################

    if M == 2: #  Use values from Sec. 3.1 in:
    # Xu, C. and G. Gertner (2007), Extending a global sensitivity analysis
    # technique to models with correlated parameters, Computational Statistics
    # and Data Analysis, 51, 5579-5590.
        omega = np.array([5, 23])
        # (free of interference through 10th order)
    elif M == 4: # Use values from Table III in Cukier et al. (1975)
        # (free of interferences through 6th order)
        omega = np.array([13, 31, 37, 41])
    else: # Use recursive algorithm in the same paper
        Omega = np.array([0, 0, 1, 5, 11, 1, 17, 23, 19, 25, 41, 31, 23, 87, 67,
                          73, 85, 143, 149, 99, 119, 237, 267, 283, 151, 385,
                          157, 215, 449, 163, 337, 253, 375, 441, 673, 773, 875,
                          873, 587, 849, 623, 637, 891, 943, 1171, 1225, 1335,
                          1725, 1663, 2019])
        d = np.array([4, 8, 6, 10, 20, 22, 32, 40, 38, 26, 56, 62, 46, 76, 96,
                      60, 86, 126, 134, 112, 92, 128, 154, 196, 34, 416, 106,
                      208, 328, 198, 382, 88, 348, 186, 140, 170, 284, 568, 302,
                      438, 410, 248, 448, 388, 596, 216, 100, 488, 166])
        # above values taken from Table VI
        omega = np.zeros((M,), dtype=int)
        omega[0] = Omega[M-1]
        for i in range(1, M):
            omega[i] = omega[i-1] + d[M-1-i]
            # equation (5.1)

    return omega

@jit
def FAST_indices(Y, M, Nharm=4, omega=[]):

    """Computes main effect (first-order) sensitivity index according to the
    Fourier Amplitude Sensitivity Test (FAST)
    (Cukier et al., 1978; Saltelli et al., 1999)

    Usage:
        Si, V, A, B, Vi = FAST.FAST_indices(Y, M, Nharm=4, omega=[])

    Input:
        Y = set of model output samples                    - numpy.ndarray(N, )
            Note: the number of elements of Y is odd    or - numpy.ndarray(N,1)
        M = number of inputs                               - integer
    Nharm = interference factor, i.e.the number of higher  - integer
            harmonics to be considered (default is 4)
    omega = angular frequencies associated to inputs       - numpy.ndarray(M, )
            (default values computed by function
            'FAST.generate_FAST_frequency.m')

    Output:
       Si = main effect (first-order) sensitivity indices  - numpy.ndarray(M, )
        V = total output variance                          - float
        A = Fourier coefficients                           - numpy.ndarray(N, )
        B = Fourier coefficients                           - numpy.ndarray(N, )
       Vi = output variances from each input               - numpy.ndarray(M, )

    References:

    Cukier, R.I., Levine, H.B., and Shuler, K.E. (1978), Nonlinear
    Sensitivity Analyis of Multiparameter Model SYstems, Journal of
    Computational Physics, 16, 1-42.

    Saltelli, A., Tarantola, S. and Chan, K.P.S. (1999), A Quantitative
    Model-Independent Method ofr Global Sensitivty Analysis of Model Output,
    Technometrics, 41(1), 39-56.

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
    if not isinstance(Y, np.ndarray):
        raise ValueError('"Y" must be a numpy.array.')
    if Y.dtype.kind != 'f' and Y.dtype.kind != 'i' and Y.dtype.kind != 'u':
        raise ValueError('"Y" must contain floats or integers.')
    Ny = Y.shape

    if len(Ny) == 2:
        if Ny[1] != 1:
            raise ValueError('"Y" must be of size (N, ) or (N,1).')
        Y = Y.flatten()
    elif len(Ny) != 1:
        raise ValueError('"Y" must be of size (N, ) or (N,1).')
    N = Ny[0]

    if not isinstance(M, (int, np.int8, np.int16, np.int32, np.int64)):
        raise ValueError('"M" must be scalar and integer.')
    if M < 0:
        raise ValueError('"M" must be positive.')

    ###########################################################################
    # Recover and check optional inputs
    ###########################################################################
    if not isinstance(Nharm, (int, np.int8, np.int16, np.int32, np.int64)):
        raise ValueError('"Nharm" must be scalar and integer.')
    if Nharm <= 0:
        raise ValueError('"Nharm" must be positive.')

    if len(omega) == 0:
        omega = generate_FAST_frequency(M)
    else:
        if not isinstance(omega, np.ndarray):
            raise ValueError('"omega" must be a numpy.array.')
        if omega.dtype.kind != 'i':
            raise ValueError('"omega" must contain integers.')
        No = omega.shape
        if len(No) > 1:
            raise ValueError('"omega" must be of shape (M, ).')
        if No[0] != M:
            raise ValueError('"omega" must be of shape (M, ).')
        if any([i < 0 for i in omega]):
            raise ValueError('"omega" must contain positive integers.')

    if N < 2*Nharm*np.max(omega)+1: # and finally check that is is consistent with omega
        raise ValueError('Sample size (i.e. the length of vector Y) is %d, ' % (N) +
                         'which is lower than the minimum sample size' +
                         '(i.e. 2*Nharm*max(omega)+1=2*' +
                         '%d*%d+1=%d)' % (Nharm, np.max(omega), 2*Nharm*np.max(omega)+1))

    ###########################################################################
    # Compute Fourier coefficients (vectors A and B) from Y
    ###########################################################################

    # The code below implements the equations given in Appendix C of
    # Saltelli et al. (1999).

    A = np.zeros((N,))
    B = np.zeros((N,))

    baseplus = np.sum(np.reshape(Y[1:], (int((N-1)/2), 2)), axis=1) # shape ((N-1)/2,1)
    baseminus = -np.diff(np.reshape(Y[1:], (int((N-1)/2), 2)), axis=1).flatten() # shape ((N-1)/2,1)

    for j in range(N):
        if (j+1) % 2 == 0: # j+1 is even
            sp = Y[0]
            for k in range(int((N-1)/2)):
                sp = sp + baseplus[k]*np.cos((j+1)*(k+1)*np.pi/N)
            A[j] = sp/N
        else: # j is odd
            sp = 0
            for k in range(int((N-1)/2)):
                sp = sp + baseminus[k]*np.sin((j+1)*(k+1)*np.pi/N)
            B[j] = sp/N

    ###########################################################################
    # Compute main effect from A and B
    ###########################################################################
    # The code below implements the equations given in Appendix B of
    # Saltelli et al. (1999) (here we use 'V' and 'Vi' for the output variances
    # while in that paper they are called 'D' and 'Di')

    V = 2*np.sum(A**2+B**2) # total output variance
    # use 'numpy.sum' instead of 'sum' to spead up the code

    Vi = np.nan * np.zeros((M,)) #  output variances from the i-th input
    for i in range(M):
         idx = np.arange(1, Nharm+1)*omega[i]
         Vi[i] = 2*np.sum(A[idx-1]**2+B[idx-1]**2)

    Si = Vi / V

    return Si, V, A, B, Vi
