"""
    Module to perform PAWN

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

    Pianosi, F. and Wagener, T. (2018), Distribution-based sensitivity
    analysis from a generic input-output sample, Env. Mod. & Soft., 108, 197-207.

    Pianosi, F. and Wagener, T. (2015), A simple and efficient method
    for global sensitivity analysis based on cumulative distribution
    functions, Env. Mod. & Soft., 67, 1-11.
"""
from __future__ import division, absolute_import, print_function

from warnings import warn
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from SAFEpython.util import empiricalcdf, split_sample
from SAFEpython.lhcube import lhcube_shrink

from SAFEpython.util import allrange, above, below

def pawn_split_sample(X, Y, n=10):

    """This functions splits a generic input-output dataset to create the
    conditional samples for the approximation of PAWN sensitivity indices
    (Pianosi and Wagener, 2018).
    This function extends the splitting strategy described in
    (Pianosi and Wagener, 2018), which consists of splitting the input and
    output sample into a number of equally spaced conditioning intervals
    (intervals with the same size) based on the value of each input Xi.
    Here, the conditioning intervals are equiprobable, i.e. there is
    (approximately) the same number of data points in each interval. While
    equally spaced intervals are adapted for inputs that have a uniform
    distribution, equiprobable intervals allow to handle inputs that have any
    distribution.

    This function is called internally in PAWN.pawn_indices, PAWN.pawn_plot_cdf.

    Usage:
        YY, xc, NC, n_eff, Xk, XX = PAWN.pawn_split_sample(X, Y, n=10)

    Input:
        X = set of inputs samples                          - numpy.ndarray(N,M)
        Y = set of output samples                          - numpy.ndarray(N,)
                                                        or - numpy.ndarray(N,1)

    Optional input:
        n = number of conditioning intervals
            - integer if all inputs have the same number of groups
            - list of M integers otherwise
            (default value: 10)

    Output:
       YY = output samples to assess the conditional CDFs. - list (M elements)
            YY[i] is a list of n_eff[i] subsamples that
            can be used to assess n_eff[i] conditional
            output distributions with respect to the i-th
            input.
            YY[i][k] is obtained by fixing the i-th input
            to its k-th conditioning interval (while the
            other inputs vary freely), and it is a
            np.ndarray of shape (NC[i][k], ).
       xc = subsamples' centers for each input Xi (i.e.    - list (M elements)
            mean value of Xi over each conditioning
            interval).
            xc[i] is a np.ndarray of shape (n_eff[i],) and
            contains the subsamples' centers for the i-th
            input Xi.
       NC = number of data points in each conditioning     - list (M elements)
            interval and for each input.
            NC[i] is a np.ndarray of shape (n_eff[i],) and
            contains the sample sizes for the i-th input.
    n_eff = number of conditioning intervals actually used - list (M elements)
            for each inputs Xi (see (*) for further
            explanation).
       Xk = subsamples' edges for each input Xi (i.e.      - list (M elements)
            bounds of Xi over each conditioning interval)
            Xk[i] is a np.ndarray of shape (n_eff[i]+1,)
            and contains the edges of the conditioning
            intervals for the i-th input.
       XX = conditional input samples corresponding to     - list (M elements)
            the samples in YY.
            XX[i] is a list of n_eff[i] subsamples for the
            i-th input.
            XX[i][k] is obtained by fixing the i-th input
            to its k-th conditioning interval (while the
            other inputs vary freely), and it is a
            np.ndarray of shape (NC[i][k],M)

    (*) NOTES:
    - When Xi is discrete and when the number of values taken by Xi (nxi) is
      lower than the prescribed number of conditioning intervals (n[i]), a
      conditioning interval is created for each value of Xi (and therefore the
      actual number of conditioning intervals is set to n_eff[i] = nxi).
    - The function ensures that values of Xi that are repeated several times
      belong to the same group. This may lead to a number of conditioning
      intervals n_eff[i] lower than the prescribed value n[i] and to a
      different number of data points between the groups.

    REFERENCES

    Pianosi, F. and Wagener, T. (2018), Distribution-based sensitivity
    analysis from a generic input-output sample, Env. Mod. & Soft., 108, 197-207.

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

    if not isinstance(Y, np.ndarray):
        raise ValueError('"Y" must be a numpy.array.')
    if Y.dtype.kind != 'f' and Y.dtype.kind != 'i' and Y.dtype.kind != 'u':
        raise ValueError('"Y" must contain floats or integers.')

    Nx = X.shape
    Ny = Y.shape
    if len(Nx) != 2:
        raise ValueError('input "X" must have shape (N,M)')
    N = Nx[0]
    M = Nx[1]

    if len(Ny) == 2:
        if Ny[1] != 1:
            raise ValueError('"Y" must be of size (N, ) or (N,1).')
        Y = Y.flatten()
    elif len(Ny) != 1:
        raise ValueError('"Y" must be of size (N, ) or (N,1).')
    if Ny[0] != N:
        raise ValueError('the number of elements in "Y" must be the same as'+
                         'the number rows in X.')

    ###########################################################################
    # Check inputs
    ###########################################################################
    if isinstance(n, (int, np.int8, np.int16, np.int32, np.int64)):
        n = [n] * M
    elif isinstance(n, list):
        if len(n) != M:
            raise ValueError('If "n" is a list, it must have M components.')
        if not all(i > 0 and isinstance(i, (int, np.int8, np.int16,
                                            np.int32, np.int64)) for i in n):
            raise ValueError('Elements in "n" must be strictly positive integers.')
    else:
        raise ValueError('Wrong data type: "n" must be an integer of a list' +
                         'of M integers.')

    ###########################################################################
    # Create sub-samples
    ###########################################################################

    # Intialise variables
    YY = [np.nan] * M
    XX = [np.nan] * M
    xc = [np.nan] * M
    NC = [np.nan] * M
    n_eff = [np.nan] * M
    Xk = [np.nan] * M

    for i in range(M): # loop over the inputs

        idx, Xk[i], xc[i], n_eff[i] = split_sample(X[:, i], n[i])
        # "idx" contains the indices for each group for the i-th input

        XX[i] = [np.nan] * n_eff[i] # conditioning samples for i-th input
        YY[i] = [np.nan] * n_eff[i]
        NC[i] = np.nan * np.ones((n_eff[i], )) # shapes of the conditioning
        # samples for i-th input

        for k in range(n_eff[i]): # loop over the conditioning intervals
            idxk = idx == k # indices of the k-th conditioning interval
            XX[i][k] = X[idxk, :]
            YY[i][k] = Y[idxk]
            NC[i][k] = np.sum(idxk)

        # Print a warning if the number of groups that were used is lower than
        # the prescribed number of groups:
        if n_eff[i] < n[i]:
            warn("For X%d, %d groups were used instead of " % (i+1, n_eff[i]) +
                 "%d so that values that are repeated several time " % (n[i]) +
                 "belong to the same group")

    return YY, xc, NC, n_eff, Xk, XX


def pawn_indices(X, Y, n, Nboot=1, dummy=False, output_condition=allrange,
                 par=[]):

    """  Compute the PAWN sensitivity indices. The method was first introduced
    in Pianosi and Wagener (2015). Here indices are computed following the
    approximation strategy proposed by Pianosi and Wagener (2018), which can be
    applied to a generic input/output sample.

    The function splits the generic output sample to create the conditional
    output by calling internally the function PAWN.pawn_split_sample. The
    splitting strategy is an extension of the strategy for uniformy distributed
    inputs described in Pianosi and Wagener (2018) to handle inputs sampled
    from any distribution(see help of PAWN.pawn_split_sample for further explanation).

    Indices are then computed in two steps:
    1. compute the Kolmogorov-Smirnov (KS) statistic between the empirical
    unconditional CDF and the conditional CDFs for different conditioning
    intervals
    2. take a statistic (median, mean and max) of the results.

    Usage:

        KS_median, KS_mean, KS_max = \
        PAWN.pawn_indices(X, Y, n, Nboot=1, dummy=False)

        KS_median, KS_mean, KS_max, KS_dummy = \
        PAWN.pawn_indices(X, Y, n, Nboot=1, dummy=True)

    Input:
            X = set of inputs samples                      - numpy.ndarray(N,M)
            Y = set of output samples                      - numpy.ndarray(N, )
                                                        or - numpy.ndarray(N,1)
            n = number of conditioning intervals to
                assess the conditional CDFs:
                - integer if all inputs have the same number of groups
                - list of M integers otherwise

    Optional input:
        Nboot = number of bootstrap resamples to derive    - scalar
                confidence intervals
        dummy = if dummy is True, an articial input is     - boolean
                added to the set of inputs and the
                sensitivity indices are calculated for the
                dummy input.
                The sensitivity indices for the dummy
                input are estimates of the approximation
                error of the sensitivity indices and they
                can be used for screening, i.e. to
                separate influential and non-influential
                inputs as described in Khorashadi Zadeh
                et al. (2017)
                Default value: False
                (see (*) for further explanation).

    Output:
    KS_median = median KS across the conditioning      - numpy.ndarray(Nboot,M)
                intervals (one value for each input
                and each bootstrap resample)
      KS_mean = mean KS across the conditioning        - numpy.ndarray(Nboot,M)
                intervals (one value for each input
                and each bootstrap resample)
       KS_max = max KS across the conditioning         - numpy.ndarray(Nboot,M)
                intervals (one value for each input
                and each bootstrap resample)

    Optional output (if dummy is True):
    KS_dummy = KS of dummy input (one value for       - numpy.ndarray(Nboot, )
                each bootstrap resample)

    --------------------------------------------------------------------------
    ADVANCED USAGE
    for Regional-Response Global Sensitivity Analysis:
    -------------------------------------------------------------------------
    Usage:

    KS_median, KS_mean, KS_max = \
    PAWN.pawn_indices(X, Y, n, Nboot=1, dummy=False,
                      output_condition=allrange, par=[]))

    KS_median, KS_mean, KS_max, KS_dummy = \
    PAWN.pawn_indices(X, Y, n, Nboot=1, dummy=True,
                      output_condition=allrange, par=[]))

    Optional input:
    output_condition = condition on the output value to be     - function
                       used to calculate KS. Use the function:
                       - allrange to keep all output values
                       - below to consider only output
                          values below a threshold value
                          (Y <= Ythreshold)
                       - above to consider only output
                          values above a threshold value
                          (Y >= Ythreshold)
                    (functions allrange, below and above are defined in
                     SAFEpython.util)
                 par = specify the input arguments of the      - list
                       'output_condition' function, i.e. the
                       threshold value when output_condition
                       is 'above' or 'below'.

    For more sophisticate conditions, the user can define its own function
    'output_condition' with the following structure:

        idx = output_condition(Y, param)

    where     Y = output samples (numpy.ndarray(N, ))
          param = parameters to define the condition (list of any size)
            idx = logical values, True if condition is satisfied, False
                  otherwise (numpy.ndarray(N, ))

    NOTE:
     (*) For screening influential and non-influential inputs, we recommend the
         use of the maximum KS across the conditioning intervals (i.e. output
         argument KS_max), and to compare KS_max with the index of the dummy
         input as in Khorashadi Zadeh et al. (2017).

    (**) For each input, the number of conditioning intervals which is actually
         used (n_eff[i]) may be lower than the prescribed number of conditioning
         intervals (n[i]) to ensure that input values that are repeated several
         time belong to the same group.
         See the help of PAWN.pawn_split_sample for further details.

    EXAMPLE:

    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.stats as st
    from SAFEpython.sampling import AAT_sampling
    from SAFEpython.model_execution import model_execution
    from SAFEpython import PAWN
    from SAFEpython.plot_functions import boxplot1
    from SAFEpython.ishigami_homma import ishigami_homma_function

    # Create a generic input-output sample:
    N = 5000 # number of samples
    M = 3 # number of inputs
    xmin = -np.pi; xmax = np.pi
    X = AAT_sampling('lhs', M, st.uniform, [xmin, xmax - xmin], N);
    Y = model_execution(ishigami_homma_function, X)

    # Compute PAWN sensitivity indices:
    n = 10; # number of conditioning intervals
    KS_median, KS_mean, KS_max = PAWN.pawn_indices(X, Y, n)
    plt.figure()
    plt.subplot(131); boxplot1(KS_median, Y_Label='KS (mean')
    plt.subplot(132); boxplot1(KS_mean, Y_Label='KS (mean')
    plt.subplot(133); boxplot1(KS_max, Y_Label='KS (max)')

    # Compute sensitivity indices for the dummy input as well:
    KS_median, KS_mean, KS_max, KS_dummy = PAWN.pawn_indices(X, Y, n, dummy=True)
    plt.figure()
    boxplot1(np.concatenate((KS_max, KS_dummy)),
             X_Labels=['X1', 'X2', 'X3', 'dummy'])

    REFERENCES

    Pianosi, F. and Wagener, T. (2018), Distribution-based sensitivity
    analysis from a generic input-output sample, Env. Mod. & Soft., 108, 197-207.

    Pianosi, F. and Wagener, T. (2015), A simple and efficient method
    for global sensitivity analysis based on cumulative distribution
    functions, Env. Mod. & Soft., 67, 1-11.

    REFERENCE FOR THE DUMMY PARAMETER:

    Khorashadi Zadeh et al. (2017), Comparison of variance-based and moment-
    independent global sensitivity analysis approaches by application to the
    SWAT model, Environmental Modelling & Software,91, 210-222.

    This function is part of the SAFE Toolbox by F. Pianosi, F. Sarrazin
    and T. Wagener at Bristol University (2015).
    SAFE is provided without any warranty and for non-commercial use only.
    For more details, see the Licence file included in the root directory
    of this distribution.
    For any comment and feedback, or to discuss a Licence agreement for
    commercial use, please contact: francesca.pianosi@bristol.ac.uk
    For details on how to cite SAFE in your publication, please see:
    https://www.safetoolbox.info
    """
    ###########################################################################
    # Check inputs and split the input sample
    ###########################################################################

    YY, xc, NC, n_eff, Xk, XX = pawn_split_sample(X, Y, n) # this function
    # checks inputs X, Y and n

    Nx = X.shape
    N = Nx[0]
    M = Nx[1]

    ###########################################################################
    # Check other optional inputs
    ###########################################################################

    if not isinstance(Nboot, (int, np.int8, np.int16, np.int32, np.int64)):
        raise ValueError('"Nboot" must be scalar and integer.')
    if Nboot < 1:
        raise ValueError('"Nboot" must be >=1.')
    if not isinstance(dummy, bool):
        raise ValueError('"dummy" must be scalar and boolean.')
    if not callable(output_condition):
        raise ValueError('"output_condition" must be a function.')

    ###########################################################################
    # Compute indices
    ###########################################################################

    # Set points at which the CDFs will be evaluated:
    YF = np.unique(Y)

    # Initialize sensitivity indices
    KS_median = np.nan * np.ones((Nboot, M))
    KS_mean = np.nan * np.ones((Nboot, M))
    KS_max = np.nan * np.ones((Nboot, M))
    if dummy: # Calculate index for the dummy input
        KS_dummy = np.nan * np.ones((Nboot, ))

    # Compute conditional CDFs
    # (bootstrapping is not used to assess conditional CDFs):
    FC = [np.nan] * M
    for i in range(M): # loop over inputs
        FC[i] = [np.nan] * n_eff[i]
        for k in range(n_eff[i]): # loop over conditioning intervals
            FC[i][k] = empiricalcdf(YY[i][k], YF)

    # Initialize unconditional CDFs:
    FU = [np.nan] * M

    # M unconditional CDFs are computed (one for each input), so that for
    # each input the conditional and unconditional CDFs are computed using the
    # same number of data points (when the number of conditioning intervals
    # n_eff[i] varies across the inputs, so does the shape of the conditional
    # outputs YY[i]).

    # Determine the sample size for the unconditional output bootsize:
    bootsize = [int(np.min(i)) for i in NC] # numpy.ndarray(M,)
    # bootsize is equal to the sample size of the conditional outputs NC, or
    # its  minimum value across the conditioning intervals when the sample size
    # varies across conditioning intervals as may happen when values of an
    # input are repeated several times (more details on this in the Note in the
    # help of the function).

    # To reduce the computational time (the calculation of empirical CDF is
    # costly), the unconditional CDF is computed only once for all inputs that
    # have the same value of bootsize[i].
    bootsize_unique = np.unique(bootsize)
    N_compute = len(bootsize_unique)  # number of unconditional CDFs that will
    # be computed for each bootstrap resample

    # Determine the sample size of the subsample for the dummy input.
    # The sensitivity
    # index for the dummy input will be estimated at this minimum sample size
    # so to estimate the 'worst' approximation error of the sensitivity index
    # across the inputs:
    if dummy:
        bootsize_min = min(bootsize) # we use the smaller sample size across
        # inputs, so that the sensitivity index for the dummy input estimates
        # the 'worst' approximation error of the sensitivity index across the
        # inputs:
        idx_bootsize_min = np.where([i == bootsize_min for i in bootsize])[0]
        idx_bootsize_min = idx_bootsize_min[0] # index of an input for which
        # the sample size of the unconditional sample is equal to bootsize_min

        if N_compute > 1:
            warn('The number of data points to estimate the conditional and '+
                 'unconditional output varies across the inputs. The CDFs ' +
                 'for the dummy input were computed using the minimum sample ' +
                 ' size to provide an estimate of the "worst" approximation' +
                 ' of the sensitivity indices across input.')

    # Compute sensitivity indices with bootstrapping
    for b in range(Nboot): # number of bootstrap resample

        # Compute empirical unconditional CDFs
        for kk in range(N_compute): # loop over the sizes of the unconditional output

            # Bootstrap resapling (Extract an unconditional sample of size
            # bootsize_unique[kk] by drawing data points from the full sample Y
            # without replacement
            idx_bootstrap = np.random.choice(np.arange(0, N, 1),
                                             size=(bootsize_unique[kk], ),
                                             replace='False')
            # Compute unconditional CDF:
            FUkk = empiricalcdf(Y[idx_bootstrap], YF)
            # Associate the FUkk to all inputs that require an unconditional
            # output of size bootsize_unique[kk]:
            idx_input = np.where([i == bootsize_unique[kk] for i in bootsize])[0]
            for i in range(len(idx_input)):
                FU[idx_input[i]] = FUkk

        # Compute KS statistic between conditional and unconditional CDFs:
        KS_all = pawn_ks(YF, FU, FC, output_condition, par)
        # KS_all is a list (M elements) and contains the value of the KS for
        # for each input and each conditioning interval. KS[i] contains values
        # for the i-th input and the n_eff[i] conditioning intervals, and it
        # is a numpy.ndarray of shape (n_eff[i], ).

        #  Take a statistic of KS across the conditioning intervals:
        KS_median[b, :] = np.array([np.median(j) for j in KS_all])  # shape (M,)
        KS_mean[b, :] = np.array([np.mean(j) for j in KS_all])  # shape (M,)
        KS_max[b, :] = np.array([np.max(j) for j in KS_all])  # shape (M,)

        if dummy:
            # Compute KS statistic for dummy parameter:
            # Bootstrap again from unconditional sample (the size of the
            # resample is equal to bootsize_min):
            idx_dummy = np.random.choice(np.arange(0, N, 1),
                                         size=(bootsize_min, ),
                                         replace='False')
            # Compute empirical CDFs for the dummy input:
            FC_dummy = empiricalcdf(Y[idx_dummy], YF)
            # Compute KS stastic for the dummy input:
            KS_dummy[b] = pawn_ks(YF, [FU[idx_bootsize_min]], [[FC_dummy]],
                                  output_condition, par)[0][0]

    if Nboot == 1:
        KS_median = KS_median.flatten()
        KS_mean = KS_mean.flatten()
        KS_max = KS_max.flatten()

    if dummy:
        return KS_median, KS_mean, KS_max, KS_dummy
    else:
        return KS_median, KS_mean, KS_max


def pawn_ks(YF, FU, FC, output_condition=allrange, par=[]):

    """ Compute the Kolmogorov-Smirnov (KS) between the unconditional and
    conditional output for each input and each conditioning interval, for
    ONE sample/bootstrap resample.

    This function is called internally in PAWN.pawn_indices, PAWN.pawn_plot_ks.

    Usage:
        KS = PAWN.pawn_ks(YF, FU, FC, output_condition=allrange, par=[])

    Input:
    YF = values of Y at which the Cumulative  Distribution - numpy.ndarray(P, )
         Functions (CDFs) FU and FC are given.
    FU = values of the empirical unconditional output CDFs - list(M elements)
         to calculate the KS statistic for each input.
         FU[i] is a list of n_eff[i] CDFs for the i-th
         input.
    FC = values of the empirical conditional output        - list(M elements)
         CDFs for each input and each conditioning
         interval.
         FC[i] is a list of n_eff[i] CDFs conditional to
         the i-th input.
         FC[i][k] is obtained by fixing the i-th input to
         its k-th conditioning interval (while the other
         inputs vary freely), and it is a np.ndarray of
         shape (P, ).

    Optional input:
    See the help of PAWN.pawn_indices for information on the optional inputs
    "output_condition" and "par".

    Output:
    KS = KS-statistic calculated between conditional       - list(M elements)
         and unconditional output for the M inputs and
         the n_eff conditioning intervals.
         KS[i] contains the KS values for the i-th input
         and the n_eff[i] conditioning intervals, and it
         is a numpy.ndarray of shape (n_eff[i], ).

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

    if not callable(output_condition):
        raise ValueError('"output_condition" must be a function.')
    if not isinstance(par, list):
        raise ValueError('"par" must be a list.')

    M = len(FC) # number of inputs

    # Initialise variable
    KS = [np.nan] * M

    # Find subset of output values satisfying a given condition
    idx = output_condition(YF, par)

    # Calculate KS statistics:
    for i in range(M): # loop over inputs
        n_effi = len(FC[i])
        KS[i] = np.nan * np.ones((n_effi,))

        for k in range(n_effi): # loop over conditioning intervals
            # Compute KS:
            KS[i][k] = np.max(abs(FU[i][idx] - FC[i][k][idx]))

    return KS

def pawn_convergence(X, Y, n, NN, Nboot=1, dummy=False,
                     output_condition=allrange, par=[]):

    """ This function computes the PAWN sensitivity indices (Pianosi and
    Wagener, 2015, 2018) using sub-samples of the original sample 'Y' of
    increasing size.

    The function splits the output sample to create the conditional output.
    The splitting strategy is an extension of the strategy for uniformy
    distributed inputs described in Pianosi and Wagener (2018) to handle inputs
    sampled from any distribution.

    Usage:
    KS_median, KS_mean, KS_max = \
    PAWN.pawn_convergence(X, Y, n, NN, Nboot=1, dummy=False)

    KS_median, KS_mean, KS_max, KS_dummy = \
    PAWN.pawn_convergence(X, Y, n, NN, Nboot=1, dummy=True)

    Input:
            X = set of inputs samples                      - numpy.ndarray(N,M)
            Y = set of output samples                      - numpy.ndarray(N, )
                                                        or - numpy.ndarray(N,1)
            n = number of conditioning intervals to
                assess the conditional CDFs:
                - integer if all inputs have the same number of groups
                - list of M integers otherwise
           NN = subsample sizes at which indices will be   - numpy.ndarray (R, )
                estimated (max(NN) must not exceed N)

    Optional input:
        Nboot = number of bootstrap resamples to derive    - scalar
                confidence intervals
        dummy = if dummy is True, an articial input is     - boolean
                added to the set of inputs and the
                sensitivity indices are calculated for the
                dummy input. See the help of
                PAWN.pawn_indices for refence and further
                explanation on the usage of the dummy input.

    Output:
    KS_median = median KS across the conditioning      - numpy.ndarray(Nboot,M)
                intervals (one value for each input
                and each bootstrap resample)
      KS_mean = mean KS across the conditioning        - numpy.ndarray(Nboot,M)
                intervals (one value for each input
                and each bootstrap resample)
       KS_max = max KS across the conditioning         - numpy.ndarray(Nboot,M)
                intervals (one value for each input
                and each bootstrap resample)

    Optional output (if dummy is True):
    KS_dummy = KS of dummy input (one value for       - numpy.ndarray(Nboot, )
                each bootstrap resample)

    --------------------------------------------------------------------------
    ADVANCED USAGE
    for Regional-Response Global Sensitivity Analysis:
    -------------------------------------------------------------------------
    Usage:

    KS_median, KS_mean, KS_max = \
    PAWN.pawn_convergence(X, Y, n, NN, Nboot=1, dummy=False,
                      output_condition=allrange, par=[])

    KS_median, KS_mean, KS_max, KS_dummy = \
    PAWN.pawn_convergence(X, Y, n, NN, Nboot=1, dummy=True,
                      output_condition=allrange, par=[])

    See the help of PAWN.pawn_indices for information on the optional inputs
    "output_condition" and "par".

    SEE ALSO PAWN.split_sample,PAWN.pawn_indices

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
    # All inputs will be checked when applying the function pawn_indices
    # apart from 'NN'.

    if not isinstance(X, np.ndarray):
        raise ValueError('"X" must be a numpy.array.')
    Nx = X.shape
    N = Nx[0]
    M = Nx[1]

    if not isinstance(NN, np.ndarray):
        raise ValueError('"NN" must be a numpy.array.')
    if NN.dtype.kind != 'i':
        raise ValueError('"NN" must contain integers.')
    if any(i < 0 for i in np.diff(NN)):
        raise ValueError('elements in "NN" must be sorted in ascending order')
    if any(i < 0 for i in NN):
        raise ValueError('elements in "NN" must be positive')
    if NN[-1] > N:
        raise ValueError('Maximum value in "NN" must not exceed N = %d' % N)
    NN_shape = NN.shape
    if len(NN_shape) > 1:
        raise ValueError('"NN" must be of shape (R,).')

    ###########################################################################
    # Compute indices
    ###########################################################################

    R = len(NN)
    # Intialise variables
    KS_median = [np.nan]*R
    KS_mean = [np.nan]*R
    KS_max = [np.nan]*R
    KS_dummy = [np.nan]*R

    for j in range(R): # loop over sample sizes

        Xj, idx_new = lhcube_shrink(X, NN[j]) # drop rows while trying to
        #  maximize the spread between the points
        Yj = Y[idx_new, :]

        if dummy:
           KS_median[j], KS_mean[j], KS_max[j], KS_dummy[j] = \
           pawn_indices(Xj, Yj, n, Nboot, dummy, output_condition, par)
        else:
           KS_median[j], KS_mean[j], KS_max[j] = \
           pawn_indices(Xj, Yj, n, Nboot, dummy, output_condition, par)

    if Nboot <= 1: # return a numpy.ndarray (R, M)
        KS_median_tmp = np.nan*np.ones((R, M))
        KS_mean_tmp = np.nan*np.ones((R, M))
        KS_max_tmp = np.nan*np.ones((R, M))
        KS_dummy_tmp = np.nan*np.ones((R, M))

        for j in range(R):
            KS_median_tmp[j, :] = KS_median[j]
            KS_mean_tmp[j, :] = KS_mean[j]
            KS_max_tmp[j, :] = KS_max[j]
            KS_dummy_tmp[j, :] = KS_dummy[j]

        KS_median = KS_median_tmp
        KS_mean = KS_mean_tmp
        KS_max = KS_max_tmp
        KS_dummy = KS_dummy_tmp

    if dummy:
        return KS_median, KS_mean, KS_max, KS_dummy
    else:
        return KS_median, KS_mean, KS_max


def pawn_plot_cdf(X, Y, n, n_col=5, Y_Label='output y', cbar=False,
                  labelinput=''):

    """ This function computes and plots the unconditional output Cumulative
    Distribution Funtions (i.e. when all inputs vary) and the conditional CDFs
    (when one input is fixed to a given conditioning interval, while the other
    inputs vary freely).

    The function splits the output sample to create the conditional output
    by calling internally the function PAWN.pawn_split_sample. The splitting
    strategy is an extension of the strategy for uniformy distributed inputs
    described in Pianosi and Wagener (2018) to handle inputs sampled from any
    distribution.
    (see help of PAWN.pawn_split_sample for further explanation).

    The sensitivity indices for the PAWN method (KS statistic) measures the
    distance between these conditional and unconditional output CDFs
    (see help of PAWN.pawn_indices for further details and reference).

    Usage:
    YF, FU, FC, xc = PAWN.pawn_plot_cdf(X, Y, n, n_col=5, Y_Label='output y',
                                        cbar=False, labelinput='')

    Input:
             X = set of inputs samples                     - numpy.ndarray(N,M)
             Y = set of output samples                     - numpy.ndarray(N,)
                                                        or - numpy.ndarray(N,1)
             n = number of conditioning intervals
                 - integer if all inputs have the same number of groups
                 - list of M integers otherwise

    Optional input:
         n_col = number of panels per row in the plot      - integer
                 (default: min(5, M))
       Y_Label = legend for the horizontal axis            - string
                 (default: 'output y')
          cbar = flag to add a colobar that indicates the  - boolean
                 centers of the conditioning intervals for
                 the different conditional CDFs:
                 - if True = colorbar
                 - if False = no colorbar
    labelinput = label for the axis of colorbar (input    - list (M elements)
                 name) (default: ['X1','X2',...,XM'])

    Output:
            YF = values of Y at which the CDFs FU and FC   - numpy.ndarray(P, )
                 are given
            FU = values of the empirical unconditional     - list(M elements)
                 output CDFs. FU[i] is a numpy.ndarray(P, )
                 (see the Note below for further
                 explanation)
            FC = values of the empirical conditional       - list(M elements)
                 output CDFs for each input and each
                 conditioning interval.
                 FC[i] is a list of n_eff[i] CDFs
                 conditional to the i-th input.
                 FC[i][k] is obtained by fixing the i-th
                 input to its k-th conditioning interval
                 (while the other inputs vary freely),
                 and it is a np.ndarray of shape (P, ).
                 (see the Note below for further
                 explanation)
           xc = subsamples' centers (i.e. mean value of    - list(M elements)
                Xi over each conditioning interval)
                xc[i] is a np.ndarray(n_eff[i],) and
                contains the centers for the n_eff[i]
                conditioning intervals for the i-th input.

    Note:
    (*)  For each input, the number of conditioning intervals which is actually
         used (n_eff[i]) may be lower than the prescribed number of conditioning
         intervals (n[i]) to ensure that input values that are repeated several
         time belong to the same group.
         See the help of PAWN.pawn_split_sample for further details.

    (**) FU[i] and FC[i][k] (for any given i and k) are built using the same
         number of data points so that two CDFs can be compared by calculating
         the KS statistic (see help of PAWN.pawn_ks and PAWN.pawn_indices
         for further explanation on the calculation of the KS statistic).

    Example:

    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.stats as st
    from SAFEpython.sampling import AAT_sampling
    from SAFEpython.model_execution import model_execution
    from SAFEpython import PAWN
    from SAFEpython.plot_functions import boxplot1
    from SAFEpython.ishigami_homma import ishigami_homma_function

    # Create a generic input-output sample:
    N = 1000 # number of samples
    M = 3 # number of inputs
    xmin = -np.pi; xmax = np.pi
    X = AAT_sampling('lhs', M, st.uniform, [xmin, xmax - xmin], N);
    Y = model_execution(ishigami_homma_function, X)

    # Calculate and plot CDFs
    n = 10 # number of conditioning intervals
    YF, FU, FC, xc = PAWN.pawn_plot_cdf(X, Y, n)
    YF, FU, FC, xc = PAWN.pawn_plot_cdf(X, Y, n, cbar=True) # Add colorbar

    This function is part of the SAFE Toolbox by F. Pianosi, F. Sarrazin
    and T. Wagener at Bristol University (2015).
    SAFE is provided without any warranty and for non-commercial use only.
    For more details, see the Licence file included in the root directory
    of this distribution.
    For any comment and feedback, or to discuss a Licence agreement for
    commercial use, please contact: francesca.pianosi@bristol.ac.uk
    For details on how to cite SAFE in your publication, please see:
    https://www.safetoolbox.info"""

    # Options for the graphic
    pltfont = {'fontname': 'DejaVu Sans', 'fontsize': 15} # font
    colorscale = 'gray' # colorscale
    # Text formating of ticklabels
    yticklabels_form = '%3.1f' # float with 1 character after decimal point
    # yticklabels_form = '%d' # integer

    ###########################################################################
    # Check inputs and split the input sample
    ###########################################################################

    YY, xc, NC, n_eff, Xk, XX = pawn_split_sample(X, Y, n) # this function
    # checks inputs X, Y and n

    Nx = X.shape
    N = Nx[0]
    M = Nx[1]

    ###########################################################################
    # Check optional inputs for plotting
    ###########################################################################

    if not isinstance(n_col, (int, np.int8, np.int16, np.int32, np.int64)):
        raise ValueError('"n_col" must be scalar and integer.')
    if n_col < 0:
        raise ValueError('"n_col" must be positive.')
    if not isinstance(Y_Label, str):
        raise ValueError('"Y_Label" must be a string.')
    if not isinstance(cbar, bool):
        raise ValueError('"cbar" must be scalar and boolean.')
    if not labelinput:
        labelinput = [np.nan]*M
        for i in range(M):
            labelinput[i] = 'X' + str(i+1)
    else:
        if not isinstance(labelinput, list):
            raise ValueError('"labelinput" must be a list with M elements.')
        if not all(isinstance(i, str) for i in labelinput):
            raise ValueError('Elements in "labelinput" must be strings.')
        if len(labelinput) != M:
            raise ValueError('"labelinput" must have M elements.')

    ###########################################################################
    # Compute CDFs
    ###########################################################################

    # Set points at which the CDFs will be evaluated:
    YF = np.unique(Y)

    # Compute conditional CDFs:
    FC = [np.nan] * M
    for i in range(M): # loop over inputs
        FC[i] = [np.nan] * n_eff[i]
        for k in range(n_eff[i]): # loop over conditioning intervals
            FC[i][k] = empiricalcdf(YY[i][k], YF)

    # Initialize unconditional CDFs:
    FU = [np.nan] * M

    # M unconditional CDFs are computed (one for each input), so that for
    # each input the conditional and unconditional CDFs are computed using the
    # same number of data points (when the number of conditioning intervals
    # n_eff[i] varies across the inputs, so does the shape of the conditional
    # outputs YY[i]).

    # Determine the sample size for the unconditional output NU:
    NU = [int(np.min(i)) for i in NC] # numpy.ndarray(M,)
    # NU is equal to the sample size of the conditional outputs NC, or its
    # minimum value across the conditioning intervals when the sample size
    # varies across conditioning intervals as may happen when values of an
    # input are repeated several times (more details on this in the Note in the
    #  help of the function).

    # To reduce the computational time (the calculation of empirical CDF is
    # costly), the unconditional CDF is computed only once for all inputs that
    # have the same value of NU[i].
    NU_unique = np.unique(NU)
    N_compute = len(NU_unique) # number of unconditional CDFs that will be computed

    for kk in range(N_compute): # loop over the sizes of the unconditional output

        # Extract an unconditional sample of size NU_unique[kk] by drawing data
        # points from the full sample Y without replacement
        idx = np.random.choice(np.arange(0, N, 1), size=(NU_unique[kk], ),
                               replace='False')
        # Compute unconditional output CDF:
        FUkk = empiricalcdf(Y[idx], YF)
        # Associate the FUkk to all inputs that require an unconditional output
        # of size NU_unique[kk]:
        idx_input = np.where([i == NU_unique[kk] for i in NU])[0]
        for j in range(len(idx_input)):
            FU[idx_input[j]] = FUkk

    ###########################################################################
    # Plot
    ###########################################################################
    n_col = min(n_col, M) # n_col <= M
    n_row = int(np.ceil(M/n_col))

    plt.figure()

    for i in range(M): # loop over inputs

        # Prepare color and legend
        cmap = mpl.cm.get_cmap(colorscale, n_eff[i]+1) # return colormap,
        # (n+1) so that the last line is not white
        # Make sure that subsample centers are ordered:
        iii = np.argsort(xc[i])
        ccc = np.sort(xc[i])

        plt.subplot(n_row, n_col, i+1)
        ax = plt.gca()

        if cbar: # plot dummy mappable to generate the colorbar
            Map = plt.imshow(np.array([[0, 1]]), cmap=cmap)
            plt.cla() # clear axes (do not display the dummy map)

        # Plot a horizontal dashed line at F=1:
        plt.plot(YF, np.ones(YF.shape), '--k')

        # Plot conditional CDFs in gray scale:
        for k in range(n_eff[i]):
            plt.plot(YF, FC[i][iii[k]], color=cmap(k), linewidth=2)
        plt.xticks(**pltfont); plt.yticks(**pltfont)
        plt.xlabel(Y_Label, **pltfont)

        # Plot unconditional CDF in red:
        plt.plot(YF, FU[i], 'r', linewidth=3)

        plt.xlim([min(YF), max(YF)]); plt.ylim([-0.02, 1.02])

        if cbar: # Add colorbar to the left
             cb_ticks = [' '] * n_eff[i]
             for k in range(n_eff[i]):
                 cb_ticks[k] = yticklabels_form % ccc[k]
             # Add colorbar (do not display the white color by adjuting the
             # input argument 'boundaries')
             cb = plt.colorbar(Map, ax=ax,
                               boundaries=np.linspace(0, 1-1/((n_eff[i]+1)),
                                                      n_eff[i]+1))
             cb.set_label(labelinput[i], **pltfont)
             cb.Fontname = pltfont['fontname']
             # set tick labels at the center of each color:
             cb.set_ticks(np.linspace(1/(2*(n_eff[i]+1)), 1-3/(2*(n_eff[i]+1)),
                                      n_eff[i]))
             cb.set_ticklabels(cb_ticks)
             cb.ax.tick_params(labelsize=pltfont['fontsize'])
             # Map.set_clim(0,1-1/(n+1))
             ax.set_aspect('auto') # Ensure that axes do not shrink

    return YF, FU, FC, xc


def pawn_plot_ks(YF, FU, FC, xc, n_col=5, X_Labels='',
                 output_condition=allrange, par=[]):

    """ This function computes and plots the Kolmogorov-Smirnov (KS) statistic
    between conditional and unconditional output CDFs for each input and each
    conditioning interval.

    Usage:
        KS_all = pawn_plot_ks(YF, FU, FC, xc, n_col=5, X_Labels='')

    Input:
         YF = values of Y at which the CDFs FU and FC      - numpy.ndarray(P, )
                 are given
         FU = values of the empirical unconditional        - list(M elements)
                 output CDFs. FU[i] is a numpy.ndarray(P, )
                 (see the Note below for further
                 explanation)
         FC = values of the empirical conditional          - list(M elements)
                 output CDFs for each input and each
                 conditioning interval.
                 FC[i] is a list of n_eff[i] CDFs
                 conditional to the i-th input.
                 FC[i][k] is obtained by fixing the i-th
                 input to its k-th conditioning interval
                 (while the other inputs vary freely),
                 and it is a np.ndarray of shape (P, ).
                 (see the Note below for further
                 explanation)
         xc = subsamples' centers (i.e. mean value of     - list(M elements)
                Xi over each conditioning interval)
                xc[i] is a np.ndarray(n_eff[i],) and
                contains the centers for the n_eff[i]
                conditioning intervals for the i-th input.

    Note: YF, FU, FC and xc are computed using the function PAWN.pawn_plot_cdf

    Optional input:
      n_col = number of panels per row in the plot        - integer
                 (default: min(5, M))
    X_Label = label for the x-axis (input name)           - list (M elements)
                 (default: ['X1','X2',...,XM'])

    Output:
     KS_all = KS-statistic calculated between conditional - list(M elements)
         and unconditional output for the M inputs and
         the n_eff conditioning intervals.
         KS[i] contains the KS values for the i-th input
         and the n_eff[i] conditioning intervals, and it
         is a numpy.ndarray of shape (n_eff[i], ).

    --------------------------------------------------------------------------
    ADVANCED USAGE
    for Regional-Response Global Sensitivity Analysis:
    -------------------------------------------------------------------------
    Usage:

    KS_all = pawn_plot_ks(YF, FU, FC, xc, n_col=5, X_Labels='',
                          output_condition=allrange, par=[])

    See the help of PAWN.pawn_indices for information on the optional inputs
    "output_condition" and "par".

    Example:

    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.stats as st
    from SAFEpython.sampling import AAT_sampling
    from SAFEpython.model_execution import model_execution
    from SAFEpython import PAWN
    from SAFEpython.plot_functions import boxplot1
    from SAFEpython.ishigami_homma import ishigami_homma_function

    # Create a generic input-output sample:
    N = 1000 # number of samples
    M = 3 # number of inputs
    xmin = -np.pi; xmax = np.pi
    X = AAT_sampling('lhs', M, st.uniform, [xmin, xmax - xmin], N);
    Y = model_execution(ishigami_homma_function, X)

    # Calculate CDFs:
    n = 10 # number of conditioning intervals
    YF, FU, FC, xc = PAWN.pawn_plot_cdf(X, Y, n)

    # Calculate and plot KS statistics:
    KS_all = PAWN.pawn_plot_ks(YF, FU, FC, xc)

    This function is part of the SAFE Toolbox by F. Pianosi, F. Sarrazin
    and T. Wagener at Bristol University (2015).
    SAFE is provided without any warranty and for non-commercial use only.
    For more details, see the Licence file included in the root directory
    of this distribution.
    For any comment and feedback, or to discuss a Licence agreement for
    commercial use, please contact: francesca.pianosi@bristol.ac.uk
    For details on how to cite SAFE in your publication, please see:
    https://www.safetoolbox.info"""

    # Options for the graphic
    pltfont = {'fontname': 'DejaVu Sans', 'fontsize': 15} # font
    colorscale = 'gray' # colorscale for the markers
    ms = 7 # size of markers

    ###########################################################################
    # Check inputs and calculate KS-statistic
    ###########################################################################
    KS_all = pawn_ks(YF, FU, FC, output_condition, par)# this function
    # checks inputs F, FU, FC, output_condition and par

    M = len(KS_all)
     ###########################################################################
    # Check optional inputs for plotting
    ###########################################################################

    if not isinstance(n_col, (int, np.int8, np.int16, np.int32, np.int64)):
        raise ValueError('"n_col" must be scalar and integer.')
    if n_col < 0:
        raise ValueError('"n_col" must be positive.')

    if not X_Labels:
        X_Labels = [np.nan]*M
        for i in range(M):
            X_Labels[i] = 'X' + str(i+1)
    else:
        if not isinstance(X_Labels, list):
            raise ValueError('"X_Labels" must be a list with M elements.')
        if not all(isinstance(i, str) for i in X_Labels):
            raise ValueError('Elements in "X_Labels" must be strings.')
        if len(X_Labels) != M:
            raise ValueError('"X_Labels" must have M elements.')

    ###########################################################################
    # Plot
    ###########################################################################
    n_col = min(n_col, M) # n_col <= M
    n_row = int(np.ceil(M/n_col))

    plt.figure()

    for i in range(M): # loop over inputs

        ni = len(KS_all[i]) # number of conditioning intervals for input i
        # plot KS values as coloured circles on a gray scale:
        col = mpl.cm.get_cmap(colorscale, ni)

        # Make sure that subsample centers are ordered:
        iii = np.argsort(xc[i])
        ccc = np.sort(xc[i])
        kkk = KS_all[i][iii]

        plt.subplot(n_row, n_col, i+1)
        # Plot black line:
        plt.plot(ccc, kkk, '-k')

        # plot KS values as circles:
        for k in range(ni):# loop over conditioning intervals
            plt.plot(ccc[k], kkk[k], 'ok', markerfacecolor=col(k), markersize=ms)

        plt.xticks(**pltfont); plt.yticks(**pltfont)
        plt.xlabel(X_Labels[i], **pltfont)
        plt.ylabel('KS', **pltfont)
        plt.xlim([ccc[0]-(ccc[1]-ccc[0])/2, ccc[-1]+(ccc[1]-ccc[0])/2])
        plt.ylim([0, 1])

    return KS_all
