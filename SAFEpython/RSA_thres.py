"""
    Module to perform Regional Sensitivity Analysis (RSA)

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

    Spear, R.C. and Hornberger, G.M. (1980). Eutrophication in peel inlet,
    II, identification of critical uncertianties via generalized sensitivity
    analysis, Water Resour. Res., 14, 43-49.

    Sieber, A., Uhlenbrook, S. (2005). Sensitivity analysis of a distributed
    catchment model to verify the model structure, Journal of Hydrology,
    310(1-4), 216-235.
"""
from __future__ import division, absolute_import, print_function

from warnings import warn
import numpy as np
from numpy.matlib import repmat
import matplotlib.pyplot as plt

from SAFEpython.util import empiricalcdf
from SAFEpython.lhcube import lhcube_shrink


def RSA_indices_thres(X, Y, threshold, Nboot=0):

    """ Computation function for Regional Sensitivity Analysis
    (as first proposed by Spear and Hornberger (1980);
    for another application example see e.g. Sieber and Uhlenbrook (2005)).

    It splits the samples in a dataset X into two datasets
    ('behavioural' Xb and 'non-behavioural' Xnb)
    depending on whether the associated sample in Y satisfies
    the condition (behavioural):

                       Y[i,j] < threshold[j]     for j = 1,...,P

    or not (non-behavioural).
    Then it assesess the distance between the CDFs of Xb and Xnb by a suitable
    measure (maximum vertical distance, area between curves, etc.).

    See also 'RSA_plot_thres' for how to visualize results.

    Usage:
       mvd, spread, irr, idxb = \
                          RSA_thres.RSA_indices_thres(X, Y, threshold, Nboot=0)

    Input:
            X = set of inputs samples                   - numpy.ndarray (N,M)
            Y = set of output samples                   - numpy.ndarray (N, )
                                                     or - numpy.ndarray (N,P)
    threshold = threshold for output values. The        if P > 1
                number of elements in threshold         - list (P elements)
                corresponds to the number of outputs    else
                in Y (i.e. P)                           - scalar (float or int)

    Optional input:
        Nboot = number of resamples used for            - integer
                boostrapping (if not specified: Nboot=0,
                i.e. no bootstrapping)

    Output:
          mvd = maximum vertical distance between    if Nboot <= 1:
                the inputs' CDFs estimated over the   - numpy.ndarray (M, )
                behavioural set and the non-         if Nboot > 1:
                behavioural set, and for each        - numpy.ndarray (Nboot,M)
                bootstrap resample when boostrapping
                is used (i.e. Nboot>1)
       spread = area between the inputs' CDF         if Nboot <= 1:
                estimated over the behavioural set   - numpy.ndarray (M, )
                and the non-behavioural set,         if Nboot > 1:
                and for each bootstrap               - numpy.ndarray (Nboot,M)
                resample when boostrapping is used
                (i.e. Nboot>1).
          irr = input range reduction in the         if Nboot <= 1:
                behavioural set compared to the       - numpy.ndarray (M, )
                original range in the full sample,   if Nboot > 1:
                estimated for each bootstrap          - numpy.ndarray (Nboot,M)
                resample when boostrapping is used
                (i.e. Nboot>1)
        idxb = indices of samples satisfying the      - numpy.ndarray (N, )
               condition (behavioural)
               You can easily derive the two datasets Xb and Xnb as:
                 Xb  = X[idxb,:]
                 Xnb = X[~idxb,:]

    Note: Available measures of distance between CDFs:

    1. mvd = maximum vertical difference between the two CDFs
           (behavioural and non-behavioural).
           The larger the mvd of an input factor,the higher the sensitivity
           to that factor. However, in contrast to ''spread'', this is an
           absolute measure, i.e. it has meaningful value per se, regardless of
           the units of measures of X and Y.
           In fact, by definition:
           - it varies between 0 and 1
           - if equal to 0 then the two CDFs are exactely the same
           - if equal to 1 then the two CDFs are 'mutually exclusive'
             (the same value is given probability 0 by one CDF and 1
              by the other)

    2. spread = area between the inputs CDF estimated over the behavioural set,
              and the CDF of the non-behavioural set.
              The larger the spread of an input factor, the higher the
              sensitivity to that factor.

    3. irr = input range reduction, that is, the relative reduction of the
           input range wrt the original range when considering the behavioural
           set only. Again an absolute measure of sensitivity, in fact:
          - it varies between 0 and 1
          - if equal to 0 then the range has been reduced to one point
          - if equal to 1 then the range has not been reduced at all
          - the lower it is, the higher the sensitivity to that input

    REFERENCES

    Spear, R.C. and Hornberger, G.M. (1980). Eutrophication in peel inlet,
    II, identification of critical uncertianties via generalized sensitivity
    analysis, Water Resour. Res., 14, 43-49.

    Sieber, A., Uhlenbrook, S. (2005). Sensitivity analysis of a distributed
    catchment model to verify the model structure, Journal of Hydrology,
    310(1-4), 216-235.

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

    if len(Ny) == 1:
        P = 1
    elif len(Ny) == 2:
        P = Ny[1]

    if Ny[0] != N:
        raise ValueError('"X" and "Y" must have the same number of rows.')

    if isinstance(threshold, list):
        if not all(isinstance(i, float) or isinstance(i, int) for i in threshold):
            raise ValueError('Elements in "threshold" must be int or float.')
        if len(threshold) != P:
            raise ValueError('"threshold" must be a list with P elements')
    elif isinstance(threshold, (int, np.int8, np.int16, np.int32, np.int64,
                                float, np.float16, np.float32, np.float64)):
        if P > 1:
            raise ValueError('"threshold" must be a list with P elements')
    else:
        raise ValueError('"threshold" must be list, int or float.')

    ###########################################################################
    # Check optional inputs
    ###########################################################################

    if not isinstance(Nboot, (int, np.int8, np.int16, np.int32, np.int64)):
        raise ValueError('"Nboot" must be scalar and integer.')
    if Nboot < 0:
        raise ValueError('"Nboot" must be positive.')

    ###########################################################################
    # Compute indices
    ###########################################################################
    if Nboot > 1: # Use bootstrapping
        bootsize = N # size of bootstrap resamples
        B = np.random.randint(N, size=(bootsize, Nboot))

        mvd = np.nan * np.ones((Nboot, M))
        spread = np.nan * np.ones((Nboot, M))
        irr = np.nan * np.ones((Nboot, M))

        for n in range(Nboot):
            Xi = X[B[:, n], :]
            if len(Ny) == 1:
                Yi = Y[B[:, n]]
            else:
                Yi = Y[B[:, n], :]

            mvd[n, :], spread[n, :], irr[n, :], _ = \
                    compute_indices(Xi, Yi, threshold)
        # Note that RSA may return a vector of NaNs if the bootstrap
        # resample Yi is such that the threshold condition is never
        # satisfied (or always satisfied).

        # Last, let's call the function once more to obtain the vector
        # 'idxb' of the indices of behavioural input samples (needed to be
        # returned among the output arguments):
        _, _, _, idxb = compute_indices(X, Y, threshold)


    else: # no bootstrapping
        mvd, spread, irr, idxb = compute_indices(X, Y, threshold)

    return  mvd, spread, irr, idxb


def compute_indices(X, Y, threshold):

    """ This function conputes the sensitivity indices for Regional Sensitivity
    Analysis for ONE sample/bootstrap resample.

    This function is called internally in RSA_thres.RSA_indices_thres.

    Usage:
        mvd, spread, irr, idxb = RSA_thres.compute_indices(X, Y, threshold)

    Input:
            X = set of inputs samples                   - numpy.ndarray (N,M)
            Y = set of output samples                   - numpy.ndarray (N, )
                                                     or - numpy.ndarray (N,P)
    threshold = threshold for output values. The        if P > 1
                number of elements in threshold         - list (P elements)
                corresponds to the number of outputs    else
                in Y (i.e. P)                           - scalar (float or int)

    Output:
         mvd = maximum vertical distance between        - numpy.ndarray (M, )
                the inputs' CDFs estimated over the
                behavioural set and the non-
                behavioural set
       spread = area between the inputs' CDF            - numpy.ndarray (M, )
                estimated over the behavioural set
                and the non-behavioural set
          irr = input range reduction in the            - numpy.ndarray (M, )
                behavioural set compared to the
                original range in the full sample
        idxb = indices of samples satisfying the        - numpy.ndarray (N, )
               condition (behavioural)
               You can easily derive the two datasets Xb and Xnb as:
                 Xb  = X[idxb,:]
                 Xnb = X[~idxb,:]

    For reference and further details see help of RSA_thres.RSA_indices_thres.

    This function is part of the SAFE Toolbox by F. Pianosi, F. Sarrazin
    and T. Wagener at Bristol University (2015).
    SAFE is provided without any warranty and for non-commercial use only.
    For more details, see the Licence file included in the root directory
    of this distribution.
    For any comment and feedback, or to discuss a Licence agreement for
    commercial use, please contact: francesca.pianosi@bristol.ac.uk
    For details on how to cite SAFE in your publication, please see:
    https://www.safetoolbox.info"""

    Nx = X.shape
    N = Nx[0]
    M = Nx[1]
    Ny = Y.shape

    # Identify sets above and below the threshold:
    if len(Ny) == 1:
        idxb = Y < repmat(threshold, N, 1).flatten()
    elif len(Ny) == 2:
        P = Ny[1]
        idxb = np.sum(Y < repmat(threshold, N, 1), axis=1) == P

    # Initialise arrays of indices:
    mvd = np.nan*np.ones([M, ])
    spread = np.nan*np.ones([M, ])
    irr = np.nan*np.ones([M, ])
    # Define above and below subsamples:
    Xb = X[idxb, :]
    Nb = Xb.shape # number of behavioural parameterisations
    Xnb = X[~idxb, :]
    Nnb = Xnb.shape # number of non-behavioural parameterisations

    if Nb[0] <= 0:
        warn('Cannot find any output value below the threshold! Try increasing the threshold value')
    elif Nnb[0] <= 0:
        warn('Cannot find any output value above the threshold! Try reducing the threshold value')

    else: # perform RSA

        for i in range(M):
            # Empirical CDF of behavioural and non-behavioural inputs:
            xx = np.unique(np.sort(X[:, i]))
            CDFb = empiricalcdf(Xb[:, i], xx)
            CDFnb = empiricalcdf(Xnb[:, i], xx)
            # Compute distances between CDFs:
            mvd[i] = np.max(abs(CDFb - CDFnb))
            spread[i] = np.trapz(np.max(np.stack((CDFb, CDFnb), axis=0), axis=0), x=xx) -\
                 np.trapz(np.min(np.stack((CDFb, CDFnb), axis=0), axis=0), x=xx)

        # Ranges of input factors that produce ''behavioural'' output:
        xmin = np.min(Xb, axis=0)
        xmax = np.max(Xb, axis=0)
        # Compute the relative reduction wrt the original ranges of input factors
        # (as they appear in the sample ''X''):
        irr = 1 - (xmax-xmin) / (np.max(X, axis=0)- np.min(X, axis=0))

    return mvd, spread, irr, idxb


def RSA_convergence_thres(X, Y, NN, threshold, Nboot=0):

    """This function computes and plots the sensitivity indices obtained by
    using the function 'RSA_indices_thres' with an increasing number of output
    samples.
    (see help of RSA_indices_thres for details about RSA and references)

    Usage:
    mvd, spread, irr = \
                  RSA_thres.RSA_convergence_thres(X, Y, NN, threshold, Nboot=0)

    Input:
            X = set of inputs samples                   - numpy.ndarray (N,M)
            Y = set of output samples                   - numpy.ndarray (N,)
                                                     or - numpy.ndarray (N,P)
           NN = subsample sizes at which                - numpy.ndarray (R, )
                indices will be estimated (max(NN)
                must not exceed N)
   threshold = threshold for output values. The        if P > 1
                number of elements in threshold         - list (P elements)
                corresponds to the number of outputs    else
                in Y (i.e. P)                           - scalar (float or int)

   Optional input:
        Nboot = number of resamples used for            - integer
                boostrapping(if not specified:
               Nboot=0, i.e. no bootstrapping)

    Output (*):
          mvd = maximum vertical distance between       if Nboot <= 1:
                the inputs' CDFs estimated over the     - numpy.ndarray (R,M)
                behavioural set and the non-            if Nboot > 1:
                behavioural set at different sample     - list(R elements)
                sizes. When bootstrapping is used
                (i.e. Nboot>1), mvd is a list and mvd[j]
                is a numpy.ndarray (Nboot,M) of the
                Nboot estimates of 'mvd' for the jth
                sample size.
       spread = area between the inputs' CDF            if Nboot <= 1:
                estimated over the behavioural set      - numpy.ndarray (R,M)
                and the non-behavioural at different    if Nboot > 1:
                sample sizes. When bootstrapping is     - list(R elements)
                used (i.e. Nboot>1), spread[j] is a
                numpy.ndarray(Nboot,M) of the Nboot
                estimates of 'spread' for the jth
                sample size.
          irr = input range reduction in the            if Nboot <= 1:
                behavioural set compared to the         - numpy.ndarray (R,M)
                original range in the full sample.      if Nboot > 1:
                When bootstrapping is used, irr[j]      - list(R elements)
                is a numpy.ndarray(Nboot,M) of the
                Nboot estimates of 'irr' for the jth
                sample size.
    (*) for more information about the sensitivity measures, see help of
    function 'RSA_indices_thres'

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
    # All inputs will be checked when applying the function RSA_indices_thres
    # apart from 'NN'.

    if not isinstance(X, np.ndarray):
        raise ValueError('"X" must be a numpy.array.')
    Nx = X.shape
    N = Nx[0]
    M = Nx[1]

    if not isinstance(Y, np.ndarray):
        raise ValueError('"Y" must be a numpy.array.')
    Ny = Y.shape

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
    # Intialize variables:
    mvd = [np.nan]*R
    spread = [np.nan]*R
    irr = [np.nan]*R

    for j in range(R): # loop on sample sizes
        Xj, idx_new = lhcube_shrink(X, NN[j]) # drop rows while trying to
        #  maximize the spread between the points

        if len(Ny) == 1:
            Yj = Y[idx_new]
        else:
            Yj = Y[idx_new, :]

        mvd[j], spread[j], irr[j], _ = \
            RSA_indices_thres(Xj, Yj, threshold, Nboot)

    if Nboot <= 1: # return a numpy.ndarray (R, M)
        mvd_tmp = np.nan*np.ones((R, M))
        spread_tmp = np.nan*np.ones((R, M))
        irr_tmp = np.nan*np.ones((R, M))

        for j in range(R):
            mvd_tmp[j, :] = mvd[j]
            spread_tmp[j, :] = spread[j]
            irr_tmp[j, :] = irr[j]

        mvd = mvd_tmp
        spread = spread_tmp
        irr = irr_tmp

    return mvd, spread, irr


def RSA_plot_thres(X, idxb, n_col=5, X_Labels=[], str_legend=['behavioural', 'non-behavioural']):

    """ Plotting function for Regional Sensitivity Analysis.
    Plot the CDF of the samples in dataset X that satisfy a given condition
    ('behavioural'), and the CDF of the samples that do not satisfy
    the condition ('non-behavioural').
    (see help of RSA_indices_thres for details about RSA and references)

    Usage:

        RSA_thres.RSA_plot_thres(X, idxb, n_col=5, X_Labels=[],
                                 str_legend['behavioural', 'non-behavioural'])

    Input:
             X = set of input samples                     - numpy.ndarray (N,M)
          idxb = indices of samples statisfying the       - numpy.ndarray (N, )
                 condition
         n_col = number of panels per row in the plot     - integer
                 (default: min(5, M))
      X_Labels = labels for the horizontal axis           - list (M elements)
                 (default: [' X1','X2',...])
    str_legend = text for legend                          - list (2 elements)
                 (default:
                     ['behavioural','non-behavioural'])

    This function is part of the SAFE Toolbox by F. Pianosi, F. Sarrazin
    and T. Wagener at Bristol University (2015).
    SAFE is provided without any warranty and for non-commercial use only.
    For more details, see the Licence file included in the root directory
    of this distribution.
    For any comment and feedback, or to discuss a Licence agreement for
    commercial use, please contact: francesca.pianosi@bristol.ac.uk
    For details on how to cite SAFE in your publication, please see:
    https://www.safetoolbox.info"""

    # Options for the graphic:
    pltfont = {'fontname': 'DejaVu Sans', 'fontsize': 15} # font
    lwn = 2 # line width for non-behavioural records
    lwb = 2 # line width for behavioural records

    # Options for the colours:
    # You can produce a coloured plot or a black and white one
    # (printer-friendly).
    # Option 1 - coloured plot: uncomment the following 2 lines:
    lcn = 'b'  # line colour for non-behavioural
    lcb = 'r'  # line colour for behavioural
    # Option 2 - B&W plot: uncomment the following 2 lines:
    #lcn =  [150 150 150]/256 ; % line colour
    #lcb = 'k' ; % line colour for behavioural

    ###########################################################################
    # Check inputs
    ###########################################################################
    if not isinstance(X, np.ndarray):
        raise ValueError('"X" must be a numpy.array.')
    if X.dtype.kind != 'f' and X.dtype.kind != 'i' and X.dtype.kind != 'u':
        raise ValueError('"X" must contain floats or integers.')

    if not isinstance(idxb, np.ndarray):
        raise ValueError('"idxb" must be a numpy.array.')
    if idxb.dtype != 'bool':
        raise ValueError('"idxb" must contain booleans.')

    Nx = X.shape
    N = Nx[0]
    M = Nx[1]
    idxb = idxb.flatten() # shape (N, )
    Ni = idxb.shape

    if Ni[0] != N:
        raise ValueError('""X"" and  "idxb" must be have the same number of rows')

    ###########################################################################
    # Check optional inputs
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

    if not isinstance(str_legend, list):
        raise ValueError('"str_legend" must be a list with 2 elements.')
    if not all(isinstance(i, str) for i in str_legend):
        raise ValueError('Elements in "str_legend" must be strings.')
    if len(str_legend) != 2:
        raise ValueError('"str_legend" must have 2 elements.')

    ###########################################################################
    # Plot
    ###########################################################################
    n_col = min(n_col, M) # n_col <= M
    n_row = int(np.ceil(M/n_col))

    # Recove behavioural and non-behavioural input sets:
    Xb = X[idxb, :]
    Xnb = X[~idxb, :]

    plt.figure()

    for i in range(M): # loop over inputs

        # Approximate behavioural and non-behavioural CDFs:
        xx = np.unique(sorted(X[:, i]))
        CDFb = empiricalcdf(Xb[:, i], xx)
        CDFnb = empiricalcdf(Xnb[:, i], xx)

        # Plot CDFs:
        plt.subplot(n_row, n_col, i+1)
        plt.plot(xx, CDFb, color=lcb, linewidth=lwb)
        plt.plot(xx, CDFnb, color=lcn, linewidth=lwn)

        # Customize plot:
        plt.xlabel(X_Labels[i], **pltfont)
        plt.axis((np.min(X[:, i]), np.max(X[:, i]), 0, 1))
        plt.xticks(**pltfont)
        plt.yticks(**pltfont)
        if ((i+1) % n_col) == 1: # first column
            plt.ylabel('cdf', **pltfont)
        if i == 0:
            plt.legend(str_legend)
        plt.box(on=True)
