"""
    Module to perform Regional Sensitivity Analysis (RSA) based on grouping

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

    Wagener, T., Boyle, D. P., Lees, M. J., Wheater, H. S., Gupta, H. V.,
    and Sorooshian, S. (2001): A framework for development and application of
    hydrological models, Hydrol. Earth Syst. Sci., 5, 13-26.
"""

from __future__ import division, absolute_import, print_function

from warnings import warn
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from SAFEpython.util import empiricalcdf, split_sample

def RSA_indices_groups(X, Y, ngroup=10, Nboot=0):

    """ Computation function for Regional Sensitivity Analysis with grouping
    (as first proposed by Wagener et al., 2001). The function can handle
    discrete outputs.

    The function splits the samples in a dataset X into 'ngroup' sub-sets
    corresponding to 'ngroup' of equal size based on the value of Y (i.e.
    'equiprobable' groups).
    Then it assesses the distance (i.e. maximum vertical distance called 'mvd'
    and area between CDFs called 'spread') between pairs of CDFs of X in the
    different sub-sets. It aggregates the values using a statistic (median,
    mean and maximum) e.g. for mvd the function computes:

       mvd_median = median( max( | Fi(x) - Fj(x) | ) )
                      i,j    x

       mvd_mean   = mean( max( | Fi(x) - Fj(x) | ) )
                    i,j    x

       mvd_max    = max( max( | Fi(x) - Fj(x) | ) )
                    i,j   x

    where Fi() is the CDF of X in the i-th group and Fj() is the CDF in the
    j-th group.

    See 'RSA_indices_thres' for more information about the sensitivity measures.
    See also 'RSA_plot_groups' on how to visualize results.

    Usage:
         mvd_median, mvd_mean, mvd_max, spread_median, spread_mean, spread_max,
         idx, Yk = RSA_groups.RSA_indices_groups(X, Y, ngroup=10, Nboot=0)

    Input:
               X = set of inputs samples              - numpy.ndarray (N,M)
               Y = set of output samples              - numpy.ndarray (N, )
                                                   or - numpy.ndarray (N,1)

   Optional input:
          ngroup = number of groups considered        - integer
                   (default: 10)

    Optional input:
           Nboot = number of resamples used for       - integer
                   boostrapping(if not specified:
                   Nboot=0, i.e. no bootstrapping)

    Output:
      mvd_median = median of mvd between pairs of     if Nboot <= 1:
                   inputs' CDFs estimated for the     - numpy.ndarray (M, )
                   different sub-sets and for each    if Nboot > 1:
                   bootstrap resample when            - numpy.ndarray (Nboot,M)
                   boostrapping is used
                   (i.e. Nboot > 1)
        mvd_mean = mean of mvd between pairs of       if Nboot <= 1:
                   inputs' CDFs estimated for the     - numpy.ndarray (M, )
                   different sub-sets and for each    if Nboot > 1:
                   bootstrap resample when            - numpy.ndarray (Nboot,M)
                   boostrapping is used
                   (i.e. Nboot > 1)
         mvd_max = mean of mvd between pairs of       if Nboot <= 1:
                   inputs' CDFs estimated for the     - numpy.ndarray (M, )
                   different sub-sets and for each    if Nboot > 1:
                   bootstrap resample when            - numpy.ndarray (Nboot,M)
                   boostrapping is used
                   (i.e. Nboot > 1)
    spread_median = median of spread between pairs of   if Nboot <= 1:
                   inputs' CDFs estimated for the     - numpy.ndarray (M, )
                   different sub-sets and for each    if Nboot > 1:
                   bootstrap resample when            - numpy.ndarray (Nboot,M)
                   boostrapping is used
                   (i.e. Nboot > 1)
     spread_mean = mean of spread between pairs of    if Nboot <= 1:
                   inputs' CDFs estimated for the     - numpy.ndarray (M, )
                   different sub-sets and for each    if Nboot > 1:
                   bootstrap resample when            - numpy.ndarray (Nboot,M)
                   boostrapping is used
                   (i.e. Nboot > 1)
     spread_max = maximum of spread between pairs of  if Nboot <= 1:
                   inputs' CDFs estimated for the     - numpy.ndarray (M, )
                   different sub-sets and for each    if Nboot > 1:
                   bootstrap resample when            - numpy.ndarray (Nboot,M)
                   boostrapping is used
                   (i.e. Nboot > 1)
             idx = respective groups of the samples   - numpy.ndarray (N, )
                   You can easily derive the n_groups
                   datasets {Xi} as:
                       Xi = X[idx == i]
              Yk = range of Y in each group           - numpy.ndarray
                                                                   (ngroup+1, )

    NOTES:

    - When Y is discrete and when the number of values taken by Y (ny) is
      lower than the prescribed number of groups (ngroup), a group is created
      for each value of Y (and therefore the number of groups is set to ny).

    - The function ensures that values of Y that are repeated several times
      belong to the same group. This may lead to a final number of group lower
      than ngroup and to a different number of data points across the groups.

    REFERENCES

    Wagener, T., Boyle, D. P., Lees, M. J., Wheater, H. S., Gupta, H. V.,
    and Sorooshian, S. (2001): A framework for development and application of
    hydrological models, Hydrol. Earth Syst. Sci., 5, 13-26.

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
    Y = Y.flatten() # shape (N, )
    Ny = Y.shape
    N = Nx[0]
    M = Nx[1]

    if Ny[0] != N:
        raise ValueError('input "X" and "Y" must have the same number of rows')

    ###########################################################################
    # Check optional inputs
    ###########################################################################
    if not isinstance(ngroup, (int, np.int8, np.int16, np.int32, np.int64)):
        raise ValueError('"ngroup" must be scalar and integer.')
    if ngroup < 0:
        raise ValueError('"ngroup" must be positive.')

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

        mvd_median = np.nan * np.ones((Nboot, M))
        mvd_mean = np.nan * np.ones((Nboot, M))
        mvd_max = np.nan * np.ones((Nboot, M))
        spread_median = np.nan * np.ones((Nboot, M))
        spread_mean = np.nan * np.ones((Nboot, M))
        spread_max = np.nan * np.ones((Nboot, M))

        for n in range(Nboot):
            Xi = X[B[:, n], :]
            Yi = Y[B[:, n]]
            mvd, spread, _, _ = compute_indices(Xi, Yi, ngroup)

            # Calculate statistics across pairs of CDFs:
            mvd_median[n, :] = np.median(mvd, axis=0)
            mvd_mean[n, :] = np.mean(mvd, axis=0)
            mvd_max[n, :] = np.max(mvd, axis=0)
            spread_median[n, :] = np.median(spread, axis=0)
            spread_mean[n, :] = np.mean(spread, axis=0)
            spread_max[n, :] = np.max(spread, axis=0)

        # Last, let's call the function once more to obtain the vector
        # 'idx' of the respective groups of the sample and the vector Yk of the
        # range of values in the different groups (needed to be returned among
        # the output arguments):
        _, _, idx, Yk = compute_indices(X, Y, ngroup)

    else:
        mvd, spread, idx, Yk = compute_indices(X, Y, ngroup)

        # Calculate statistics across pairs of CDFs:
        mvd_median = np.median(mvd, axis=0)
        mvd_mean = np.mean(mvd, axis=0)
        mvd_max = np.max(mvd, axis=0)
        spread_median = np.median(spread, axis=0)
        spread_mean = np.mean(spread, axis=0)
        spread_max = np.max(spread, axis=0)

    return mvd_median, mvd_mean, mvd_max, spread_median, spread_mean, \
    spread_max, idx, Yk

def compute_indices(X, Y, ngroup):

    """ This function computes the sensitivity indices for Regional Sensitivity
    Analysis with grouping for ONE sample/bootstrap resample.

    This function is called internally in RSA_groups.RSA_indices_groups.

    Usage:
        mvd, spread, idx, Yk = RSA_groups.compute_indices(X, Y, ngroup)

    Input:
         X = set of inputs samples              - numpy.ndarray (N,M)
         Y = set of output samples              - numpy.ndarray (N, )
    ngroup = number of groups considered        - integer

    Output:
       mvd = mvd between pairs of inputs' CDFs  - numpy.ndarray(
             estimated for the different        ngroup_eff*(ngroup_eff-1)/2,M)
             sub-sets. ngroup_eff is the actual
             number of groups used.
    spread = spread between pairs of inputs'    - numpy.ndarray(
             CDFs estimated for the different   ngroup_eff*(ngroup_eff-1)/2,M)
             sub-sets
       idx = respective groups of the samples   - numpy.ndarray (N, )
             You can easily derive the n_groups
             datasets {Xi} as:
                    Xi = X[idx == i]
        Yk = range of Y in each group           - numpy.ndarray(ngroup+1, )

    Note:
    The function ensures that values of Y that are repeated several times
    belong to the same group. This may lead to a final number of group lower
    than ngroup and to a different number of data points across the groups
    ngroup_eff).

    For reference and further details see help of RSA_groups.RSA_indices_groups.

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
    M = Nx[1]

    # Split the output sample in equiprobable groups:
    idx, Yk, _, ngroup_eff = split_sample(Y, ngroup)

    if ngroup_eff < ngroup:
        warn("%d groups were used instead of %d so that " %(ngroup_eff, ngroup) +
             "values that are repeated several time belong to the same group")

    # Initialize arrays of indices
    mvd = np.nan*np.ones([int(ngroup_eff*(ngroup_eff-1)/2), M])
    spread = np.nan*np.ones([int(ngroup_eff*(ngroup_eff-1)/2), M])

    for i in range(M):
        # Approximate CDF of the i-th parameter for each group:
        L = len(np.unique(X[:, i]))
        CDF_ = np.nan * np.ones((L, ngroup_eff))
        xx = np.unique(sorted(X[:, i]))

        for j in range(ngroup_eff):
            CDF_[:, j] = empiricalcdf(X[idx == j, i], xx)

        # Compute the distance between the different CDFs
        count = 0
        for j in range(ngroup_eff):
            for k in range(j+1, ngroup_eff, 1):
                mvd[count, i] = np.max(abs(CDF_[:, j] - CDF_[:, k]))
                spread[count, i] = \
                np.trapz(np.max(np.stack((CDF_[:, j], CDF_[:, k]), axis=0), axis=0), x=xx) -\
                np.trapz(np.min(np.stack((CDF_[:, j], CDF_[:, k]), axis=0), axis=0), x=xx)
                count = count+1

    return mvd, spread, idx, Yk

def RSA_plot_groups(X, idx, Yk, n_col=5, X_Labels=[], legend_title='Y'):

    """ Plotting function for Regional Sensitivity Analysis with grouping.
    Plot 'Ng' CDFs of the samples in X with different colours.
    (see help of RSA_indices_groups for details about RSA and references)

    Usage:
          RSA_groups.RSA_plot_groups(X, idx, Yk, n_col=5, X_Labels=[],
                                     legend_title='Y')

    Input:
               X = set of input samples                   - numpy.ndarray (N,M)
             idx = index of group to which input samples  - numpy.ndarray (N, )
                   belong (integers)
              Yk = range of Y in each group                numpy.ndarray
                                                                   (ngroup+1, )

    Optional input:
           n_col = number of panels per row in the plot   - integer
                  (default: min(5, M))
        X_labels = labels for the horizontal axis         - list (M elements)
                  (default: [' X1','X2',...])
    legend_title = label for legend (default: 'Y')        - string

    This function is part of the SAFE Toolbox by F. Pianosi, F. Sarrazin
    and T. Wagener at Bristol University (2015).
    SAFE is provided without any warranty and for non-commercial use only.
    For more details, see the Licence file included in the root directory
    of this distribution.
    For any comment and feedback, or to discuss a Licence agreement for
    commercial use, please contact: francesca.pianosi@bristol.ac.uk
    For details on how to cite SAFE in your publication, please see:
    https://www.safetoolbox.info """

    # Options for the figure
    lw = 2 # line width
    pltfont = {'fontname': 'DejaVu Sans', 'fontsize': 15} # font
    # Colorscale
    colorscale = 'jet' # color plot
    # colorscale = 'gray' # balck and white plot
    # Text formating of colorbar's ticklabels
    ticklabels_form = '%6.1f' # float with 1 character after decimal point
    # ticklabels_form = '%d' # integer

    ###########################################################################
    # Check inputs
    ###########################################################################
    if not isinstance(X, np.ndarray):
        raise ValueError('"X" must be a numpy.array.')
    if X.dtype.kind != 'f' and X.dtype.kind != 'i' and X.dtype.kind != 'u':
        raise ValueError('"X" must contain floats or integers.')

    if not isinstance(idx, np.ndarray):
        raise ValueError('"idx" must be a numpy.array.')
    if idx.dtype.kind != 'i':
        raise ValueError('"idx" must contain integers.')

    Nx = X.shape
    N = Nx[0]
    M = Nx[1]
    Ni = idx.shape

    if len(Ni) > 1:
        raise ValueError('"idx" must be a numpy.ndarray(N, )')

    if Ni[0] != N:
        raise ValueError('""X"" and  "idx" must be have the same number of rows')

    if not isinstance(Yk, np.ndarray):
        raise ValueError('"Yk" must be a numpy.array.')
    if X.dtype.kind != 'f' and X.dtype.kind != 'i' and X.dtype.kind != 'u':
        raise ValueError('"X" must contain floats or integers.')
    Ny = Yk.shape
    if len(Ny) > 1:
        raise ValueError('"Yk" must be a numpy.ndarray(ngroup+1, )')
    ngroup = Ny[0] - 1

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

    if not isinstance(legend_title, str):
        raise ValueError('"str_legend" must be a string.')

    ###########################################################################
    # Plot
    ###########################################################################
    n_col = min(n_col, M) # n_col <= M
    n_row = int(np.ceil(M/n_col))

    # Set colour scale:
    cmap = mpl.cm.get_cmap(colorscale, ngroup) # return colormap
    fig = plt.figure()

    # Colobar label:s
    cb_labels = [np.nan]*(ngroup+1)
    for i in range(ngroup+1):
        cb_labels[i] = ticklabels_form % (Yk[i])

    for i in range(M): # loop over inputs
        xx = np.unique(sorted(X[:, i]))
        plt.subplot(n_row, n_col, i+1)

        for j in range(ngroup): # loop over groups

            # Compute empirical CDF
            CDFj = empiricalcdf(X[idx == j, i], xx)
            # Plot the CDF
            plt.plot(xx, CDFj, color=cmap(j), linewidth=lw)

        plt.xlabel(X_Labels[i], **pltfont)
        plt.xlim(xx[0], xx[-1])
        plt.ylim(0, 1)
        plt.xticks(**pltfont); plt.yticks(**pltfont)
        if ((i+1) % n_col) == 1: # first column
            plt.ylabel('cdf', **pltfont)
        plt.box(on=True)

        if i == M-1: # Create colorbar
            c = np.arange(0, ngroup + 1)/(ngroup) # xticks
            # Add axes for the colorbar:
            cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
            cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, orientation='vertical', ticks=c)
            cb.set_label(legend_title, **pltfont)
            cb.set_ticks(c)
            cb.set_ticklabels(cb_labels)
            cb.Fontname = pltfont['fontname']
            cb.ax.tick_params(labelsize=pltfont['fontsize'])
