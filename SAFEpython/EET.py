"""
    Module to perform the Elementary Effect Test (EET) or method of Morris

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
    Morris, M.D. (1991), Factorial sampling plans for preliminary
    computational experiments, Technometrics, 33(2).

    Saltelli, A., et al. (2008), Global Sensitivity Analysis, The Primer,
    Wiley.

    Campolongo, F., Cariboni, J., Saltelli, A. (2007), An effective
    screening design for sensitivity analysis of large models. Environ. Model.
    Softw. 22 (10), 1509-1518.
"""
from __future__ import division, absolute_import, print_function

from warnings import warn
import numpy as np
from numpy.matlib import repmat
import matplotlib.pyplot as plt

def EET_indices(r, xmin, xmax, X, Y, design_type, Nboot=0):

    """Compute the sensitivity indices according to the Elementary Effects Test
    (Saltelli, 2008) or 'method of Morris' (Morris, 1991).
    These are: the mean (mi) of the Elementary Effects (EEs) associated to
    input 'i', which measures the input influence; and the standard deviation
    (sigma) of the EEs, which measures its level of interactions with other
    inputs.
    For the mean EE, we use the version suggested by Campolongo et al.
    (2007), where absolute values of the EEs are used (this is to avoid that
    EEs with opposite sign would cancel each other out).

    Usage:
     mi, sigma, EE = EET.EET_indices(r, xmin, xmax, X, Y, design_type, Nboot=0)


    Input:
              r = number of sampling points          - scalar
           xmin = lower bounds of input ranges       - list (M elements)
           xmax = upper bounds of input ranges       - list (M elements)
              X = matrix of sampling datapoints      - numpy.ndarray(r*(M+1),M)
                  where EE must be computed
              Y = associated output values           - numpy.ndarray(r*(M+1), )
                                                  or - numpy.ndarray(r*(M+1),1)
      des_type = design type                         - string
                 Options: 'trajectory', 'radial'

    Optional input:
          Nboot = number of resamples used for       - integer
                  boostrapping(if not specified:
                  Nboot=0, i.e. no bootstrapping)
    Output:
             mi = mean of absolute elementary       if Nboot <= 1:
                  effects estimated for each        - numpy.ndarray (M, )
                  bootstrap resample when           if Nboot > 1:
                  bootstrapping is used (Nboot>1)   - numpy.ndarray (Nboot,M)
         sigma = standard deviation of elementary   if Nboot <= 1:
                  effects estimated for each        - numpy.ndarray (M, )
                  bootstrap resample when           if Nboot > 1:
                  bootstrapping is used (Nboot>1)   - numpy.ndarray (Nboot,M)
         EE = matrix of 'r' elementary effects      - numpy.ndarray(r,M)

    NOTE: If the vector Y includes any NaN values, the function will
    identify them and exclude them from further computation. A Warning message
    about the number of discarded NaN elements (and hence the actual number
    of samples used for estimating mi and sigma) will be displayed.

    REFERENCES:

    Morris, M.D. (1991), Factorial sampling plans for preliminary
    computational experiments, Technometrics, 33(2).

    Saltelli, A., et al. (2008), Global Sensitivity Analysis, The Primer,
    Wiley.

    Campolongo, F., Cariboni, J., Saltelli, A. (2007), An effective
    screening design for sensitivity analysis of large models. Environ. Model.
    Softw. 22 (10), 1509-1518.

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
        raise ValueError('all components of "xmax" must be higher than the' +
                         'corresponding ones in "xmin"')

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
        raise ValueError('input "X" must have shape (N, M)')
    if Nx[1] != M:
        raise ValueError('input "X" must have M columns')
    if Nx[0] != r*(M+1):
        raise ValueError('input "X" must have r*(M+1) rows')

    if len(Ny) == 2:
        if Ny[1] != 1:
            raise ValueError('"Y" must be of size (N, ) or (N,1).')
        Y = Y.flatten()
    elif len(Ny) != 1:
        raise ValueError('"Y" must be of size (N, ) or (N,1).')
    if Ny[0] != Nx[0]:
        raise ValueError('The number of elements in "Y" must be equal to the number of rows in X.')

    ###########################################################################
    # Check optional inputs
    ###########################################################################
    if not isinstance(Nboot, (int, np.int8, np.int16, np.int32, np.int64)):
        raise ValueError('"Nboot" must be scalar and integer.')
    if Nboot < 0:
        raise ValueError('"Nboot" must be positive.')

    ###########################################################################
    # Compute elementary effects
    ###########################################################################
    EE = np.nan * np.ones((r, M)) # array of elementary effects
    k = 0
    ki = 0 # index of the first element of the block for the current elementary effect

    for i in range(r):
        for j in range(M):

            if design_type == 'radial': # radial design: EE is the difference
            # between output at one point in the i-th block and output at
            # the 1st point in the block
                EE[i, j] = (Y[k+1]-Y[ki]) / (X[k+1, j]-X[ki, j]) * Dr[j]

            elif design_type == 'trajectory': #  trajectory design: EE is the difference
            # between output at one point in the i-th block and output at
            # the previous point in the block (the "block" is indeed a
            # trajectory in the input space composed of points that
            # differ in one component at the time)
                idx = abs(X[k+1, :]-X[k, :]) > 0
                # if using 'morris' sampling, the points in the block may not
                # be in the proper order, i.e. each point in the block differs
                # from the previous/next one by one component but we don't know
                # which one; this is here computed and saved in 'idx'
                if sum(idx) == 0:
                    raise ValueError('X[%d,:] and X[%d,:] are equal' % (k, k+1))
                if sum(idx) > 1:
                    raise ValueError('X[%d,:] and X[%d,:] differ in more ' % (k, k+1) +
                                     'than one component')
                EE[i, idx] = (Y[k+1]-Y[k]) / (X[k+1, idx]-X[k, idx]) * Dr[idx]
            else:
                raise ValueError('"design_type" must be one among ["radial",  "trajectory"]')

            k = k + 1

        k = k + 1
        ki = k

    ###########################################################################
    # Compute mean and standard deviation
    ###########################################################################

    if Nboot > 1:
        bootsize = r # size of bootstrap resamples
        B = np.random.randint(r, size=(bootsize, Nboot))

        mi = np.nan * np.ones((Nboot, M))
        sigma = np.nan * np.ones((Nboot, M))
        idx_EE = np.nan * np.ones((Nboot, M))

        for n in range(Nboot):
            mi[n, :], sigma[n, :], idx_EE[n, :] = compute_indices(EE[B[:, n], :])

        # Print to screen a warning message if any NAN was found in Y
        if np.sum(np.isnan(Y)):
            warn('\n%d NaNs were found in Y' % np.sum(np.isnan(Y)))

            str_EE = "\nX%d: %1.0f" % (1, np.mean(idx_EE[:, 0]))
            for i in range(1, M):
                str_EE = str_EE + "\nX%d: %1.0f" % (i+1, np.mean(idx_EE[:, i]))

            print('\nAverage number of samples that could be used to evaluate mean ' +
                  'and standard deviation of elementary effects is:')
            print(str_EE)
            print('\n')

    else:
        mi, sigma, idx_EE = compute_indices(EE)

       # Print to screen a warning message if any NAN was found in Y
        if np.sum(np.isnan(Y)):
            warn('\n%d NaNs were found in Y' % np.sum(np.isnan(Y)))

            str_EE = "\nX%d: %1.0f" % (1, idx_EE[0])
            for i in range(1, M):
                str_EE = str_EE + "\nX%d: %1.0f" % (i+1, idx_EE[i])

            print('\nNumber of samples that could be used to evaluate mean ' +
                  'and standard deviation of elementary effects is:')
            print(str_EE)
            print('\n')

    return mi, sigma, EE


def compute_indices(EE):

    """This function computes the sensitivity indices according to the
    Elementary Effects Test (Saltelli, 2008) or 'method of Morris'
    (Morris, 1991) for ONE sample/bootstrap resample.

    This function is called internally in EET.EET_indices.

    Usage:
        mi, sigma, idx_EE = EET.compute_indices(EE)

    Input:
        EE = matrix of 'r' elementary effects (EEs)        - numpy.ndarray(r,M)

    Output:
        mi = mean of absolute elementary                   - numpy.ndarray(M, )
             effects
     sigma = standard deviation of the                     - numpy.ndarray(M, )
                  elementary effects
    idx_EE = array that indicates the number of NaNs       - numpy.ndarray(M, )
             for each of the M columns of EE.

    NOTE: If the vector Y includes any NaN values, the function will
    identify them and exclude them from further computation.

    For reference and further details see help of EET.EET_indices.

    This function is part of the SAFE Toolbox by F. Pianosi, F. Sarrazin
    and T. Wagener at Bristol University (2015).
    SAFE is provided without any warranty and for non-commercial use only.
    For more details, see the Licence file included in the root directory
    of this distribution.
    For any comment and feedback, or to discuss a Licence agreement for
    commercial use, please contact: francesca.pianosi@bristol.ac.uk
    For details on how to cite SAFE in your publication, please see:
    https://www.safetoolbox.info/"""

    idx_EE = np.sum(~np.isnan(EE), axis=0) # find number of NaN in EE
    mi = np.nanmean(abs(EE), axis=0) # mean absolute value of EE (excluding NaNs)
    sigma = np.nanstd(EE, axis=0) # std of EE (excluding NaNs)

    return mi, sigma, idx_EE


def EET_convergence(EE, rr, Nboot=0):

    """Compute mean and standard deviation of Elementary Effects
    using sub-samples of the original sample 'EE' of increasing size
    (see help of EET.EET_indices for more details about the EET and references)

    Usage:
        mi, sigma = EET.EET_convergence(EE, rr, Nboot=0)

    Input:
       EE = matrix of 'r' elementary effects (EEs)         - numpy.ndarray(r,M)
       rr = reduced number of EEs at which indices will be - numpy.ndarray(R, )
           estimated (must be a vector of positive integer
           values not exceeding 'r')

    Optional input:
    Nboot = number of resamples used for boostrapping      - integer
            (if not specified: Nboot=0, i.e. no
            bootstrapping)

    Output:
      mi = mean of absolute EEs at different sample sizes if Nboot <= 1:
           When bootstrapping is used (i.e. Nboot>1),     - numpy.ndarray (R,M)
           mi[j] is a numpy.ndarray(Nboot,M) of the       if Nboot > 1:
           Nboot estimates of 'spread' for the jth        - list(R elements)
           sample size
   sigma = mean of absolute EEs at different sample sizes if Nboot <= 1:
           When bootstrapping is used (i.e. Nboot>1),     - numpy.ndarray (R,M)
           sigma[j] is a numpy.ndarray(Nboot,M) of the    if Nboot > 1:
           Nboot estimates of 'spread' for the jth        - list(R elements)
           sample size

    NOTE: If the vector EE includes any NaN values, the function will
    identify them and exclude them from further computation. A Warning message
    about the number of discarded NaN elements (and hence the actual number
    of samples used for estimating mi and sigma) will be displayed.

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
    if not isinstance(EE, np.ndarray):
        raise ValueError('"EE" must be a numpy.array.')
    if EE.dtype.kind != 'f' and EE.dtype.kind != 'i' and EE.dtype.kind != 'u':
        raise ValueError('"EE" must contain floats or integers.')

    NE = EE.shape
    r = NE[0]
    M = NE[1]

    if not isinstance(rr, np.ndarray):
        raise ValueError('"rr" must be a numpy.array.')
    if rr.dtype.kind != 'i':
        raise ValueError('"rr" must contain integers.')
    if any(i < 0 for i in np.diff(rr)):
        raise ValueError('elements in "rr" must be sorted in ascending order')
    if any(i < 0 for i in rr):
        raise ValueError('elements in "rr" must be positive')
    if rr[-1] > r:
        raise ValueError('Maximum value in "rr" must not exceed r = %d' % r)
    Nr = rr.shape
    if len(Nr) > 1:
        raise ValueError('"rr" must be of shape (R,).')

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

    R = len(rr)
    # Intialise variables
    if Nboot > 1:
        mi = [np.nan]*R
        sigma = [np.nan]*R
    else:
        mi = np.nan * np.ones((R, M))
        sigma = np.nan * np.ones((R, M))

    for j in range(R): # loop on sample sizes
        # Option 1: randomly select a subset of NN(j) without replacement
        # (which is our recommendation in particular for small sample size)
        replace = False
        # Option 2: resample with replacement
        #replace = True
        idx = np.random.choice(np.arange(0, r, 1), size=(rr[j], ), replace=replace)

        EEj = EE[idx,:] # extract the selected sample

        if Nboot > 1:
            bootsize = rr[j] # size of bootstrap resamples
            B = np.random.randint(rr[j], size=(bootsize, Nboot))
            mi[j] = np.nan * np.ones((Nboot, M))
            sigma[j] = np.nan * np.ones((Nboot, M))
            idx_EEj = np.nan * np.ones((Nboot, M))

            for n in range(Nboot):
                mi[j][n, :], sigma[j][n, :], idx_EEj[n, :] = \
                compute_indices(EEj[B[:, n], :])

            # Print to screen a warning message if any NAN was found in EE
            if np.sum(np.isnan(EE)):
                warn('\n%d NaNs were found in EE' % np.sum(np.isnan(EE)))

                str_EE = "\nX%d: %1.0f" % (1, np.mean(idx_EEj[:,0]))
                for i in range(1, M):
                    str_EE = str_EE + "\nX%d: %1.0f" % (i+1, np.mean(idx_EEj[:,i]))

                print('\nAverage number of samples that could be used to evaluate mean ' +
                      'and standard deviation of elementary effects is:')
                print(str_EE)
                print('\n')

        else:
            mi[j, :], sigma[j, :], idx_EEj = compute_indices(EEj)

             # Print to screen a warning message if any NAN was found in EE
            if np.sum(np.isnan(EE)):
                warn('\n%d NaNs were found in EE' % np.sum(np.isnan(EE)))

                str_EE = "\nX%d: %1.0f" % (1, idx_EEj[0])
                for i in range(1, M):
                    str_EE = str_EE + "\nX%d: %1.0f" % (i+1, idx_EEj[i])

                    print('\nNumber of samples that could be used to evaluate mean ' +
                          'and standard deviation of elementary effects is:')
                    print(str_EE)
                    print('\n')

    return mi, sigma

def EET_plot(mi, sigma, labelinput=[], mi_lb=np.array([]),
             mi_ub=np.array([]), sigma_lb=np.array([]), sigma_ub=np.array([])):

    """ Plot the sensitivity indices computed by the Elementary Effects Test -
    mean (mi) of the EEs on the horizontal axis and standard deviation (sigma)
    of the EEs on the vertical axis.
    (see help of EET_indices for more details about the EET and references)

    Usage:

    EET.EET_plot(mi, sigma)
    EET.EET_plot(mi, sigma, labelinput)
    EET.EET_plot(mi, sigma, labelinput=[], mi_lb=np.array([]),
                 mi_ub=np.array([]), sigma_lb=np.array([]), sigma_ub=np.array([]))

    Inputs:
            mi = mean of the elementary effects             - numpy.ndarray(M,)
         sigma = standard deviation of the elementary       - numpy.ndarray(M,)
                 effects

    Optional inputs:
    labelinput = strings for the x-axis labels              - list(M)
         mi_lb = lower bound of 'mi'                        - numpy.ndarray(M,)
         mi_ub = upper bound of 'mi'                        - numpy.ndarray(M,)
      sigma_lb = lower bound of 'sigma'                     - numpy.ndarray(M,)
      sigma_ub = upper bound of 'sigma'                     - numpy.ndarray(M,)

    This function is part of the SAFE Toolbox by F. Pianosi, F. Sarrazin
    and T. Wagener at Bristol University (2015).
    SAFE is provided without any warranty and for non-commercial use only.
    For more details, see the Licence file included in the root directory
    of this distribution.
    For any comment and feedback, or to discuss a Licence agreement for
    commercial use, please contact: francesca.pianosi@bristol.ac.uk
    For details on how to cite SAFE in your publication, please see:
    https://www.safetoolbox.info/"""

    # Options for the graphic
    pltfont = {'fontname': 'DejaVu Sans', 'fontsize': 15} # font for axes
    pltfont_leg = {'family': 'DejaVu Sans', 'size': 15} # font for legend
    ms = 10 # Marker size

    # Options for the legend
    sorting = 1 # If 1, inputs will be displayed in the legend
    # according to their influence, i.e. from most sensitive to least sensitive
    # (if 0 they will be displayed according to their original order)
    nb_legend = 13  # number of input names that will be displayed in the legend

    # Options for the colours:
    # You can produce a coloured plot or a black and white one
    # (printer-friendly). Furthermore, you can use matplotlib colourmaps or
    # repeat 5 'easy-to-distinguish' colours (see http://colorbrewer2.org/).
    # The variable 'col' must be a np.ndarray
    # Option 1a - coloured using colorbrewer: uncomment the following lines:
    col = np.array([[228, 26, 28], [55, 126, 184], [77, 175, 74],
                    [152, 78, 163], [255, 127, 0]])/256
    cc = 'k'
    # Option 1b - coloured using matplotlib colormap: uncomment the following line:
    # colorscale = 'jet'
    # col =  eval('plt.cm.'+colorscale)(np.linspace(0,1,5))
    # cc = 'k'
    # Option 1a - B&W using matlab colorbrewer: uncomment the following lines:
    # col = np.array([37 37 37],[90 90 90],[150 150 150],[189 189 189],[217 217 217]])/256
    # Option 1b - B&W using matlab colormap: uncomment the following line:
    # colorscale = 'grey'
    # col =  eval('plt.cm.'+colorscale)(np.linspace(0,1,5))
    # cc = 'w'

    ###########################################################################
    # Check inputs
    ###########################################################################
    if not isinstance(mi, np.ndarray):
        raise ValueError('"mi" must be a numpy.array.')
    if mi.dtype.kind != 'f' and mi.dtype.kind != 'i' and mi.dtype.kind != 'u':
        raise ValueError('"mi" must contain floats or integers.')
    if not isinstance(sigma, np.ndarray):
        raise ValueError('"sigma" must be a numpy.array.')
    if sigma.dtype.kind != 'f' and sigma.dtype.kind != 'i' and sigma.dtype.kind != 'u':
        raise ValueError('"sigma" must contain floats or integers.')

    Nm = mi.shape
    if len(Nm) > 1:
        raise ValueError('"mi" must be of size (M, ).')
    M = Nm[0]
    Ns = sigma.shape
    if len(Ns) > 1:
        raise ValueError('"sigma" must be of size (M, ).')
    if Ns[0] != M:
        raise ValueError('"mi" and "sigma" must have the same number of elements')

    ###########################################################################
    # Check optional inputs
    ###########################################################################
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

    if len(mi_lb) != 0 or np.isnan(mi_lb).any():
        if not isinstance(mi_lb, np.ndarray):
            raise ValueError('"mi_lb" must be a numpy.array.')
        if mi_lb.dtype.kind != 'f' and mi_lb.dtype.kind != 'i' and mi_lb.dtype.kind != 'u':
            raise ValueError('"mi_lb" must contain floats or integers.')
        mi_lb = mi_lb.flatten()
        Nm_lb = mi_lb.shape
        if Nm_lb[0] != M:
            raise ValueError('"mi" and "mi_lb" must have the same number of elements')
        if (mi_lb-mi > 0).any():
            raise ValueError('"mi_lb" must be lower or equal to mi.')
    else:
        mi_lb = np.array([0]*M)

    if len(mi_ub) != 0 or np.isnan(mi_ub).any():
        if not isinstance(mi_ub, np.ndarray):
            raise ValueError('"mi_ub" must be a numpy.array.')
        if mi_ub.dtype.kind != 'f' and mi_ub.dtype.kind != 'i' and mi_ub.dtype.kind != 'u':
            raise ValueError('"mi_ub" must contain floats or integers.')

        if (mi_ub-mi < 0).any():
            raise ValueError('"mi_ub" must be higher or equal to mi.')
        mi_ub = mi_ub.flatten()
        Nm_ub = mi_ub.shape
        if Nm_ub[0] != M:
            raise ValueError('"mi" and "mi_ub" must have the same number of elements')
    else:
        mi_ub = np.array([0]*M)

    if len(sigma_lb) != 0 or np.isnan(sigma_lb).any():
        if not isinstance(sigma_lb, np.ndarray):
            raise ValueError('"sigma_lb" must be a numpy.array.')
        if sigma_lb.dtype.kind != 'f' and sigma_lb.dtype.kind != 'i' and \
           sigma_lb.dtype.kind != 'u':
            raise ValueError('"sigma_lb" must contain floats or integers.')
        sigma_lb = sigma_lb.flatten()
        Ns_lb = sigma_lb.shape
        if Ns_lb[0] != M:
            raise ValueError('"sigma" and "sigma_lb" must have the same number of elements')
        if (sigma_lb-sigma > 0).any():
            raise ValueError('"sigma_lb" must be lower or equal to sigma.')
    else:
        sigma_lb = np.array([0]*M)

    if len(sigma_ub) != 0 or np.isnan(sigma_ub).any():
        if not isinstance(sigma_ub, np.ndarray):
            raise ValueError('"sigma_ub" must be a numpy.array.')
        if sigma_ub.dtype.kind != 'f' and sigma_ub.dtype.kind != 'i' and \
           sigma_ub.dtype.kind != 'u':
            raise ValueError('"sigma_ub" must contain floats or integers.')

        if (sigma_ub-sigma < 0).any():
            raise ValueError('"sigma_ub" must be higher or equal to sigma.')
        sigma_ub = sigma_ub.flatten()
        Ns_ub = sigma_ub.shape
        if Ns_ub[0] != M:
            raise ValueError('"sigma" and "sigma_ub" must have the same number of elements')
    else:
        sigma_ub = np.array([0]*M)

    ###########################################################################
    # Produce plot
    ###########################################################################
    A = len(col)
    L = int(np.ceil(M/A))
    clrs = repmat(col, L, 1)

    labelinput_new = [np.nan]*M

    if sorting:
        Sidx = np.flip(np.argsort(mi), axis=0)
        mi = mi[Sidx]
        sigma = sigma[Sidx]
        for i in range(M):
            labelinput_new[i] = labelinput[Sidx[i]]
        if len(mi_ub) != 0:
            mi_ub = mi_ub[Sidx]
        if len(mi_lb) != 0:
            mi_lb = mi_lb[Sidx]
        if len(sigma_ub) != 0:
            sigma_ub = sigma_ub[Sidx]
        if len(sigma_lb) != 0:
            sigma_lb = sigma_lb[Sidx]

    if nb_legend < M:
        labelinput_new = labelinput_new[0:nb_legend]
        labelinput_new[-1] = labelinput_new[-1] + '...'

    plt.figure()

    # First plot EEs mean & std as circles:
    for i in range(M):
        plt.plot(mi[i], sigma[i], 'ok', markerfacecolor=clrs[i], markersize=ms,
                 markeredgecolor='k')

    # plot first the larger confidence areas
    size_bounds = mi_ub - mi_lb
    idx = np.flip(np.argsort(size_bounds), axis=0)
    for i in range(M): # add rectangular shape
        plt.fill([mi_lb[idx[i]], mi_lb[idx[i]], mi_ub[idx[i]], mi_ub[idx[i]]],
                 [sigma_lb[idx[i]], sigma_ub[idx[i]], sigma_ub[idx[i]],
                  sigma_lb[idx[i]]], color=clrs[idx[i]])

    # Plot again the circles (in case some have been overriden by the rectangles
    # representing confidence bounds)
#    for i in range(M):
#        plt.plot(mi[i], sigma[i],'ok',markerfacecolor=clrs[i],markersize=ms,markeredgecolor='k')

    # Create legend:
    plt.legend(labelinput_new, loc='upper right', prop=pltfont_leg)

    plt.xlabel('Mean of EEs', **pltfont)
    plt.ylabel('Standard deviation of EEs', **pltfont)
    plt.grid(linestyle='--')
    plt.xticks(**pltfont)
    plt.yticks(**pltfont)
