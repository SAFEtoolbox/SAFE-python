"""
    Module to visualise the sensitivity analysis results

    This module is part of the SAFE Toolbox by F. Pianosi, F. Sarrazin and
    T. Wagener at Bristol University (2015).
    SAFE is provided without any warranty and for non-commercial use only.
    For more details, see the Licence file included in the root directory
    of this distribution.
    For any comment and feedback, or to discuss a Licence agreement for
    commercial use, please contact: francesca.pianosi@bristol.ac.uk
    For details on how to cite SAFE in your publication, please see:
    bristol.ac.uk/cabot/resources/safe-toolbox/

    Package version: SAFEpython_v0.0.0
"""
from __future__ import division, absolute_import, print_function

from warnings import warn
import numpy as np
from numpy.matlib import repmat
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from SAFEpython.util import empiricalcdf
#from statsmodels.distributions.empirical_distribution import ECDF

def boxplot1(S, X_Labels=[], Y_Label='Sensitivity',
             S_lb=np.array([]), S_ub=np.array([])):

    """ Plot a set of indices as coloured boxes. The height of the box is fixed
    (and thin) if the indices are given as deterministic values, and it is
    equal to the uncertainty interval if the indices are associated with a
    lower and upper bound.

    Usage:
         plot_functions.boxplot1(S, X_Labels=[], Y_Label='Sensitivity',
                                 S_lb=np.array([]), S_ub=np.array([]))

    Input:
           S = vector of indices                          - numpy.ndarray (M, )

    Optional input:
    X_Labels = strings for the x-axis labels              - list (M elements)
               (default: ['X1','X2',...,XM'])
     Y_Label = y-axis label  (default: 'Sensitivity')     - string
        S_lb = lower bound of 'S' (default: empty)        - numpy.ndarray (M, )
        S_ub = upper bound of 'S' (default: empty)        - numpy.ndarray (M, )

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
    dh = 0.40 # semi-width of the box
    dv = 0.01 # semi-height of the box for deterministic value (no bootstrap)
    dv = 0.005 # semi-height of the box for bootstrap mean

    # Options for the colours:
    ec = 'k' # color of edges
    # You can produce a coloured plot or a black and white one
    # (printer-friendly). Furthermore, you can use matplotlib colourmaps or
    # repeat 5 'easy-to-distinguish' colours (see http://colorbrewer2.org/).
    # The variable 'col' must be a np.ndarray
    # Option 1a - coloured using colorbrewer: uncomment the following line:
    col = np.array([[228, 26, 28], [55, 126, 184], [77, 175, 74],
                    [152, 78, 163], [255, 127, 0]])/256
    # Option 1b - coloured using matplotlib colormap: uncomment the following lines:
    # colorscale = plt.cm.jet
    # col =  colorscale(np.linspace(0, 1, 5))
    # Option 1a - B&W using matlab colorbrewer: uncomment the following line:
    # col = np.array([[37, 37, 37], [90, 90, 90], [150, 150, 150],
    #                [189, 189, 189], [217, 217, 217]])/256
    # Option 1b - B&W using matlab colormap: uncomment the following lines:
    # colorscale = plt.cm.gray
    # col =  colorscale(np.linspace(0, 1, 5))

    ###########################################################################
    # Check inputs
    ###########################################################################
    if not isinstance(S, np.ndarray):
        raise ValueError('"S" must be a numpy.array.')
    if S.dtype.kind != 'f' and S.dtype.kind != 'i' and S.dtype.kind != 'u':
        raise ValueError('"S" must contain floats or integers.')

    Ns = S.shape
    if len(Ns) > 1:
        raise ValueError('"S" must be of size (M, ).')
    M = Ns[0]

    ###########################################################################
    # Check optional inputs
    ###########################################################################

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

    if not isinstance(Y_Label, str):
        raise ValueError('"str_legend" must be a string.')

    if len(S_lb) != 0:
        if np.isnan(S_lb).any():
            S_lb = np.array([])
        else:
            if not isinstance(S_lb, np.ndarray):
                raise ValueError('"S_lb" must be a numpy.array.')
            if S_lb.dtype.kind != 'f' and S_lb.dtype.kind != 'i' and S_lb.dtype.kind != 'u':
                raise ValueError('"S_lb" must contain floats or integers.')
            S_lb = S_lb.flatten()
            Ns_lb = S_lb.shape
            if Ns_lb[0] != M:
                raise ValueError('"S" and "S_lb" must have the same number of elements')
            if (S_lb-S > 0).any():
                raise ValueError('"S_lb" must be lower or equal to S.')

    if len(S_ub) != 0:
        if np.isnan(S_ub).any():
            S_ub = np.array([])
        else:
            if not isinstance(S_ub, np.ndarray):
                raise ValueError('"S_ub" must be a numpy.array.')
            if S_ub.dtype.kind != 'f' and S_ub.dtype.kind != 'i' and S_ub.dtype.kind != 'u':
                raise ValueError('"S_ub" must contain floats or integers.')

            if (S_ub-S < 0).any():
                raise ValueError('"S_ub" must be higher or equal to S.')
            S_ub = S_ub.flatten()
            Ns_ub = S_ub.shape
            if Ns_ub[0] != M:
                raise ValueError('"S" and "S_ub" must have the same number of elements')

    ###########################################################################
    # Produce plots
    ###########################################################################
    A = len(col)
    L = int(np.ceil(M/A))
    clrs = repmat(col, L, 1)

    # Plot on curent figure
    if plt.get_fignums(): # if there is a figure recover axes of current figure
        ax = plt.gca()
    else: # else create a new figure
        plt.figure()
        ax = plt.gca()

    for j in range(M):

        if len(S_lb) == 0: # no confidence intervals
            # Plot the value as a tick line:
            if ec == 'none':
                ax.add_patch(Rectangle((j+1-dh, S[j]-dv), 2*dh, 2*dv, color=clrs[j]))
            else:
                ax.add_patch(Rectangle((j+1-dh, S[j]-dv), 2*dh, 2*dv,
                                       facecolor=clrs[j], edgecolor=ec))
        else:
            # Plot the confidence interval as a rectangle:
            if ec == 'none':
                ax.add_patch(Rectangle((j+1-dh, S_lb[j]), 2*dh, S_ub[j]-S_lb[j],
                                       color=clrs[j]))
            else:
                ax.add_patch(Rectangle((j+1-dh, S_lb[j]), 2*dh, S_ub[j]-S_lb[j],
                                       facecolor=clrs[j], edgecolor=ec))
            # Plot the mean as a tick line:
            ax.add_patch(Rectangle((j+1-dh, S[j]-dv), 2*dh, 2*dv, color='black'))

    x1 = 0
    x2 = M+1

    if len(S_lb) != 0:
        y1 = min(-0.1, np.min(S_lb))
    else:
        y1 = min(-0.1, np.min(S))
    if len(S_ub) != 0:
        y2 = max(1.1, np.max(S_ub))
    else:
        y2 = max(1.1, np.max(S))

    plt.plot([x1, x2], [0, 0], ':k') # Plot zero line
    plt.xlim((x1, x2)) # set axes limits
    plt.ylim((y1, y2)) # set axes limits
    plt.xticks(np.arange(1, M+1, 1), X_Labels, **pltfont)
    plt.yticks(**pltfont)
    plt.ylabel(Y_Label, **pltfont)
    plt.grid(axis='x')


def boxplot2(S, X_Labels=[], Y_Label='Sensitivity',
             S_lb=np.array([]), S_ub=np.array([]), legend=[]):

    """ Plot a set of indices as coloured boxes. The height of the box is fixed
    (and thin) if the indices are given as deterministic values, and it is
    equal to the uncertainty interval if the indices are associated with a
    lower and upper bound.

    Usage:
         plot_functions.boxplot2(S, X_Labels=[], Y_Label='Sensitivity',
             S_lb=np.array([]), S_ub=np.array([]), legend=[])

    Input:
           S = vector of indices                          - numpy.ndarray (2,M)

    Optional input:
    X_Labels = strings for the x-axis labels              - list (M elements)
              (default: ['X1','X2',...,XM'])
     Y_Label = y-axis label  (default: 'Sensitivity')     - string
        S_lb = lower bound of 'S' (default: empty)        - numpy.ndarray (2,M)
        S_ub = upper bound of 'S' (default: empty)        - numpy.ndarray (2,M)
      legend = text for legend (name of the two types     - list of strings
               of sensitivity indices (default: empty)

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
    pltfont_leg = {'family': 'DejaVu Sans', 'size': 15} # font for legend
    dh = 0.40 # semi-width of the box
    dv = 0.01 # semi-height of the box for deterministic value (no bootstrap)
    dv = 0.005 # semi-height of the box for bootstrap mean

    # Options for the colours:
    ec1 = 'k' # color of edges
    ec2 = 'k' # color of edges
    # You can produce a coloured plot or a black and white one
    # (printer-friendly).
    # Option 1 - coloured using colorbrewer: uncomment the following lines:
    fc1 = [239/256, 138/256, 98/256]
    fc2 = [103/256, 169/256, 207/256]
    # Option 2 - B&W using matlab colorbrewer: uncomment the following lines:
    #fc1 = [150/256, 150/256, 150/256]
    #fc2 = [37/256, 37/256, 37/256]

    ###########################################################################
    # Check inputs
    ###########################################################################
    if not isinstance(S, np.ndarray):
        raise ValueError('"S" must be a numpy.array.')
    if S.dtype.kind != 'f' and S.dtype.kind != 'i' and S.dtype.kind != 'u':
        raise ValueError('"S" must contain floats or integers.')

    Ns = S.shape
    if len(Ns) != 2 or Ns[0] != 2:
        raise ValueError('"S" must be of shape (2,M).')
    M = Ns[1]

    ###########################################################################
    # Check optional inputs
    ###########################################################################

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

    if not isinstance(Y_Label, str):
        raise ValueError('"str_legend" must be a string.')

    if len(S_lb) != 0:
        if np.isnan(S_lb).any():
            S_lb = np.array([])
        else:
            if not isinstance(S_lb, np.ndarray):
                raise ValueError('"S_lb" must be a numpy.array.')
            if S_lb.dtype.kind != 'f' and S_lb.dtype.kind != 'i' and S_lb.dtype.kind != 'u':
                raise ValueError('"S_lb" must contain floats or integers.')
            Ns_lb = S_lb.shape
            if len(Ns_lb) != 2 or Ns_lb[0] != 2:
                raise ValueError('"S_lb" must be of shape (2,M).')
            if Ns_lb[1] != M:
                raise ValueError('"S" and "S_lb" must have the same number of elements')

    if len(S_ub) != 0:
        if np.isnan(S_ub).any():
            S_ub = np.array([])
        else:
            if not isinstance(S_ub, np.ndarray):
                raise ValueError('"S_ub" must be a numpy.array.')
            if S_ub.dtype.kind != 'f' and S_ub.dtype.kind != 'i' and S_ub.dtype.kind != 'u':
                raise ValueError('"S_ub" must contain floats or integers.')

            if (S_ub-S < 0).any():
                raise ValueError('"S_ub" must be higher or equal to S.')
            Ns_ub = S_ub.shape
            if len(Ns_ub) != 2 or Ns_ub[0] != 2:
                raise ValueError('"S_ub" must be of shape (2,M).')
            if Ns_ub[1] != M:
                raise ValueError('"S" and "S_ub" must have the same number of elements')

    if not isinstance(legend, list):
        raise ValueError('"legend" must be a list of strings.')
    if not all(isinstance(i, str) for i in legend):
        raise ValueError('Elements in "legend" must be strings.')

    ###########################################################################
    # Produce plots
    ###########################################################################
    # Option 1: plot on curent figure
    if plt.get_fignums(): # if there is a figure recover axes of current figure
        ax = plt.gca()
    else: # else create a new figure
        plt.figure()
        ax = plt.gca()
    # Option 2: create a new figure
    #plt.figure()
    #ax = plt.gca()

    for j in range(M):

        if len(S_lb) == 0: # no confidence intervals
            # Plot the value as a tick line:
            if ec1 == 'none':
                ax.add_patch(Rectangle((j+1-dh, S[0, j]-dv), dh-0.05, 2*dv,
                                       color=fc1))
            else:
                ax.add_patch(Rectangle((j+1-dh, S[0, j]-dv), dh-0.05, 2*dv,
                                       facecolor=fc1, edgecolor=ec1))

            if ec2 == 'none':
                ax.add_patch(Rectangle((j+1+0.05, S[1, j]-dv), dh-0.05, 2*dv,
                                       color=fc2))
            else:
                ax.add_patch(Rectangle((j+1+0.05, S[1, j]-dv), dh-0.05, 2*dv,
                                       facecolor=fc2, edgecolor=ec2))
        else:
            # Plot the confidence interval as a rectangle:
            if ec1 == 'none':
                ax.add_patch(Rectangle((j+1-dh, S_lb[0, j]), dh-0.05,
                                       S_ub[0, j]-S_lb[0, j], color=fc1))
            else:
                ax.add_patch(Rectangle((j+1-dh, S_lb[0, j]), dh-0.05,
                                       S_ub[0, j]-S_lb[0, j], facecolor=fc1, edgecolor=ec1))
            if ec2 == 'none':
                ax.add_patch(Rectangle((j+1+0.05, S_lb[1, j]), dh-0.05,
                                       S_ub[1, j]-S_lb[1, j], color=fc2))
            else:
                ax.add_patch(Rectangle((j+1+0.05, S_lb[1, j]), dh-0.05,
                                       S_ub[1, j]-S_lb[1, j], facecolor=fc2, edgecolor=ec2))
            # Plot the mean as a tick line:
            ax.add_patch(Rectangle((j+1-dh, S[0, j]-dv), dh-0.05, 2*dv, color='black'))
            ax.add_patch(Rectangle((j+1+0.05, S[1, j]-dv), dh-0.05, 2*dv, color='black'))

    if legend:
        plt.legend(legend, prop=pltfont_leg)

    x1 = 0
    x2 = M+1

    if len(S_lb) != 0:
        y1 = min(-0.1, np.min(S_lb))
    else:
        y1 = min(-0.1, np.min(S))
    if len(S_ub) != 0:
        y2 = max(1.1, np.max(S_ub))
    else:
        y2 = max(1.1, np.max(S))

    # plot vertical lines between one input and another
    for j in range(M+1):
        plt.plot([j+0.5, j+0.5], [y1, y2], '--k')

    plt.plot([x1, x2], [0, 0], ':k') # Plot zero line
    plt.xlim((x1, x2)) # set axes limits
    plt.ylim((y1, y2)) # set axes limits
    plt.xticks(np.arange(1, M+1, 1), X_Labels, **pltfont)
    plt.yticks(**pltfont)
    plt.ylabel(Y_Label, **pltfont)
    plt.grid(axis='x')


def plot_convergence(S, NN, S_lb=np.array([]), S_ub=np.array([]),
                     SExact=np.array([]), X_Label='Sample size',
                     Y_Label='Sensitivity', labelinput=[]):

    """Plots sequence of one or more indices estimated using samples
    of different sizes

    Usage:
    plot_functions.plot_convergence(S, NN, S_lb=np.array([]), S_ub=np.array([]),
                                    SExact=np.array([]), X_Label='Sample size',
                                    Y_Label='Sensitivity', labelinput=[])

    Input:
              S = sequence of estimates of M indices      - numpy.ndarray (R,M)
             NN = sample sizes at which indices           - numpy.ndarray (R, )
                  were estimated. NN must have at least    increasing integers
                  2 elements.

    Optional input:
           S_lb = lower bound of (uncertain) index        - numpy.ndarray (R,M)
                  estimates
           S_ub = upper bound of (uncertain) index        - numpy.ndarray (R,M)
                  estimates
         SExact = exact value of the indices (if known)   - numpy.ndarray (M, )
        X_Label = x-axis label (default: 'Sample size')   - string
        Y_Label = y-axis label   (default: 'Sensitivity)  - string
     labelinput = legend labels                           - list (M elements)
                  (default: ['X1','X2',...,XM'])

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
    pltfont = {'fontname': 'DejaVu Sans', 'fontsize': 15} # font for axes
    pltfont_leg = {'family': 'DejaVu Sans', 'size': 15} # font for legend
    # Options for the legend
    sorting = 1  # If 1, inputs will be displayed in the legend
    # according to their influence, i.e. from most sensitive to least sensitive
    # (if 0 they will be displayed according to their original order)
    nb_legend = 5 # number of input names that will be displayed in the legend
    end_length = 0.3 # adjust the space left for the legend

    # Options for the colours:
    # You can produce a coloured plot or a black and white one
    # (printer-friendly). Furthermore, you can use matplotlib colourmaps or
    # repeat 5 'easy-to-distinguish' colours (see http://colorbrewer2.org/).
    # The variable 'col' must be a np.ndarray
    # Option 1a - coloured using colorbrewer: uncomment the following lines:
    col = np.array([[228, 26, 28], [55, 126, 184], [77, 175, 74],
                    [152, 78, 163], [255, 127, 0]])/256
    # Option 1b - coloured using matplotlib colormap: uncomment the following line:
    # colorscale = plt.cm.jet
    # col =  colorscale(np.linspace(0, 1, 5))
    # Option 1a - B&W using matlab colorbrewer: uncomment the following lines:
      # col = np.array([[37, 37, 37], [90, 90, 90], [150, 150, 150],
    #                [189, 189, 189], [217, 217, 217]])/256
    # Option 1b - B&W using matlab colormap: uncomment the following line:
    # colorscale = plt.cm.gray
    # col =  colorscale(np.linspace(0, 1, 5))

    ###########################################################################
    # Check inputs
    ###########################################################################
    if not isinstance(S, np.ndarray):
        raise ValueError('"S" must be a numpy.array.')
    if S.dtype.kind != 'f' and S.dtype.kind != 'i' and S.dtype.kind != 'u':
        raise ValueError('"S" must contain floats or integers.')

    if not isinstance(NN, np.ndarray):
        raise ValueError('"NN" must be a numpy.array.')
    if NN.dtype.kind != 'i':
        raise ValueError('"NN" must contain integers.')
    if any(i < 0 for i in np.diff(NN)):
        raise ValueError('elements in "NN" must be sorted in ascending order')
    if any(i < 0 for i in NN):
        raise ValueError('elements in "NN" must be positive')
    NN_shape = NN.shape
    if len(NN_shape) > 1:
        raise ValueError('"NN" must be of shape (R,).')
    R = len(NN)
    if R <= 1:
        raise ValueError('"NN" must have at least 2 elements')

    Ns = S.shape
    if Ns[0] != R:
        raise ValueError('number of rows in "S" must be equal to the number of elements in "NN"')
    M = Ns[1]
    ###########################################################################
    # Check optional inputs
    ###########################################################################
    if len(S_lb) != 0:
        if not isinstance(S_lb, np.ndarray):
            raise ValueError('"S_lb" must be a numpy.array.')
        if S_lb.dtype.kind != 'f' and S_lb.dtype.kind != 'i' and S_lb.dtype.kind != 'u':
            raise ValueError('"S_lb" must contain floats or integers.')
        Ns_lb = S_lb.shape
        if Ns_lb[0] != R:
            raise ValueError('"S" and "S_lb" must have the same number of rows')
        if Ns_lb[1] != M:
            raise ValueError('"S" and "S_lb" must have the same number of colums')

    if len(S_ub) != 0:
        if not isinstance(S_ub, np.ndarray):
            raise ValueError('"S_ub" must be a numpy.array.')
        if S_ub.dtype.kind != 'f' and S_ub.dtype.kind != 'i' and S_ub.dtype.kind != 'u':
            raise ValueError('"S_ub" must contain floats or integers.')
        Ns_ub = S_ub.shape
        if Ns_ub[0] != R:
            raise ValueError('"S" and "S_ub" must have the same number of rows')
        if Ns_ub[1] != M:
            raise ValueError('"S" and "S_ub" must have the same number of colums')

    if len(SExact) != 0:
        if not isinstance(SExact, np.ndarray):
            raise ValueError('"SExact" must be a numpy.array.')
        if SExact.dtype.kind != 'f' and SExact.dtype.kind != 'i' and SExact.dtype.kind != 'u':
            raise ValueError('"SExact" must contain floats or integers.')
        NS_E = SExact.shape
        if len(NS_E) > 1:
            raise ValueError('"SExact" must be of shape (M, )')
        if NS_E[0] != M:
            raise ValueError('number of elements in "SExact" must be equal' +
                             'to number of columns in "S"')

    if not isinstance(X_Label, str):
        raise ValueError('"X_Label" must be a string.')
    if not isinstance(Y_Label, str):
        raise ValueError('"Y_Label" must be a string.')

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
    # Create plot
    ###########################################################################
    R = len(NN)
    A = len(col)
    L = int(np.ceil(M/A))
    clrs = repmat(col, L, 1)

    # Set horizontal and vertical limits:
    if NN[0] - np.mean(np.diff(NN)) > 0:
        H1 = NN[0] - np.mean(np.diff(NN))
    else:
        H1 = 0
    H2 = NN[-1] + end_length*(NN[-1] - NN[0])

    # Set minimum and maximum for y-axis
    if  len(S_lb) != 0:
        V1 = min(-0.1, np.min(S_lb.flatten()))
    else:
        V1 = min(-0.1, np.min(S.flatten()))
    if  len(S_ub) != 0:
        V2 = max(1.1, np.max(S_ub.flatten()))
    else:
        V2 = max(1.1, np.max(S.flatten()))

    labelinput_new = [np.nan]*M

    if sorting:
        Sidx = np.flip(np.argsort(S[-1, :]), axis=0)
        S = S[:, Sidx]
        for i in range(M):
            labelinput_new[i] = labelinput[Sidx[i]]
        if len(S_ub) != 0:
            S_ub = S_ub[:, Sidx]
        if len(S_lb) != 0:
            S_lb = S_lb[:, Sidx]
        if len(SExact) != 0:
            SExact = SExact[Sidx]

    if nb_legend < M:
        labelinput_new = labelinput_new[0:nb_legend]
        labelinput_new[-1] = labelinput_new[-1] + '...'

    # plt.figure()

    # For each index, plot final estimated value:
    for i in range(M):
        plt.plot(NN[-1], S[-1, i], 'o', markerfacecolor=clrs[i],
                 markeredgecolor='k', markersize=10)

    # Draw an horizontal line at 0:
    plt.plot([H1, H2], [0, 0], 'k')

    for i in range(M):
        # Plot trajectory with increasing number of samples:
        plt.plot(NN, S[:, i], color=clrs[i], linewidth=2.5)
        plt.box(on=True)

        if len(SExact) != 0:
            plt.plot([H1, H2], [SExact[i], SExact[i]], '--', color=clrs[i],
                     linewidth=2)

    # plot confidence bounds
    if len(S_lb) != 0:
        for i in range(M):
            plt.plot(NN, S_lb[:, i], '--', color=clrs[i], linewidth=1.2)

    if len(S_ub) != 0:
        for i in range(M):
            plt.plot(NN, S_ub[:, i], '--', color=clrs[i], linewidth=1.2)

    # Axes labels:
    plt.xlabel(X_Label, **pltfont)
    plt.ylabel(Y_Label, **pltfont)

    plt.legend(labelinput_new, loc='upper right', prop=pltfont_leg)

    # Tick labels for horizontal axis:
    xtick_label = [np.nan]*R
    for k in range(R):
        xtick_label[k] = '%d' % (NN[k])
    plt.xlim(H1, H2)
    plt.ylim(V1, V2)
    plt.xticks(NN, label=xtick_label, **pltfont)
    plt.grid(linestyle='--')


def scatter_plots(X, Y, n_col=5, Y_Label='Y', X_Labels=[], idx=np.array([])):

    """ This function produces M scatter plots, each one plotting the output
    sample (Y) against one component of the input vector sample
    (i.e. one column of X).

    Usage:
    plot_functions.scatter_plots(X, Y, n_col=5, Y_Label='Y', X_Labels=[],
                                 idx=np.array([]))

    Input:
               X = set of inputs samples                  - numpy.ndarray (N,M)
               Y = set of output samples                  - numpy.ndarray (N, )
                                                       or - numpy.ndarray (N,1)
   Optional input:
           n_col = number of panels per row in the plot   - integer
                (default: min(5, M))
         Y_Label = y-axis label   (default: 'Y')          - string
        X_labels = labels for the horizontal axis         - list (M elements)
                  (default: [' X1','X2',...])
            idx = indices of datapoints to be            - numpy.ndarray (N, )
                   highlighted. idx can the vector of
                   indices given by the function
                   RSA_indices_thres.

    Example:

    import numpy as np
    import SAFEpython.plot_functions as pf

    X = np.random.random((100, 3))
    sin_vect = np.vectorize(np.sin)
    Y = sin_vect(X[:, 0]) + 2*(sin_vect(X[:, 1]))**2 + (X[:, 2])**4*sin_vect(X[:, 0])
    pf.scatter_plots(X, Y)
    pf.scatter_plots(X, Y, n_col=2)
    pf.scatter_plots(X, Y, Y_Label='y')
    pf.scatter_plots(X, Y, Y_Label='y', X_Labels=['x1', 'x2', 'x3'])
    pf.scatter_plots(X, Y, Y_Label='y', X_Labels=['x1', 'x2', 'x3'], idx=Y > 2)

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
    mt = 'o' # marker type
    mth = 'o' # marker type for highlighted datapoints

    # Options for the colours:
    # You can produce a coloured plot or a black and white one
    # (printer-friendly).
    # Option 1 - coloured plot: uncomment the following 4 lines:
    me = 'b' # marker edge colour
    meh = 'r' # marker edge colour for highlighted datapoints
    mc = 'b' # marker face colour
    mch = 'r' # marker face colour for highlighted datapoints
    # Option 2 - B&W plot: uncomment the following 4 lines:
    #me  = [100/256, 100/256, 100/256] # marker edge colour
    #meh = 'k' # marker edge colour for highlighted datapoints
    #mc  = 'w' # marker face colour
    #mch = 'k' # marker face colour for highlighted datapoints

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
    if not isinstance(n_col, (int, np.int8, np.int16, np.int32, np.int64)):
        raise ValueError('"n_col" must be scalar and integer.')
    if n_col < 0:
        raise ValueError('"n_col" must be positive.')

    if not isinstance(Y_Label, str):
        raise ValueError('"Y_Label" must be a string.')

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

    if len(idx) != 0:
        if not isinstance(idx, np.ndarray):
            raise ValueError('"idx" must be a numpy.array.')
        if idx.dtype != 'bool':
            raise ValueError('"idx" must contain booleans.')
        idx = idx.flatten()
        Ni = idx.shape
        if Ni[0] != N:
            raise ValueError('""X"" and  "idx" must be have the same number of rows')

    ###########################################################################
    # Create plot
    ###########################################################################
    n_col = min(n_col, M) # n_col <= M
    n_row = int(np.ceil(M/n_col))

    #plt.figure()
    for i in range(M):
        plt.subplot(n_row, n_col, i+1)
        plt.plot(X[:, i], Y, mt, markerfacecolor=mc, markeredgecolor=me)

        # add axis, labels, etc.
        if ((i+1) % n_col) == 1: # first column
            plt.ylabel(Y_Label, **pltfont)

        plt.xlabel(X_Labels[i], **pltfont)
        plt.xlim((np.min(X[:, i]), np.max(X[:, i])))
        plt.ylim((np.min(Y) - np.std(Y), np.max(Y) + np.std(Y)))
        plt.xticks(**pltfont); plt.yticks(**pltfont)

        if len(idx) != 0:
            plt.plot(X[idx, i], Y[idx], mth, markerfacecolor=mch,
                     markeredgecolor=meh)

def scatter_plots_col(X, Y, i1, i2, ms=7, X_Labels=[], Y_Label='Y', ax=[]):

    """This function produces scatter plots of X[i1] against X[i2].
    The marker colour is proportional to the value of Y.

    Usage:
    plot_functions.scatter_plots_col(X, Y, i1, i2, ms=7, X_Labels=[],
                                     Y_Label='Y', ax=[])

    Input:
               X = set of inputs samples                  - numpy.ndarray (N,M)
               Y = set of output samples                  - numpy.ndarray (N, )
                                                       or - numpy.ndarray (N,1)
             i1 = index of input on the horizontal axis   - integer
             i2 = index of input on the vertical axis     - integer

    Optional input:
             ms = size of marker (default: 7)             - integer
       X_labels = labels for the horizontal axis          - list (M elements)
                  (default: [' X1','X2',...])
        Y_Label = label for colorbar   (default: 'Y')     - string

    Example:

    import numpy as np
    import matplotlib.pyplot as plt
    import SAFEpython.plot_functions as pf

    X = np.random.random((100, 3))
    sin_vect = np.vectorize(np.sin)
    Y = sin_vect(X[:, 0]) + 2 * (sin_vect(X[:, 1]))**2 + (X[:, 2])**4*sin_vect(X[: ,0])
    plt.figure()
    pf.scatter_plots_col(X, Y, 0, 1)
    plt.figure()
    pf.scatter_plots_col(X, Y, 1, 2, ms=16)
    plt.figure()
    pf.scatter_plots_col(X, Y, 1, 2, X_Labels=['x1', 'x2', 'x3'], Y_Label='output')

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
    # Colorscale
    colorscale = 'jet'
    #colorscale = 'gray' # black and white plot

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

    if not isinstance(i1, (int, np.int8, np.int16, np.int32, np.int64)):
        raise ValueError('"i1" must be scalar and integer.')
    if i1 < 0 or i1 > M-1:
        raise ValueError('"i1" must be in [0, M-1].')

    if not isinstance(i2, (int, np.int8, np.int16, np.int32, np.int64)):
        raise ValueError('"i2" must be scalar and integer.')
    if i2 < 0 or i2 > M-1:
        raise ValueError('"i2" must be in [0, M-1].')

    ###########################################################################
    # Check optional inputs
    ###########################################################################

    if not isinstance(Y_Label, str):
        raise ValueError('"Y_Label" must be a string.')

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

    if not isinstance(ms, (int, np.int8, np.int16, np.int32, np.int64)):
        raise ValueError('"ms" must be scalar and integer.')
    if ms <= 0:
        raise ValueError('"i2" must be positive.')

    ###########################################################################
    # Create plot
    ###########################################################################

    # Option 1: plot on curent figure
    if plt.get_fignums(): # if there is a figure recover axes of current figure
        ax = plt.gca()
    else: # else create a new figure
        plt.figure()
        ax = plt.gca()
    # Option 2: create a new figure
    #plt.figure()
    #ax = plt.gca()

    map_plot = plt.scatter(X[:, i1], X[:, i2], s=ms, c=Y, cmap=colorscale)
    plt.xlabel(X_Labels[i1], **pltfont)
    plt.ylabel(X_Labels[i2], **pltfont)
    plt.xlim((np.min(X[:, i1]), np.max(X[:, i1])))
    plt.ylim((np.min(X[:, i2]), np.max(X[:, i2])))
    plt.xticks(**pltfont)
    plt.yticks(**pltfont)

    # Add colorbar
    cb = plt.colorbar(map_plot, ax=ax)
    cb.set_label(Y_Label, **pltfont)
    cb.Fontname = pltfont['fontname']
    cb.ax.tick_params(labelsize=pltfont['fontsize'])


def scatter_plots_interaction(X, Y, ms=7, X_Labels=[], Y_Label='Y'):

    """ This function produces scatter plots of X[i] against X[j],
    for all possible combinations of (i,j). In each plot,
    the marker colour is proportional to the value of Y.

    Usage:
    plot_functions.scatter_plots_interaction(X, Y, ms=7, X_Labels=[],
                                             Y_Label='Y')

    Input:
               X = set of inputs samples                  - numpy.ndarray (N,M)
               Y = set of output samples                  - numpy.ndarray (N, )
                                                       or - numpy.ndarray (N,1)
    Optional input:
             ms = size of marker (default: 7)             - integer
       X_labels = labels for the horizontal axis          - list (M elements)
                  (default: [' X1','X2',...])
        Y_Label = label for colorbar   (default: 'Y')     - string

    Example:

    import numpy as np
    import SAFEpython.plot_functions as pf

    X = np.random.random((100, 3))
    sin_vect = np.vectorize(np.sin)
    Y = sin_vect(X[:, 0]) + 2 * (sin_vect(X[:, 1]))**2 + (X[:, 2])**4*sin_vect(X[:, 0])
    pf.scatter_plots_interaction(X, Y)
    pf.scatter_plots_interaction(X, Y, ms=16)
    pf.scatter_plots_interaction(X, Y, X_Labels=['x1', 'x2', 'x3'], Y_Label='output')

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
    # Colorscale
    colorscale = 'jet'
    #colorscale = 'gray' # black and white plot

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

    if not isinstance(Y_Label, str):
        raise ValueError('"Y_Label" must be a string.')

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

    if not isinstance(ms, (int, np.int8, np.int16, np.int32, np.int64)):
        raise ValueError('"ms" must be scalar and integer.')
    if ms <= 0:
        raise ValueError('"i2" must be positive.')

    ###########################################################################
    # Create plot
    ###########################################################################

    fig = plt.figure()

    k = 1
    for i in range(M-1):
        for j in range(i+1, M, 1):
            plt.subplot(M-1, M-1, k)
            map_plot = plt.scatter(X[:, i], X[:, j], s=ms, c=Y, cmap=colorscale)
            plt.title(X_Labels[i] + ' vs ' + X_Labels[j], **pltfont)
            plt.xlim((np.min(X[:, i]), np.max(X[:, i])))
            plt.ylim((np.min(X[:, j]), np.max(X[:, j])))
            plt.xticks([])
            plt.yticks([])
            k = k + 1
        k = k + i

    # Create colorbar
    cax = fig.add_axes([0.92, 0.05, 0.02, 0.8]) # Add axes for the colorbar
    cb = plt.colorbar(map_plot, ax=cax, fraction=1, extendfrac=1, extendrect=True)
    cb.set_label(Y_Label, **pltfont)
    cb.Fontname = pltfont['fontname']
    cb.ax.tick_params(labelsize=pltfont['fontsize'])
    # Make axes of the colorbar invisible
    cax.set_visible(False)


def parcoor(X, X_Labels=[], i_axis=-1, idx=np.array([])):

    """Create Parallel Coordinate Plot.

    Usage:
          plot_functions.parcoor(X, X_Labels=[], i_axis=-1, idx=np.array([]))

    Input:
         X = set of samples                               - numpy.ndarray (N,M)

    Optional input:
    XLabel = labels for the horizontal axis               - list (R elements)
             (default: {' X1','X2',...})
    i_axis = index of input to be used for assigning units- integer           -
             of measurement to the vertical axis if all
             the inputs have the same range.
             (if empty or not specified the ranges of all
             the inputs are displayed)
       idx = indices of samples statisfying some condition- numpy.ndarray (N, )
             which will be highlighted in different
             colours. idx is a vector of integers that
             contains the number of the different groups.
             idx can also be a boolean vector (when two
             groups). idx can be the vector of indices
             given by the function RSA_indices_thres or
             RSA_indices_groups.

    Example:

    import numpy as np
    import matplotlib.pyplot as plt
    import SAFEpython.plot_functions as pf

    X = np.random.random((100, 3))
    X[:, 2] = X[:, 2] + 2
    plt.figure()
    pf.parcoor(X, X_Labels=['a', 'b', 'c'])
    pf.parcoor(X, X_Labels=['a', 'b', 'c'], i_axis=2)
    idx = X[:, 0] < 0.3
    pf.parcoor(X, X_Labels=['a', 'b', 'c'], i_axis=2, idx=idx)

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
    colorscale = plt.cm.jet# colorscale for highlighted records
    #colorscale = 'gray'
    # Text formating of yticklabels
    yticklabels_form = '%3.1f' # float with 1 digit after decimal point
    # yticklabels_form = '%d' # integer

    ###########################################################################
    # Check inputs
    ###########################################################################
    if not isinstance(X, np.ndarray):
        raise ValueError('"X" must be a numpy.array.')
    if X.dtype.kind != 'f' and X.dtype.kind != 'i' and X.dtype.kind != 'u':
        raise ValueError('"X" must contain floats or integers.')
    Nx = X.shape
    N = Nx[0]
    M = Nx[1]

    ###########################################################################
    # Check optional inputs
    ###########################################################################
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

    if not isinstance(i_axis, (int, np.int8, np.int16, np.int32, np.int64)):
        raise ValueError('"i_axis" must be scalar and integer.')
    if i_axis < -1 or i_axis > M-1:
        raise ValueError('"i_axis" must be in [0, M-1].')

    if len(idx) != 0:
        if not isinstance(idx, np.ndarray):
            raise ValueError('"idx" must be a numpy.array.')
        if idx.dtype != 'bool' and idx.dtype.kind != 'i':
            raise ValueError('"idx" must contain booleans or integers.')
        idx = idx.flatten()
        Ni = idx.shape
        if Ni[0] != N:
            raise ValueError('""X"" and  "idx" must be have the same number of rows')

    ###########################################################################
    # Create plot
    ###########################################################################

    # Rescale data
    xmin = np.nan * np.ones((1, M))
    xmax = np.nan * np.ones((1, M))
    xmin[0, :] = np.min(X, axis=0)
    xmax[0, :] = np.max(X, axis=0)
    X2 = (X-repmat(xmin, N, 1))/(repmat(xmax-xmin, N, 1)) # (N,M)
    X2 = np.transpose(X2) # shape (M,N)

    # Number of groups
    if len(idx) == 0:
        m = 1
    else:
        groups = np.unique(idx)
        m = len(groups)

    # Set plot color and linewidth
    if m == 1:
        lc = [0, 0, 0]  # linecolor
        lw = 1  # linewidth
    else:
        lc = [150/256, 150/256, 150/256]
        lch = colorscale(np.linspace(0, 1, m-1)) # color for highlighted records
        lw = 1
        lwh = 2

    # Plot
    #plt.figure()
    plt.plot(X2, color=lc, linewidth=lw)
    for j in range(m-1):
        plt.plot(X2[:, idx == groups[j+1]], color=lch[j], linewidth=lwh)

    # Costumize horizontal axis
    plt.xticks(np.arange(0, M, 1), X_Labels, **pltfont)
    plt.xlim([-0.3, M-0.7])
    plt.ylim([-0.1, 1.1])
    # Label ticks on vertical axis

    if i_axis > 0: # Show only the input range specified by user
        Yticklabel = [np.nan]*10
        _, Ytick = np.histogram(X[:, i_axis], 10)
        for j in range(10):
            Yticklabel[j] = yticklabels_form % ((Ytick[j]+Ytick[j+1])/2)
        _, Ytick = np.histogram(X2[i_axis, :], 10)
        plt.yticks((Ytick[np.arange(0, 10, 1)]+Ytick[np.arange(1, 11, 1)])/2,
                   Yticklabel, **pltfont)
        plt.ylabel(X_Labels[i_axis], **pltfont)

    else: # Show ranges of all inputs
        for i in range(M):
            xmin_i = yticklabels_form % (xmin[0, i])
            xmax_i = yticklabels_form % (xmax[0, i])
            plt.text(i+0.05, 0, xmin_i, horizontalalignment='center',
                     verticalalignment='top', **pltfont)
            plt.text(i+0.05, 1, xmax_i, horizontalalignment='center',
                     verticalalignment='bottom', **pltfont)
            plt.yticks([])


def plot_cdf(Y, Y_Label='Y'):

    """This function plot the empirical Cumulative Distribution Function (CDF)
    of a given sample (y).

    Usage:
        plot_functions.plot_cdf(Y, Y_Label='Y')

    Input:
          Y = sample variable values                      - numpy.ndarray (N, )
                                                       or - numpy.ndarray (N,1)
    Optional input:
    Y_label = label for horizontal axis                   - string

    NOTE: If Y includes any NaN values, the function will identify them from
    the CDF.

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
    lc = 'k' # line colour

    ###########################################################################
    # Check inputs
    ###########################################################################
    if not isinstance(Y, np.ndarray):
        raise ValueError('"Y" must be a numpy.array.')
    if Y.dtype.kind != 'f' and Y.dtype.kind != 'i' and Y.dtype.kind != 'u':
        raise ValueError('"Y" must contain floats or integers.')
    Ny = Y.shape
    if len(Ny) > 1:
        if Ny[1] != 1:
            raise ValueError('"Y" be of shape (N,1) or (N, ).')
    N = Ny[0]
    Y = Y.flatten() # shape (N, )

    if np.isnan(Y).any():
        warn('some data in "Y" are nan')
    if np.isinf(Y).any():
        warn('some data in "Y" are inf')

    if not isinstance(Y_Label, str):
        raise ValueError('"Y_Label" must be a string.')

    ###########################################################################
    # Create plot
    ###########################################################################

    Y = Y[~np.isnan(Y)] # remove NaNs
    ymin = np.min(Y)
    ymax = np.max(Y)

    #plt.figure()

    # Option 1: use the function empiricalcdf of SAFE
    Nmin = 5000
    if N > Nmin:
        Yi = np.sort(Y)
        F = empiricalcdf(Y, Yi)
        plt.plot(Yi, F, '.', color=lc)
    else:
        Yi = np.linspace(ymin, ymax, Nmin)
        F = empiricalcdf(Y, Yi)
        plt.plot(Yi, F, color=lc)

    # Option 2: use the ECDF function of the python package 'statsmodels'
    #ecdf = ECDF(Y)
    #plt.plot(ecdf.x,ecdf.y, color=lc)

    # Customise plot
    plt.xticks(**pltfont); plt.yticks(**pltfont)
    plt.xlabel(Y_Label, **pltfont)
    plt.ylabel('CDF', **pltfont)
    plt.box(on=True)

    # Limit for horizontal axisym = min(y);
    if ymin == ymax: # (i.e., if all data have the same value)
        ymin = ymin - ymin/10
        ymax = ymax + ymax/10
    plt.xlim((ymin, ymax))
    plt.ylim((0, 1))


def plot_pdf(Y, Y_Label='Y', nbins=0):

    """This function plot the empirical Probability Distribution Function (PDF)
    of a given sample (y). The empirical PDF is approximated by the histogram

    Usage:
        fi, yi = plot_functions.plot_pdf((Y, Y_Label='Y', nbins=0)

    Input:
          Y = sample variable values                  - numpy.ndarray (N, )
                                                   or - numpy.ndarray (N,1)
    Y_label = label for horizontal axis               - string
       nbins = number of bins for histogram           - integer
              (default: min(100, N/10))
    Output:
         fi = frequency of each bin                   - numpy.ndarray (nbins, )
         yi = center of each bin                      - numpy.ndarray (nbins, )

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
    lc = 'k' # line colour

    ###########################################################################
    # Check inputs
    ###########################################################################
    if not isinstance(Y, np.ndarray):
        raise ValueError('"Y" must be a numpy.array.')
    if Y.dtype.kind != 'f' and Y.dtype.kind != 'i' and Y.dtype.kind != 'u':
        raise ValueError('"Y" must contain floats or integers.')
    Ny = Y.shape
    if len(Ny) > 1:
        if Ny[1] != 1:
            raise ValueError('"Y" be of shape (N,1) or (N, ).')
    N = Ny[0]
    Y = Y.flatten() # shape (N, )

    if np.isnan(Y).any():
        warn('some data in "Y" are nan')
    if np.isinf(Y).any():
        warn('some data in "Y" are inf')

    ###########################################################################
    # Check optional inputs
    ###########################################################################
    if not isinstance(Y_Label, str):
        raise ValueError('"Y_Label" must be a string.')

    if not isinstance(nbins, (int, np.int8, np.int16, np.int32, np.int64)):
        raise ValueError('"nbins" must be an integer.')
    if nbins < 0:
        raise ValueError('"nbins" must be positive.')
    elif nbins == 0:
        nbins = min(100, N/10) #  set default value

    ###########################################################################
    # Create plot
    ###########################################################################

    fi, yi = np.histogram(Y.flatten(), nbins, density=True)
    yi = (yi[np.arange(0, nbins, 1)]+yi[np.arange(1, nbins+1, 1)])/2 # bin centers

    # Limit for vertical axis
    ymin = np.min(Y)
    ymax = np.max(Y)
    if ymin == ymax: # (i.e., if all data have the same value)
        ymin = ymin - ymin/10
        ymax = ymax + ymax/10

    # Limt for horizontal axis
    fmin = 0
    fmax = np.max(fi)
#    if np.mean(fi)/np.std(fi) > 2: # if frequency is rather constant (''flat''),
#        # expand the vertical axis so to highlight this:
#        fmax = min(1,np.max(fi)+10*np.mean(fi))
#    else:
#        fmax = min(1,np.max(fi)+np.std(fi))

    # Plot
    #plt.figure()
    plt.plot(yi, fi, '.-', linewidth=2, color=lc)
    plt.xlim((ymin, ymax))
    plt.ylim((fmin, fmax))
    plt.xticks(**pltfont); plt.yticks(**pltfont)
    plt.xlabel(Y_Label, **pltfont)
    plt.ylabel('PDF', **pltfont)
    plt.box(on=True)

    return fi, yi

def stackedbar(S, labelinput=[], Y_Label='Sensitivity', horiz_tick=[], horiz_tick_label=[]):

    """Plot and compare N different set of indices using stacked bar plot.

    Usage:
         plot_functions.stackedbar(S, labelinput=[], Y_Label='Sensitivity',
                                   horiz_tick=[], horiz_tick_label=[])

    Input:
                   S = set of indices (N sets, each       - numpy.ndarray (N,M)
                       composed of M indices)
    Optional input:
          labelinput = names of inputs to appear in the   - list (M elements)
                       legend (default: no legend)
             Y_Label = label for vertical axis            - string
                       (default: 'Sensitivity')
          horiz_tick = ticks for horizontal axis          - list (N number)
                      (default: [1,2,...,N] )
    horiz_tick_label = labels for ticks of horizontal     - list (N strings)
                      axis (default: ['1','2',...,'H'] )

    Example:

    import numpy as np
    import matplotlib.pyplot as plt
    from SAFEpython.plot_functions import stackedbar

    S = np.random.random((5, 4))
    plt.figure(); stackedbar(S)
    plt.figure(); stackedbar(S, labelinput=['a', 'b', 'c', 'd'])
    plt.figure(); stackedbar(S, labelinput=['a', 'b', 'c', 'd'], Y_Label='total')
    plt.figure(); stackedbar(S, Y_Label='total', horiz_tick=[1, 2, 9, 11, 12])
    plt.figure(); stackedbar(S, Y_Label='total', horiz_tick=[1, 2, 9, 11, 12],
                             horiz_tick_label=['d1', 'd2', 'd3', 'd4', 'd5'])

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
    pltfont_leg = {'family': 'DejaVu Sans', 'size': 15} # font for legend
    ###########################################################################
    # Check inputs
    ###########################################################################
    if not isinstance(S, np.ndarray):
        raise ValueError('"S" must be a numpy.array.')
    if S.dtype.kind != 'f' and S.dtype.kind != 'i' and S.dtype.kind != 'u':
        raise ValueError('"S" must contain floats or integers.')

    Ns = S.shape
    if len(Ns) == 1:
        M = Ns[0]
        N = 1
    else:
        N = Ns[0]
        M = Ns[1]

    ###########################################################################
    # Check optional inputs
    ###########################################################################
    if not isinstance(Y_Label, str):
        raise ValueError('"str_legend" must be a string.')

    if labelinput:
        if not isinstance(labelinput, list):
            raise ValueError('"labelinput" must be a list with M elements.')
        if not all(isinstance(i, str) for i in labelinput):
            raise ValueError('Elements in "labelinput" must be strings.')
        if len(labelinput) != M:
            raise ValueError('"labelinput" must have M elements.')

    if not horiz_tick:
        horiz_tick = [np.nan]*N
        for i in range(N):
            horiz_tick[i] = i
    else:
        if not all(isinstance(i, float) or isinstance(i, int) for i in horiz_tick):
            raise ValueError('Elements in "horiz_tick" must be int or float.')
        if len(horiz_tick) != N:
            raise ValueError('"horiz_tick" must have M elements.')

    if horiz_tick_label:
        if not isinstance(horiz_tick_label, list):
            raise ValueError('"horiz_tick_label" must be a list with M elements.')
        if not all(isinstance(i, str) for i in horiz_tick_label):
            raise ValueError('Elements in "horiz_tick_label" must be strings.')
        if len(horiz_tick_label) != N:
            raise ValueError('"horiz_tick_label" must have N elements.')

    ###########################################################################
    # Create plot
    ###########################################################################

    #plt.figure()
    plt.bar(horiz_tick, S[:, 0])
    for i in range(1, M, 1):
        plt.bar(horiz_tick, S[:, i], bottom=np.sum(S[:, 0:i], axis=1))
    plt.xticks(horiz_tick, **pltfont)
    plt.grid(axis='x')
    if horiz_tick_label:
        plt.xticks(horiz_tick, horiz_tick_label, **pltfont)
    else:
        plt.xticks(horiz_tick, **pltfont)

    if labelinput:
        plt.legend(labelinput, prop=pltfont_leg)

    plt.ylabel(Y_Label, **pltfont)
