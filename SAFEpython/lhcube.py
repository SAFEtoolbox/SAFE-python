"""
    Module to perform latin hypercube sampling

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

    this is a python version of the code in:
            J.C. Rougier, calibrate package
            http://www.maths.bris.ac.uk/~mazjcr/#software
"""
from __future__ import division, absolute_import, print_function

import numpy as np
from scipy.spatial.distance import pdist

def lhcube(N, M, nrep=5):

    """Generate a latin hypercube of N datapoints in the M-dimensional hypercube
    [0,1]x[0,1]x...x[0,1]. If required, generation can be repeated for a
    prescribed number of times and the maximin latin hypercube is returned.

    Usage:
        X, d = lhcube.lhcube(N, M, nrep=5)

    Input:
       N = number of samples                              - positive int
       M = number of inputs                               - positive int
    nrep = number of repetition (default: 5)              - positive int

    Output:
       X = sample points                                  - numpy.ndarray (N,M)
       d = minimum distance between two points (rows) in X- scalar

    Example:

    import numpy as np
    import matplotlib.pyplot as plt
    from SAFEpython.lhcube import lhcube
    N = 10
    M = 2
    X, _ = lhcube(N, M)
    plt.figure()
    plt.plot(X[:, 0], X[:, 1], '.')
    plt.xticks(np.arange(0, 1+1/N, 1/N), '')
    plt.yticks(np.arange(0, 1+1/N, 1/N), '')
    plt.grid(linestyle='--')

    References:

    this is a python version of the code in:
            J.C. Rougier, calibrate package
            http://www.maths.bris.ac.uk/~mazjcr/#software

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
    if not isinstance(N, (int, np.int8, np.int16, np.int32, np.int64)):
        raise ValueError('"N" must be scalar and integer.')
    if N <= 0:
        raise ValueError('"N" must be positive.')

    if not isinstance(M, (int, np.int8, np.int16, np.int32, np.int64)):
        raise ValueError('"M" must be scalar and integer.')
    if M <= 0:
        raise ValueError('"M" must be positive.')

    if not isinstance(nrep, (int, np.int8, np.int16, np.int32, np.int64)):
        raise ValueError('"nrep" must be scalar and integer.')
    if nrep <= 0:
        raise ValueError('"nrep" must be positive.')

    ###########################################################################
    # Generate latin hypercube sample
    ###########################################################################
    d = 0

    # Generate nrep hypercube sample X and keep the one that maximises the
    # minimum inter-point Euclidean distance between any two sampled points:
    for k in range(nrep):

        # Generate a latin-hypercube:
        ran = np.random.random((N, M))
        Xk = np.zeros((N, M))
        for i in range(M):
            idx = np.random.choice(np.arange(1, N+1, 1), size=(N, ), replace=False)
            Xk[:, i] = (idx - ran[:, i])/N

        # Compute the minimum distance between points in X:
        dk = np.min(pdist(Xk, metric='euclidean'))

        # If the current latin hypercube has minimum distance higher than
        # the best so far, it will be retained as the best.
        if dk > d:
            X = Xk
            d = dk

    return X, d


def lhcube_shrink(X, N_new, nrep=10):

    """This function drop rows from a latin hypercube using the maximin
    criterion.

    Usage:
        X_new, idx_new = lhcube.lhcube_shrink(X, N_new, nrep=10)


    Input:
          X = initial latin hypercube                 - numpy.ndarray (N,M)
      N_new = new dimension for the latin hypercube   - integer
       nrep = number of replicate to select the best  - integer
              hypercube (default value: 10)

    Output:
      X_new = new latin hypercube                     - numpy.ndarray (N_new,M)
    idx_new = indices of the rows selected from X     - numpy.ndarray (N_new, )
               ( i.e. Xnew = X[idx_new,:] )

    Example:

    import numpy as np
    import matplotlib.pyplot as plt
    from SAFEpython.lhcube import lhcube, lhcube_shrink
    N = 30
    M =  2
    X, _ = lhcube(N, M) # create LHS
    plt.figure()
    plt.plot(X[:, 0], X[:, 1], 'x');
    plt.xticks(np.arange(0, 1+1/N, 1/N), '')
    plt.yticks(np.arange(0, 1+1/N, 1/N), '')
    plt.grid(linestyle='--')
    N_new = 20
    X_new, _ = lhcube_shrink(X, N_new)
    plt.plot(X_new[:, 0], X_new[:, 1], 'or', fillstyle='none')

    References:

    this is a python version of the code in:
            J.C. Rougier, calibrate package
            http://www.maths.bris.ac.uk/~mazjcr/#software

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

    if not np.isscalar(N_new):
        raise ValueError('"N_new" must be scalar')
    if (N_new - np.floor(N_new)) != 0:
        raise ValueError('"N_new" must be integer')
    if N_new <= 0:
        raise ValueError('"N_new" must be positive.')

    if not isinstance(nrep, (int, np.int8, np.int16, np.int32, np.int64)):
        raise ValueError('"nrep" must be scalar and integer.')
    if nrep <= 0:
        raise ValueError('"nrep" must be positive.')

    Nx = X.shape
    N = Nx[0]

    ###########################################################################
    # Drop rows
    ###########################################################################

    X_new = np.nan
    ddbest = 0
    idx_new = np.nan

    # Generate nrep subsamples of the initial latin hypercube sample X and keep
    # the one that maximises the minimum inter-point Euclidean distance between
    # any two sampled points:
    for i in range(nrep):
        idx = np.random.choice(N, size=(N_new, ), replace=False)
        Xi = X[idx,]
        dd = np.min(pdist(Xi, metric='euclidean'))
        if dd > ddbest:
            X_new = Xi
            idx_new = idx
            ddbest = dd

    return X_new, idx_new


def lhcube_extend(X, N_new, nrep=10):

    """ This function add rows to a latin hypercube using the maximin criterion.

    Usage:
        X_new = lhcube.lhcube_extend(X, N_new, nrep=10)

    Input:
          X = initial latin hypercube                 - numpy.ndarray (N,M)
      N_new = new dimension for the latin hypercube   - scalar
       nrep = number of replicate to select the best  - scalar
             hypercube (default value: 10)

    Output:
      X_new = best latin hypercube                    - numpy.ndarray (N_new,M)
    idx_new = indices of the rows selected from X     - vector (N_new,1)
               [ i.e. Xnew = X(idx_new,:) ]

    Example:

    import numpy as np
    import matplotlib.pyplot as plt
    from SAFEpython.lhcube import lhcube, lhcube_extend

    N = 30
    M =  2
    X, _ = lhcube(N, M) # create LHS
    plt.figure()
    plt.plot(X[:, 0], X[:, 1], 'x')
    plt.xticks(np.arange(0, 1+1/N, 1/N), '')
    plt.yticks(np.arange(0, 1+1/N, 1/N), '')
    plt.grid(linestyle='--')
    N_new = 40
    X_new = lhcube_extend(X, N_new)
    plt.plot(X_new[:, 0], X_new[:, 1], 'or', fillstyle='none')

    References:

    this is a python version of the code in:
            J.C. Rougier, calibrate package
            http://www.maths.bris.ac.uk/~mazjcr/#software

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

    if not isinstance(N_new, (int, np.int8, np.int16, np.int32, np.int64)):
        raise ValueError('"N_new" must be scalar and integer.')
    if N_new <= 0:
        raise ValueError('"N_new" must be positive.')

    if not isinstance(nrep, (int, np.int8, np.int16, np.int32, np.int64)):
        raise ValueError('"nrep" must be scalar and integer.')
    if nrep <= 0:
        raise ValueError('"nrep" must be positive.')

    Nx = X.shape
    N = int(Nx[0])
    M = int(Nx[1])
    # force numbers of to integers (so that it works for Python 2)

    ###########################################################################
    # Drop rows
    ###########################################################################

    X_new = np.nan
    ddbest = 0

    # Generate nrep extended samples of the initial latin hypercube sample X
    # and keep the one that maximises the minimum inter-point Euclidean
    # distance between any two sampled points:
    for i in range(nrep):
        Xext, _ = lhcube(N_new-N, M)
        Xi = np.concatenate((X, Xext), axis=0)
        dd = np.min(pdist(Xi, metric='euclidean'))
        if dd > ddbest:
            X_new = Xi
            ddbest = dd

    return X_new
