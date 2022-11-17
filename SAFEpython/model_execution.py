"""
    Module to execute the model

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

import numpy as np

def model_execution(fun_test, X, *args, ExtraArgOut=False):

    """Executes the model coded in the matlab function 'fun_test' against
    each sample input vector in matrix X, and returns the associated output
    vectors (in matrix Y).

    Usage:
    Y = model_execution.model_execution(fun_test, X, *args, ExtraArgOut=False)

    Input:
     fun_test = function implementing the model.           - python function
                The first output argument of
                'fun_test' must be a numpy.ndarray of
                shape (P, ), (1,P) or (P,1).
           X = matrix of N input sample                    - numpy.ndarray(N,M)
              (each input sample has M components)

    Optional input:
        *args = extra arguments needed to execute 'fun_test'

    ExtraArgOut = specifies if extra output arguments must - boolean
                  be returned beyond Y in case
                  'fun_test' provides extra output
                  arguments.
                  (default: False, i.e. no extra output
                  arguments are returned)

    Output:
            Y = vector (NxP) of associated output samples, - numpy.ndarray(N,P)
                P being the number of model outputs
                associated to each sampled input
                combination (corresponds to the first
                output argument of 'fun_test')

    Optional output:
       argout = extra output arguments provided by         - tuple
                'fun_test' if ExtraArgOut is True. argout
                is a tuple and each element of argout
                corresponds to an output argument of fun_test.

    NOTES:
    1) If the 'fun_test' function requires other arguments besides 'X',
    or produce other output besides 'Y', they can be passed as optional
    arguments after 'X' and recovered as optional output after 'Y'.
    2) The assignment of the 'argout' variable must be customised by the user
    for the specific case study (see L141-145)

    Example:

    import numpy as np
    from SAFEpython.model_execution import model_execution
    from SAFEpython.sobol_g import sobol_g_function

    fun_test = sobol_g_function
    X =  np.random.random((3, 4))
    a = np.ones((4,))
    Y = model_execution(fun_test, X, a) # Or:
    Y, tmp = model_execution(fun_test, X, a, ExtraArgOut=True)
    V = tmp[0]
    Si_ex = tmp[1]

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
    N = Nx[0]
    if len(Nx) != 2:
        raise ValueError('"X" must have at least two rows.')
    if not isinstance(ExtraArgOut, bool):
        raise ValueError('"ExtraArgOut" must be scalar and boolean.')

    ###########################################################################
    # Recover the type and shapes of the output arguments of my_fun
    ###########################################################################

    # Perform one model run to determine P (number output to be saved ) and the
    # number of extra arguments given by 'fun_test':
    tmp = fun_test(X[0, :], *args)  # modified

    if isinstance(tmp, tuple): # my_fun provides extra output arguments
        # Recover the total number of extra arguments:
        NumExtraArgOut = len(tmp) - 1 # number of extra arguments
        # Check format of first argument of the function
        if not isinstance(tmp[0], np.ndarray):
            raise ValueError('the first output argument returned by ' +
                             '"fun_test" must be a numpy.ndarray')
        if tmp[0].dtype.kind != 'f' and tmp[0].dtype.kind != 'i' and \
        tmp[0].dtype.kind != 'u':
            raise ValueError('the first output argument returned by ' +
                             '"fun_test" must contains float or integers.')
        N1 = tmp[0].shape
        if len(N1) > 2:
            raise ValueError('the first output argument returned by ' +
                             '"fun_test" must be of shape (P, ), (P,1) or (1,P).')
        elif len(N1) == 2:
            if N1[0] != 1 and N1[1] != 1:
                raise ValueError('the first output argument returned by ' +
                                 '"fun_test" must be of shape (P, ), (P,1) or (1,P).')
        P = len(tmp[0].flatten())

    else: # no extra arguments
        NumExtraArgOut = 0
        if not isinstance(tmp, np.ndarray):
            raise ValueError('the first output argument returned by ' +
                             '"fun_test" must be a numpy.ndarray')
        if tmp.dtype.kind != 'f' and tmp.dtype.kind != 'i' and tmp.dtype.kind != 'u':
            raise ValueError('the first output argument returned by '  +
                             '"fun_test" must contains float or integers.')
        N1 = tmp.shape
        if len(N1) > 2:
            raise ValueError('the first output argument returned by ' +
                             '"fun_test" must be of shape (P, ), (P,1) or (1,P).')
        elif len(N1) == 2:
            if N1[0] != 1 and N1[1] != 1:
                raise ValueError('the first output argument returned by ' +
                                 '"fun_test" must be of shape (P, ), (P,1) or (1,P).')
        P = len(tmp.flatten())

    # Perform the model runs

    Y = np.nan * np.ones((N, P)) # variable initialization
    for j in range(N):
        if NumExtraArgOut == 0:
            Y[j, :] = fun_test(X[j, :], *args).flatten()
        else: # save extra output arguments in a tuple
            tmp = fun_test(X[j, :], *args)
            Y[j, :] = tmp[0].flatten()
            # Assign extra output argument (these lines should be customised by
            # the user for the specific case study):
            argout = ()
            for i in range(NumExtraArgOut):
                argout = argout + (tmp[i+1],)

    # Determine the output arguments to be returned
    if ExtraArgOut and NumExtraArgOut != 0: # Return extra arguments
        return Y, argout
    else:
        return Y
