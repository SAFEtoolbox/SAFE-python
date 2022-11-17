"""
    Module to compute the Sobol' g-function

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

     Saltelli et al. (2008) Global Sensitivity Analysis, The Primer, Wiley.
"""

from __future__ import division, absolute_import, print_function

import numpy as np
from numba import jit

@jit
def sobol_g_function(x, a):

    """Implements the Sobol' g-function, a standard benchmark function in
    the Sensitivity Analysis literature (see for instance Sec. 3.6 in
    Saltelli et al. (2008)).

    Usage:
    y, V, Si_ex = sobol_g_function(x, a)

    Input:
         x = function inputs [x(i)~Unif(0,1) for all i]    - numpy.ndarray(M, )
         a = function parameters (fixed)                   - numpy.ndarray(M, )

    Output:
         y = output                                        - numpy.ndarray(1, )
         V = output variance (*)                           - numpy.ndarray(1, )
     Si_ex = first-order sensitivity indices (*)           - numpy.ndarray(3, )

    (*) = exact value computed analytically

    REFERENCES

    Saltelli et al. (2008) Global Sensitivity Analysis, The Primer, Wiley.

    This function is part of the SAFE Toolbox by F. Pianosi, F. Sarrazin
    and T. Wagener at Bristol University (2015).
    SAFE is provided without any warranty and for non-commercial use only.
    For more details, see the Licence file included in the root directory
    of this distribution.
    For any comment and feedback, or to discuss a Licence agreement for
    commercial use, please contact: francesca.pianosi@bristol.ac.uk
    For details on how to cite SAFE in your publication, please see:
    https://www.safetoolbox.info"""

    if len(a) != len(x):
        raise ValueError('x and a must have the same number of elements')

    g = (abs(4*x-2)+a) / (1+a)
    y = np.array(np.prod(g))

    # By model definition:
    # Vi = VAR(E(Y|Xi)) = 1 / ( 3(1+ai)**2 )
    # VARy = VAR(Y) = - 1 + np.prod( 1 + Vi )
    Vi = np.array(1/(3*(1+a)**2))
    V = np.array(-1 + np.prod(1+Vi))
    Si_ex = Vi/V

    return y, V, Si_ex
