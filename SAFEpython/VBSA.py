"""
    Module to perform Variance Based Sensitivity Analysis (VBSA)

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

    Homma, T. and A., Saltelli (1996). Importance measures in global
    sensitivity analysis of nonlinear models.
    Reliability Engineering & System Safety, 52(1), 1-17.

    Saltelli et al. (2008), Global Sensitivity Analysis, The Primer, Wiley.

    Saltelli et al. (2010), Variance based sensitivity analysis of model
    output. Design and estimator for the total sensitivity index, Computer
    Physics Communications, 181, 259-270.
"""
from __future__ import division, absolute_import, print_function

from warnings import warn
import numpy as np


def vbsa_resampling(X):

    """ This function implements the resampling strategy needed to build the
    approximators of the first-order (main effects) and total order
    sensitivity indices (e.g. Saltelli et al. 2008; 2010).

    Usage:
        XA, XB, XC = VBSA.vbsa_resampling(X)

    Input:
     X = matrix of 2*N input samples                     - numpy.ndarray(2*N,M)

    Output:
    XA = first N rows of X                               - numpy.ndarray(N,M)
    XB = last N rows of X                                - numpy.ndarray(N,M)
    XC = Block array of M 'recombinations of XA and XB   - numpy.ndarray(N*M,M)
         XC = numpy.concatenate((XC1, XC2, ..., XCM),
                            axis=0)
         Each block XCi is a (N,M) array whose columns
         are all taken from XB exception made for i-th,
         which is taken from XA.

    This function is meant to be used in combination with 'VBSA.vbsa_indices'.
    See help of that function for more details and examples.

    REFERENCES:

    Saltelli et al. (2008), Global Sensitivity Analysis, The Primer, Wiley.

    Saltelli et al. (2010), Variance based sensitivity analysis of model
    output. Design and estimator for the total sensitivity index, Computer
    Physics Communications, 181, 259-270.

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
    NX = Nx[0]
    M = Nx[1]

    if (NX) % 2 > 0:
        NX = NX - 1
        warn("i=Input matrix X has an odd number of rows, using the first %d " % (NX) +
             "rows only.")
    N = int(NX/2)
    XA = X[0:N, :]
    XB = X[N:2*N, :]
    XC = np.nan*np.ones((N*M, M))

    Ci = np.zeros((N, M))

    for i in range(M):
        idxnot_i = np.concatenate((np.arange(0, i), np.arange(i+1, M)))
        Ci[:, idxnot_i] = XB[:, idxnot_i]
        Ci[:, i] = XA[:, i]
        XC[i*N:(i+1)*N, :] = Ci

    return XA, XB, XC


def vbsa_indices(YA, YB, YC, M, Nboot=0, dummy=False):

    """ This function estimates the variance-based first-order indices
    (or 'main effects') and total-order ('total effects') indices
    (Homma and Saltelli, 1996) by using the approximation technique described
    e.g. in Saltelli et al. (2008) and (2010).

    Usage:
                     Si, STi = \
                         VBSA.vbsa_indices(YA, YB, YC, M, Nboot=0, dummy=False)
    Si, STi, Sdummy, STdummy = \
                         VBSA.vbsa_indices(YA, YB, YC, M, Nboot=0, dummy=True)

    Input:
         YA = set of output samples                      - numpy.ndarray(N, )
                                                      or - numpy.ndarray(N,1)
         YB = set of output samples                      - numpy.ndarray(N, )
             (independent from YA)                    or - numpy.ndarray(N,1)
         YC = set of output samples from resampling (*)  - numpy.ndarray(N*M,)
                                                      of - numpy.ndarray(N*M,1)
         M = number of input variables                   - integer

    Optional input:
     Nboot = number of resamples used for boostrapping   - integer
           (default: 0)
      dummy = if dummy is True, an articial input is     - boolean
              added to the set of inputs and the
              sensitivity indices are calculated for the
              dummy input.
              The sensitivity indices for the dummy
              input are estimates of the approximation
              error of the sensitivity indices and they
              can be used for screening, i.e. to
              separate influential and non-influential
              inputs.

    Output:
        Si  = first-order sensitivity indices         if Nboot <= 1:
              estimated for each bootstrap resample   - numpy.ndarray (M, )
              when bootstrapping is used (Nboot>1)    if Nboot > 1:
                                                      - numpy.ndarray (Nboot,M)
       STi  = total effect sensitivity indices         if Nboot <= 1:
            estimated for each bootstrap resample     - numpy.ndarray (M, )
            when bootstrapping is used (Nboot>1)      if Nboot > 1:
                                                      - numpy.ndarray (Nboot,M)
    Optional output (if dummy is True):
    Sdummy  = first-order sensitivity indices         if Nboot <= 1:
              for the dummy input estimated for each  - numpy.ndarray (M, )
              bootstrap resample when bootstrapping   if Nboot > 1:
              is used                                 - numpy.ndarray (Nboot,M)

    STdummy = total effect sensitivity indices        if Nboot <= 1:
              for the dummy input estimated for each  - numpy.ndarray (M, )
              bootstrap resample when bootstrapping   if Nboot > 1:
              is used                                 - numpy.ndarray (Nboot,M)
    NOTES:

    (*) By default, here we use the estimators described by Saltelli et al.
        (2008) and Saltelli et al. (2010) (see comments in the code of the
        function VBSA.compute_indices for specific references to the equations
        implemented here!). These are obtained from 3 sets of output samples
        (YA, YB and YC), which are obtained by executing the model against
        three input matrices XA, XB and XC generated by the
        'VBSA.vbsa_resampling' function:
        - see example below about how to use 'vbsa_resampling' and
          'vb_firstorder_indices' sequentially;
        - see help of 'vbsa_resampling' to learn more about XA, XB and XC.

    (**) If the vectors YA, YB or YC include any NaN values, the function
          will identify them and exclude them from further computation.
          A Warning message about the number of discarded NaN elements (and
          hence the actual number of samples used for estimating Si and STi)
          will be displayed.

    REFERENCES:

    Homma, T. and A., Saltelli (1996). Importance measures in global
    sensitivity analysis of nonlinear models.
    Reliability Engineering & System Safety, 52(1), 1-17.

    Saltelli et al. (2008), Global Sensitivity Analysis, The Primer, Wiley.

    Saltelli et al. (2010), Variance based sensitivity analysis of model
    output. Design and estimator for the total sensitivity index, Computer
    Physics Communications, 181, 259-270.

    REFERENCE FOR THE DUMMY PARAMETER:

    Khorashadi Zadeh et al. (2017), Comparison of variance-based and moment-
    independent global sensitivity analysis approaches by application to the
    SWAT model, Environmental Modelling & Software,91, 210-222.

    EXAMPLE:

    import numpy as np
    import scipy.stats as st
    import SAFEpython.VBSA as VB
    from SAFEpython.model_execution import model_execution
    from SAFEpython.sampling import AAT_sampling

    from SAFEpython.ishigami_homma import ishigami_homma_function

    fun_test = ishigami_homma_function
    M = 3
    distr_fun = st.uniform
    xmin = -np.pi
    xmax = np.pi
    distr_par = [xmin, xmax - xmin]
    N = 10000
    samp_strat = 'lhs'
    X = AAT_sampling(samp_strat, M, distr_fun ,distr_par, 2*N)
    [ XA, XB, XC ] = VB.vbsa_resampling(X)
    YA = model_execution(fun_test, XA)
    YB = model_execution(fun_test, XB)
    YC = model_execution(fun_test, XC)
    Si, STi = VB.vbsa_indices(YA, YB, YC, M)

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
    if not isinstance(YA, np.ndarray):
        raise ValueError('"YA" must be a numpy.array.')
    if YA.dtype.kind != 'f' and YA.dtype.kind != 'i' and YA.dtype.kind != 'u':
        raise ValueError('"YA" must contain floats or integers.')

    if not isinstance(YB, np.ndarray):
        raise ValueError('"YB" must be a numpy.array.')
    if YB.dtype.kind != 'f' and YB.dtype.kind != 'i' and YB.dtype.kind != 'u':
        raise ValueError('"YB" must contain floats or integers.')

    if not isinstance(YC, np.ndarray):
        raise ValueError('"YC" must be a numpy.array.')
    if YC.dtype.kind != 'f' and YC.dtype.kind != 'i' and YC.dtype.kind != 'u':
        raise ValueError('"YC" must contain floats or integers.')

    if not isinstance(M, (int, np.int8, np.int16, np.int32, np.int64)):
        raise ValueError('"M" must be scalar and integer.')
    if M <= 0:
        raise ValueError('"M" must be positive.')

    Na = YA.shape
    Nb = YB.shape
    Nc = YC.shape

    if len(Na) == 2:
        if Na[1] != 1:
            raise ValueError('"YA" must be of size (N, ) or (N,1).')
        YA = YA.flatten()
    elif len(Na) != 1:
        raise ValueError('"YA" must be of size (N, ) or (N,1).')
    N = Na[0]

    if len(Nb) == 2:
        if Nb[1] != 1:
            raise ValueError('"YB" must be of size (N, ) or (N,1).')
        YB = YB.flatten()
    elif len(Nb) != 1:
        raise ValueError('"YB" must be of size (N, ) or (N,1).')
    if Nb[0] != N:
        raise ValueError('"YA" and "YB" must have the same number of elements.')

    if len(Nc) == 2:
        if Nc[1] != 1:
            raise ValueError('"YC" must be of size (N*M, ) or (N*M,1).')
        YC = YC.flatten()
    elif len(Nc) != 1:
        raise ValueError('"YC" must be of size (N*m, ) or (N*M,1).')
    tmp = Nc[0]/N
    if tmp != M:
        raise ValueError('number of rows in input "YC" must be M*N, ' +
                         'where N is the number of rows in "YA"')

    ###########################################################################
    # Check optional inputs
    ###########################################################################
    if not isinstance(Nboot, (int, np.int8, np.int16, np.int32, np.int64)):
        raise ValueError('"Nboot" must be scalar and integer.')
    if Nboot < 0:
        raise ValueError('"Nboot" must be positive.')

    if not isinstance(dummy, bool):
        raise ValueError('"dummy" must be boolean.')

    ###########################################################################
    # Compute indices
    ###########################################################################

    YC = np.reshape(YC, (M, N)) # each row of Y is used for the computation of
    # indices for a different input. We reshape YC so to iterate on the rows of
    # YC to calculate the indices (with numpy this should optimize the
    # computation time compared to an iteration over the columns).

    # Print to screen a warning message if any NAN was found in YA,YB or YC:
    if np.sum(np.isnan(YA)) > 0:
        warn('%d' % (np.sum(np.isnan(YA))) + ' NaNs were found in YA')
    if np.sum(np.isnan(YB)) > 0:
        warn('%d' % (np.sum(np.isnan(YB))) + ' NaNs were found in YB')
    if np.sum(np.isnan(YC.flatten())) > 0:
        warn('%d' % (np.sum(np.isnan(YC))) + ' NaNs were found in YC')

    if Nboot > 1: # Use bootstrapping
        bootsize = N # size of bootstrap resamples
        B = np.random.randint(N, size=(bootsize, Nboot))

        Si = np.nan * np.ones((Nboot, M))
        STi = np.nan * np.ones((Nboot, M))
        Sdummy = np.nan * np.ones((Nboot,))
        STdummy = np.nan * np.ones((Nboot,))
        idxSi = np.nan * np.ones((Nboot, M))
        idxSTi = np.nan * np.ones((Nboot, M))
        idxdummy = np.nan * np.ones((Nboot,))

        for n in range(Nboot):
            Si[n, :], STi[n, :], Sdummy[n], STdummy[n],\
            idxSi[n, :], idxSTi[n, :], idxdummy[n] = \
                    compute_indices(YA[B[:, n]], YB[B[:, n]], YC[:, B[:, n]])

        # Print to screen a warning message if any NAN was found in YA,YB. YC:
        if np.sum(np.isnan(YA)) + np.sum(np.isnan(YB)) + np.sum(np.isnan(YC.flatten())):

            str_Si = "\nX%d: %1.0f" % (1, np.mean(idxSi[:, 0]))
            str_STi = "\nX%d: %1.0f" % (1, np.mean(idxSTi[:, 0]))

            for i in range(1, M):
                str_Si = str_Si + "\nX%d: %1.0f" % (i+1, np.mean(idxSi[:, i]))
                str_STi = str_STi + "\nX%d: %1.0f" % (i+1, np.mean(idxSTi[:, i]))
            if dummy:
                str_Si = str_Si + "\ndummy: %1.0f" % (np.mean(idxdummy))
                str_STi = str_STi + "\ndummy: %1.0f" % (np.mean(idxdummy))

            warn('\n The average number of samples that could be used to '+
                  'approximate main effects (Si) is:' + str_Si)
            warn('\n The average number of samples that could be used to '+
                  'approximate total effects (STi) is:' + str_STi + '\n')

    else:
        Si, STi, Sdummy, STdummy, idxSi, idxSTi, idxdummy = \
                                             compute_indices(YA, YB, YC)

         # Print to screen a warning message if any NAN was found in YA,YB. YC:
        if np.sum(np.isnan(YA)) + np.sum(np.isnan(YB)) + np.sum(np.isnan(YC.flatten())):

            str_Si = "\nX%d: %1.0f" % (1, idxSi[0])
            str_STi = "\nX%d: %1.0f" % (1, idxSTi[0])
            for i in range(1, M, 1):
                str_Si = str_Si + "\nX%d: %1.0f" % (i+1, idxSi[i])
                str_STi = str_STi + "\nX%d: %1.0f" % (i+1, idxSTi[i])
            if dummy:
                str_Si = str_Si + "\ndummy: %1.0f" % (idxdummy)
                str_STi = str_STi + "\ndummy: %1.0f" % (idxdummy)

            warn('\n The number of samples that could be used to '+
                  'approximate main effects (Si) is:' + str_Si)
            warn('\n The number of samples that could be used to '+
                  'approximate total effects (STi) is:' + str_STi + '\n')
    if dummy:
        return Si, STi, Sdummy, STdummy
    else:
        return Si, STi

def compute_indices(YA, YB, YC):

    """This function estimates the variance-based first-order indices
    (or 'main effects') and total-order ('total effects') indices
    (Homma and Saltelli, 1996) by using the approximation technique described
    e.g. in Saltelli et al. (2008) and (2010) for ONE sample/bootstrap resample.

    This function is called internally in VBSA.VBSA_indices.

    Usage:
    Si, STi, Sdummy, STdummy, idxSi, idxSTi, idxdummy = \
                                               VBSA.compute_indices(YA, YB, YC)

    Input:
           YA = set of output samples                    - numpy.ndarray(N, )
           YB = set of output samples                    - numpy.ndarray(N, )
              (independent from YA)
           YC = set of output samples from resampling    - numpy.ndarray(N*M,)

    Output:
           Si = first-order sensitivity indices          - numpy.ndarray (M, )
          STi = total effect sensitivity indices         - numpy.ndarray (M, )
       Sdummy = first-order sensitivity indices          - numpy.ndarray (M, )
                for the dummy input
      STdummy = total effect sensitivity indices         - numpy.ndarray (M, )
                for the dummy input
        idxSi = number of samples that could actually    - numpy.ndarray (M, )
                be used to estimate the main effects
                (after discarding the NaNs)
       idxSTi = number of samples that could actually    - numpy.ndarray (M, )
                be used to estimate the total effects
                (after discarding the NaNs)
        idxSi = number of samples that could actually    - numpy.ndarray (M, )
                be used to estimate the main effects
               (after discarding the NaNs)
     idxdummy = number of samples that could actually    - numpy.ndarray (M, )
                be used to estimate the indices for the
               dummy input (after discarding the NaNs)

    For reference and further details see help of VBSA.VBSA_indices.

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
    Nc = YC.shape
    M = Nc[0]

    nanB = np.isnan(YB)
    nanA = np.isnan(YA)

    f0 = np.mean(YA[~nanA])
    VARy = np.mean(YA[~nanA]**2) - f0**2

    Si = np.nan * np.ones((M,))
    STi = np.nan * np.ones((M,))
    idxSi = np.nan * np.ones((M,))
    idxSTi = np.nan * np.ones((M,))

    for i in range(M):

        yCi = YC[i, :]
        nanC = np.isnan(yCi)

        idx = nanA | nanC #  find indices where either YA or YCi is a NaN
        # and that will need to be excluded from computation

        Si[i] = (np.mean(YA[~idx]*yCi[~idx]) - f0**2) / VARy
        # This is Eq (4.21) in Saltelli et al. (2008)
        # and also Eq. (12) in Saltelli et al. (2010), where the method is
        # attributed to Sobol (1993).

        idxSi[i] = np.sum(~idx) # save number of samples that could be actually
        # used for estimating main effects
        # use 'numpy.sum' instead of 'sum' to spead up the code

        idx = nanB | nanC # find indices where either YB or YCi is a NaN
        # and that will need to be excluded from computation

        STi[i] = 1 - (np.mean(YB[~idx]*yCi[~idx]) - f0**2) / VARy
        # This is Eq (4.22) in Saltelli et al. (2008)

        idxSTi[i] = np.sum(~idx) # save number of samples that could be actually
        # used for estimating total effects
        # use 'numpy.sum' instead of 'sum' to spead up the code

    # Compute indices for the dummy parameter:
    idx = nanA | nanB

    Sdummy = (np.mean(YA[~idx]*YB[~idx]) - f0**2) / VARy
    # This is Eq (3) and (12) in Khorashadi Zadeh et al. (2017)

    STdummy = 1 - (np.mean(YB[~idx]**2) - f0**2) / VARy
    # This is Eq (4) and (13) in Khorashadi Zadeh et al. (2017)

    idxdummy = np.sum(~idx)

    return Si, STi, Sdummy, STdummy, idxSi, idxSTi, idxdummy

def vbsa_convergence(YA, YB, YC, M, NN, Nboot=0, dummy=False):

    """ This function computes the variance-based first-order indices
    (or 'main effects') and total-order ('total effects') indices
    (Homma and Saltelli, 1996) using sub-samples of the original sample 'Y'
    of increasing size.

    Usage:
    Si, STi  = VBSA.vbsa_convergence(YA, YB, YC, M, NN, Nboot=0)
    Si, STi, STdummy = VBSA.vbsa_convergence(YA, YB, YC, M, NN, Nboot,
                                             dummy=True)

    Input:
       YA = set of output samples                        - numpy.ndarray(N, )
                                                      or - numpy.ndarray(N,1)
       YB = set of output samples                        - numpy.ndarray(N, )
            (independent from YA)                     or - numpy.ndarray(N,1)
       YC = set of output samples from resampling (*)    - numpy.ndarray(N*M,)
                                                      of - numpy.ndarray(N*M,1)
        M = number of input variables                    - integer
       NN = subsample sizes at which indices will be     - numpy.ndarray (R, )
            estimated (max(NN) must not exceed N)

    Optional input:
    Nboot = number of resamples used for                 - integer
            boostrapping (if not specified:
            Nboot=0, i.e. no bootstrapping)
    dummy = if dummy is True, an articial input is       - boolean
            added to the set of inputs and the
            sensitivity indices are calculated for the
            dummy input.
            The sensitivity indices for the dummy
            input are estimates of the approximation
            error of the sensitivity indices and they
            can be used for screening, i.e. to
            separate influential and non-influential
            inputs.

    Output:
         Si = first order sensitivity indices.            if Nboot <= 1:
              at different sample sizes.                  - numpy.ndarray (R,M)
              When bootstrapping is used (i.e. Nboot>0),  if Nboot > 1:
              Si is a list, and Si[j] is a                - list(R elements)
              numpy.ndarray(Nboot,M) of the Nboot
              estimates of Si for the jth sample size.

         STi = total order sensitivity indices.            if Nboot <= 1:
              at different sample sizes.                  - numpy.ndarray (R,M)
              When bootstrapping is used (i.e. Nboot>0),  if Nboot > 1:
              STi is a list, and Si[j] is a                - list(R elements)
              numpy.ndarray(Nboot,M) of the Nboot
              estimates of STi for the jth sample size.

   Optional output (if dummy is True):
     Sdummy = first order sensitivity index for the       if Nboot <= 1:
              dummy input.                                - numpy.ndarray (R,M)
              When bootstrapping is used (i.e. Nboot>0),  if Nboot > 1:
              Sdummy is a list, and Sdummy[j] is a        - list(R elements)
              numpy.ndarray(Nboot,M) of the Nboot
              estimates of Si for the jth sample size.

    STdummy = total order sensitivity index for the       if Nboot <= 1:
              dummy input.                                - numpy.ndarray (R,M)
              When bootstrapping is used (i.e. Nboot>0),  if Nboot > 1:
              Sdummy is a list, and Sdummy[j] is a        - list(R elements)
              numpy.ndarray(Nboot,M) of the Nboot
              estimates of Si for the jth sample size.

    SEE ALSO VBSA.vbsa_indices and VBSA.vbsa_resampling

    NOTE: If the vector Y includes any NaN values, the function will identify
          them and exclude them from further computation. A Warning message
          about the number of discarded NaN elements (and hence the actual
          number of samples used for estimating Si and STi) will be displayed.

    EXAMPLE:

    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.stats as st
    import SAFEpython.VBSA as VB
    import SAFEpython.plot_functions as pf
    from SAFEpython.model_execution import model_execution
    from SAFEpython.sampling import AAT_sampling

    from SAFEpython.ishigami_homma import ishigami_homma_function

    fun_test = ishigami_homma_function
    M = 3
    distr_fun = st.uniform
    xmin = -np.pi
    xmax = np.pi
    distr_par = [xmin, xmax - xmin]
    N = 1000
    samp_strat = 'lhs'
    X = AAT_sampling(samp_strat, M, distr_fun ,distr_par, 2*N)
    [ XA, XB, XC ] = VB.vbsa_resampling(X)
    YA = model_execution(fun_test, XA)
    YB = model_execution(fun_test, XB)
    YC = model_execution(fun_test, XC)
    NN = np.linspace(N/10,N,10).astype(int)
    Si, STi = VB.vbsa_convergence(YA, YB, YC, M, NN)
    plt.figure()
    plt.subplot(121)
    pf.plot_convergence(Si, NN*(M+2))
    plt.subplot(122)
    pf.plot_convergence(STi, NN*(M+2))


    REFERENCES:

    Homma, T. and A., Saltelli (1996). Importance measures in global
    sensitivity analysis of nonlinear models.
    Reliability Engineering & System Safety, 52(1), 1-17.

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
    https://www.safetoolbox.info"""

    ###########################################################################
    # Check inputs
    ###########################################################################
    # Check YA and YC to recover the number of input factors M
    if not isinstance(YA, np.ndarray):
        raise ValueError('"YA" must be a numpy.array.')

    if not isinstance(YC, np.ndarray):
        raise ValueError('"YC" must be a numpy.array.')

    if not isinstance(M, (int, np.int8, np.int16, np.int32, np.int64)):
        raise ValueError('"M" must be scalar and integer.')
    if M <= 0:
        raise ValueError('"M" must be positive.')

    Na = YA.shape
    Nc = YC.shape

    if len(Na) == 2:
        if Na[1] != 1:
            raise ValueError('"YA" must be of size (N, ) or (N,1).')
        YA = YA.flatten()
    elif len(Na) != 1:
        raise ValueError('"YA" must be of size (N, ) or (N,1).')
    N = Na[0]

    if len(Nc) == 2:
        if Nc[1] != 1:
            raise ValueError('"YC" must be of size (N*M, ) or (N*M,1).')
        YC = YC.flatten()
    elif len(Nc) != 1:
        raise ValueError('"YC" must be of size (N*m, ) or (N*M,1).')
    tmp = Nc[0]/N
    if tmp != M:
        raise ValueError('number of rows in input "YC" must be M*N, '+
                         'where N is the number of rows in "YA"')

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
    # Check optional inputs
    ###########################################################################
    if not isinstance(Nboot, (int, np.int8, np.int16, np.int32, np.int64)):
        raise ValueError('"Nboot" must be scalar and integer.')
    if Nboot < 0:
        raise ValueError('"Nboot" must be positive.')

    if not isinstance(dummy, bool):
        raise ValueError('"dummy" must be boolean.')

    ###########################################################################
    # Compute indices
    ###########################################################################

    R = len(NN)
    # Intialise variables
    Si = [np.nan]*R
    STi = [np.nan]*R
    Sdummy = [np.nan]*R
    STdummy = [np.nan]*R

    YC = np.reshape(YC, (M, N))

    for j in range(R): # loop over sample sizes

        # Option 1: randomly select a subset of NN(j) without replacement
        # (which is our recommendation in particular for small sample size)
        replace = False
        # Option 2: resample with replacement
        #replace = True

        idx = np.random.choice(np.arange(0, N, 1), size=(NN[j], ), replace=replace)

        YAj = YA[idx]
        YBj = YB[idx]
        YCj = YC[:, idx]

        Si[j], STi[j], Sdummy[j], STdummy[j] = \
                         vbsa_indices(YAj, YBj, YCj.flatten(), M, Nboot, dummy=True)

    if Nboot <= 1: # return a numpy.ndarray (R, M)
        Si_tmp = np.nan*np.ones((R, M))
        STi_tmp = np.nan*np.ones((R, M))
        Sdummy_tmp = np.nan*np.ones((R, M))
        STdummy_tmp = np.nan*np.ones((R, M))

        for j in range(R):
            Si_tmp[j, :] = Si[j]
            STi_tmp[j, :] = STi[j]
            Sdummy_tmp[j, :] = Sdummy[j]
            STdummy_tmp[j, :] = STdummy[j]

        Si = Si_tmp
        STi = STi_tmp
        Sdummy = Sdummy_tmp
        STdummy = STdummy_tmp

    if dummy:
        return Si, STi, Sdummy, STdummy
    else:
        return Si, STi
