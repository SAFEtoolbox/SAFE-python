"""
    Module to simulate the HyMod model

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

    Boyle, D. (2001). Multicriteria calibration of hydrological models.
    PhD thesis, Dep. of Hydrol. and Water Resour., Univ. of Ariz., Tucson.

    Wagener, T., Boyle, D., Lees, M., Wheater, H., Gupta, H., and Sorooshian,
    S. (2001). A framework for development and application of hydrological
    models. Hydrol. Earth Syst. Sci., 5, 13-26.
"""
from __future__ import division, absolute_import, print_function

import numpy as np
from numba import jit # the function jit allows to compile the code and reduced
# the running time


from SAFEpython.util import NSE, RMSE

@jit
def hymod_sim(param, rain, ept):

    """This function simulates the Hymod rainfall-runoff model
    (Boyle, 2001; Wagener et al., 2001).

    Usage:
        Q_sim, STATES, FLUXES = HyMod.simulate(param, rain, ept)

    Input:
     param = 5 elements vector of model parameters         - numpy.ndarray(5, )
             1. Sm   = maximum soil moisture [mm]
             2. beta = exponent in the soil moisture
                       routine [-]
             3. alfa = partition coefficient [-]
             4. Rs   = slow reservoir coefficient [1/Dt]
             5. Rf   = fast reservoir coefficient [1/Dt]
      rain = time series of rainfall                       - numpy.ndarray(T, )
      ept = time series of potential evaporation           - numpy.ndarray(T, )

    Output:
     Q_sim = time series of simulated flow                 - numpy.ndarray(T, )
    STATES = time series of simulated states               - numpy.ndarray(T,5)
    FLUXES = time series of simulated fluxes               - numpy.ndarray(T,4)
             (see comments in the code for the definition of state and flux
             variables)

    Recommended parameter values:  Smax should fall in [0,400] (mm)
                                   beta should fall in [0,2] (-)
                                   alfa should fall in [0,1] (-)
                                     Rs should fall in [0,k] (-)
                                     Rf should fall in [k,1] (-) with 0 < k < 1
    References:

    Boyle, D. (2001). Multicriteria calibration of hydrological models.
    PhD thesis, Dep. of Hydrol. and Water Resour., Univ. of Ariz., Tucson.

    Wagener, T., Boyle, D., Lees, M., Wheater, H., Gupta, H., and Sorooshian,
    S. (2001). A framework for development and application of hydrological
    models. Hydrol. Earth Syst. Sci., 5, 13-26.

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
    # Recover model parameters
    ###########################################################################
    Sm = max(np.spacing(1), param[0]) # Maximum Soil Moisture (cannot
                                              # be zero! ) See lines 35 and 37)
    beta = param[1] # Exponential parameter in soil routine [-]
    alfa = param[2] # Partitioning factor [-]
    Rs = param[3] # Slow reservoir outflow coefficient (ratio) [1/Dt]
    Rf = param[4] # Fast reservoir outflow coefficient (ratio) [1/Dt]

    T = len(rain)

    ###########################################################################
    # Initialise variables
    ###########################################################################
    Pe = np.zeros((T, )) # Recharge from the soil [mm/Dt]
    Ea = np.zeros((T, )) # Actual Evapotranspiration [mm/Dt]
    sm = np.zeros((T+1, )) # Soil Moisture [mm]
    sL = np.zeros((T+1, )) # Slow reservoir moisture [mm]
    sF1 = np.zeros((T+1, )) # Fast reservoir 1 moisture [mm]
    sF2 = np.zeros((T+1, )) # Fast reservoir 2 moisture [mm]
    sF3 = np.zeros((T+1, )) # Fast reservoir 3 moisture [mm]
    QsL = np.zeros((T, )) # Slow flow [mm/Dt]
    QsF = np.zeros((T, )) # Fast flow [mm/Dt]

    for t in range(T):
    ###########################################################################
    # Soil moisture dynamics
    ###########################################################################
        F = 1 - (1-sm[t]/Sm)**beta
        Pe[t] = F * rain[t] # Compute the value of the outflow
        # (we assumed that this process is faster than evaporation)
        sm_temp = max(min(sm[t] + rain[t] - Pe[t], Sm), 0)
        # Compute the water balance with the value of the outflow
        Pe[t] = Pe[t] + max(sm[t] + rain[t] - Pe[t] - Sm, 0) + \
                min(sm[t] + rain[t] - Pe[t], 0)
        # Adjust Pe by an amount equal to the possible negative sm amount or
        # to the possible sm amount above Sm.

        W = min(np.abs(sm[t]/Sm), 1) # Correction factor for evaporation
        Ea[t] = W * ept[t] # Compute the evaporation
        sm[t+1] = max(min(sm_temp - Ea[t], Sm), 0) # Compute the water balance
        Ea[t] = Ea[t] + max(sm_temp - Ea[t] - Sm, 0) + min(sm_temp - Ea[t], 0)
        # Adjust Ea by an amount equal to the possible negative sm amount or to
        # the possible sm amount above Sm.

    ###########################################################################
    # Groundwater dynamics
    ###########################################################################
        # slow flow
        QsL[t] = Rs * sL[t]
        sL[t+1] = sL[t] + (1-alfa)*Pe[t] - QsL[t]
        # fast flow
        sF1[t+1] = sF1[t] +  alfa*Pe[t] - Rf*sF1[t]
        sF2[t+1] = sF2[t] +  Rf*sF1[t] - Rf*sF2[t]
        sF3[t+1] = sF3[t] +  Rf*sF2[t] - Rf*sF3[t]
        QsF[t] = Rf * sF3[t]

    Q_sim = QsL + QsF
    STATES = np.column_stack((sm, sL, sF1, sF2, sF3))
    FLUXES = np.column_stack((Pe, Ea, QsL, QsF))

    return Q_sim, STATES, FLUXES


def hymod_nse(x, rain, ept, flow, warmup):

    """This function runs the rainfall-runoff Hymod model
    and returns the associated Nash-Sutcliffe Efficiency

    y, Q_sim, STATES, FLUXES = HyMod.hymod_nse(x, rain, ept, flow, warmup)

    Input:
         x = 5 elements vector of model parameters         - numpy.ndarray(5, )
             (Smax, beta, alfa, Rs, Rf)
      rain = time series of rainfall                       - numpy.ndarray(T, )
      ept = time series of potential evaporation          - numpy.ndarray(T, )
      flow = time series of observed flow                  - numpy.ndarray(T, )
    warmup =  number of time steps for model warm-up       - integer

    Output:
         y = Nash-Sutcliffe Efficiency                     - numpy.ndarray(1, )
     Q_sim = time series of simulated flow                 - numpy.ndarray(T, )
    STATES = time series of simulated storages             - numpy.ndarray(T,5)
             (all in mm)
    FLUXES = time series of simulated fluxes               - numpy.ndarray(T,4)
            (all in mm/Dt)

    See also HyMod.hymod_sim about the model parameters, simulated variables,
    and references."""

    ###########################################################################
    # Check inputs
    ###########################################################################
    M = 5 # number of model parameters
    if not isinstance(x, np.ndarray):
        raise ValueError('"x" must be a numpy.array.')
    if x.dtype.kind != 'f' and x.dtype.kind != 'i' and x.dtype.kind != 'u':
        raise ValueError('"x" must contain floats or integers.')
    Nx = x.shape
    if len(Nx) != 1 or len(x) != M:
        raise ValueError('"x" must have shape (5, ).')

    if not isinstance(rain, np.ndarray):
        raise ValueError('"rain" must be a numpy.array.')
    if rain.dtype.kind != 'f' and rain.dtype.kind != 'i' and rain.dtype.kind != 'u':
        raise ValueError('"rain" must contain floats or integers.')
    Nrain = rain.shape
    if len(Nrain) != 1:
        raise ValueError('"rain" must be of shape (T, ).')
    T = Nrain[0]

    if not isinstance(ept, np.ndarray):
        raise ValueError('"ept" must be a numpy.array.')
    if ept.dtype.kind != 'f' and ept.dtype.kind != 'i' and ept.dtype.kind != 'u':
        raise ValueError('"ept" must contain floats or integers.')
    Nept = ept.shape
    if len(Nept) != 1:
        raise ValueError('"ept" must be of shape (T, ).')
    if len(ept) != T:
        raise ValueError('"ept" and "prec" must have the same number of elements.')

    if not isinstance(flow, np.ndarray):
        raise ValueError('"flow" must be a numpy.array.')
    if flow.dtype.kind != 'f' and flow.dtype.kind != 'i' and flow.dtype.kind != 'u':
        raise ValueError('"flow" must contain floats or integers.')
    Nflow = flow.shape
    if len(Nflow) != 1:
        raise ValueError('"flow" must be of shape (T, ).')
    if len(flow) != T:
        raise ValueError('"flow" and "rain" must have the same number of elements.')

    if not isinstance(warmup, (int, np.int8, np.int16, np.int32, np.int64)):
        raise ValueError('"warmup" must be scalar and integer.')
    if warmup < 0 or warmup >= T:
        raise ValueError('"warmup" must be in [0, T).')

    ###########################################################################
    # Simulate HyMod and compute scalar output
    ###########################################################################
    Q_sim, STATES, FLUXES = hymod_sim(x, rain, ept)

    Qs = Q_sim[warmup:len(Q_sim)+1]
    Qo = flow[warmup:len(Q_sim)+1]
    y = NSE(Qs, Qo)

    return y, Q_sim, STATES, FLUXES


def hymod_max(x, rain, ept, warmup):

    """This function runs the rainfall-runoff Hymod model
    and returns the maximum flow in the simulated time series

    y, Q_sim, STATES, FLUXES = HyMod.hymod_max(x, rain, ept, warmup)

    Input:
         x = 5 elements vector of model parameters         - numpy.ndarray(5, )
             (Smax,beta,alfa,Rs,Rf)
      rain = time series of rainfall                       - numpy.ndarray(T, )
      ept = time series of potential evaporation          - numpy.ndarray(T, )
    warmup =  number of time steps for model warm-up       - integer

    Output:
         y = maximum flow over the simulation horizon      - numpy.ndarray(1, )
     Q_sim = time series of simulated flow                 - numpy.ndarray(T, )
    STATES = time series of simulated storages             - numpy.ndarray(T,5)
             (all in mm)
    FLUXES = time series of simulated fluxes               - numpy.ndarray(T,4)
            (all in mm/Dt)

    See also hymod_sim about the model parameters, simulated variables,
    and references."""

    ###########################################################################
    # Check inputs
    ###########################################################################
    M = 5 # number of model parameters
    if not isinstance(x, np.ndarray):
        raise ValueError('"x" must be a numpy.array.')
    if x.dtype.kind != 'f' and x.dtype.kind != 'i' and x.dtype.kind != 'u':
        raise ValueError('"x" must contain floats or integers.')
    Nx = x.shape
    if len(Nx) != 1 or len(x) != M:
        raise ValueError('"x" must have shape (5, ).')

    if not isinstance(rain, np.ndarray):
        raise ValueError('"rain" must be a numpy.array.')
    if rain.dtype.kind != 'f' and rain.dtype.kind != 'i' and rain.dtype.kind != 'u':
        raise ValueError('"rain" must contain floats or integers.')
    Nrain = rain.shape
    if len(Nrain) != 1:
        raise ValueError('"rain" must be of shape (T, ).')
    T = Nrain[0]

    if not isinstance(ept, np.ndarray):
        raise ValueError('"ept" must be a numpy.array.')
    if ept.dtype.kind != 'f' and ept.dtype.kind != 'i' and ept.dtype.kind != 'u':
        raise ValueError('"ept" must contain floats or integers.')
    Nept = ept.shape
    if len(Nept) != 1:
        raise ValueError('"ept" must be of shape (T, ).')
    if len(ept) != T:
        raise ValueError('"ept" and "prec" must have the same number of elements.')

    if not isinstance(warmup, (int, np.int8, np.int16, np.int32, np.int64)):
        raise ValueError('"warmup" must be scalar and integer.')
    if warmup < 0 or warmup >= T:
        raise ValueError('"warmup" must be in [0, T).')

    ###########################################################################
    # Simulate HyMod and compute scalar output
    ###########################################################################
    Q_sim, STATES, FLUXES = hymod_sim(x, rain, ept)

    y = np.array(np.max(Q_sim[warmup:len(Q_sim)+1]))

    return y, Q_sim, STATES, FLUXES


def hymod_MulObj(x, rain, ept, flow, warmup):

    """This function runs the rainfall-runoff Hymod model
    and returns 2 metrics of model performance: RMSE and BIAS

    y, Q_sim, STATES, FLUXES = HyMod.hymod_MulObj(x, rain, ept, flow, warmup)

    Input:
         x = 5 elements vector of model parameters         - numpy.ndarray(5, )
             (Smax,beta,alfa,Rs,Rf)
      rain = time series of rainfall                       - numpy.ndarray(T, )
      ept = time series of potential evaporation          - numpy.ndarray(T, )
      flow = time series of observed flow                  - numpy.ndarray(T, )
    warmup =  number of time steps for model warm-up       - integer

    Output:
        y = vector of objective functions (RMSE,BIAS)      - numpy.ndarray(2, )
     Q_sim = time series of simulated flow                 - numpy.ndarray(T, )
    STATES = time series of simulated storages             - numpy.ndarray(T,5)
             (all in mm)
    FLUXES = time series of simulated fluxes               - numpy.ndarray(T,4)
            (all in mm/Dt)

    See also hymod_sim about the model parameters, simulated variables,
    and references."""

    ###########################################################################
    # Check inputs
    ###########################################################################
    M = 5 # number of model parameters
    if not isinstance(x, np.ndarray):
        raise ValueError('"x" must be a numpy.array.')
    if x.dtype.kind != 'f' and x.dtype.kind != 'i' and x.dtype.kind != 'u':
        raise ValueError('"x" must contain floats or integers.')
    Nx = x.shape
    if len(Nx) != 1 or len(x) != M:
        raise ValueError('"x" must have shape (5, ).')

    if not isinstance(rain, np.ndarray):
        raise ValueError('"rain" must be a numpy.array.')
    if rain.dtype.kind != 'f' and rain.dtype.kind != 'i' and rain.dtype.kind != 'u':
        raise ValueError('"rain" must contain floats or integers.')
    Nrain = rain.shape
    if len(Nrain) != 1:
        raise ValueError('"rain" must be of shape (T, ).')
    T = Nrain[0]

    if not isinstance(ept, np.ndarray):
        raise ValueError('"ept" must be a numpy.array.')
    if ept.dtype.kind != 'f' and ept.dtype.kind != 'i' and ept.dtype.kind != 'u':
        raise ValueError('"ept" must contain floats or integers.')
    Nept = ept.shape
    if len(Nept) != 1:
        raise ValueError('"ept" must be of shape (T, ).')
    if len(ept) != T:
        raise ValueError('"ept" and "prec" must have the same number of elements.')

    if not isinstance(flow, np.ndarray):
        raise ValueError('"flow" must be a numpy.array.')
    if flow.dtype.kind != 'f' and flow.dtype.kind != 'i' and flow.dtype.kind != 'u':
        raise ValueError('"flow" must contain floats or integers.')
    Nflow = flow.shape
    if len(Nflow) != 1:
        raise ValueError('"flow" must be of shape (T, ).')
    if len(flow) != T:
        raise ValueError('"flow" and "rain" must have the same number of elements.')

    if not isinstance(warmup, (int, np.int8, np.int16, np.int32, np.int64)):
        raise ValueError('"warmup" must be scalar and integer.')
    if warmup < 0 or warmup >= T:
        raise ValueError('"warmup" must be in [0, T).')

    ###########################################################################
    # Simulate HyMod and compute scalar output
    ###########################################################################
    Q_sim, STATES, FLUXES = hymod_sim(x, rain, ept)

    Qs = Q_sim[warmup:len(Q_sim)+1]
    Qo = flow[warmup:len(Q_sim)+1]

    y = np.nan * np.ones((2,))
    y[0] = RMSE(Qs, Qo) # RMSE
    y[1] = np.abs(np.mean(Qs - Qo)) # BIAS

    return y, Q_sim, STATES, FLUXES
