"""
    Module to simulate the HBV model

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

     Seibert, J.(1997), Estimation of Parameter Uncertainty in the HBV Model,
     Nordic Hydrology, 28(4/5), 247-262.
"""
from __future__ import division, absolute_import, print_function

import numpy as np
from numba import jit # the function jit allows to compile the code and reduced
# the running time

from SAFEpython.util import RMSE, NSE

@jit
def snow_routine(param, temp, prec):

    """This function simulates a simple, conceptual snow accumulation/melting
    model based on a degree day approach.

    Usage:
        P, STATES, FLUXES = HBV.snow_routine(param, temp, prec)

    Input:
     param = model parameters                              - numpy.ndarray(4, )
               1. Ts    = threshold temperature [C]
               2. CFMAX = degree day factor [mm/C]
               3. CFR   = refreezing factor [-]
               4. CWH   = water holding capacity of snow [-]
      temp = time series of temperature                    - numpy.ndarray(T, )
      prec = time series of precipitation                  - numpy.ndarray(T, )

    Output:
         P = time series of simulated flow exiting from    - numpy.ndarray(T, )
             the snowpack (as a result of melt-refreezing)
             [mm/Dt]
    STATES = time series of simulated storages (all in mm) - numpy.ndarray(T,2)
             1st column: water content of snowpack
                         (snow component)
             2nd column: water content of snowpack
                         (liquid component)
    FLUXES = time series of simulated fluxes (all in mm/Dt)- numpy.ndarray(T,2)
             1st column: refreezing
             2nd column: snowmelt

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
    # Recover model parameters
    ###########################################################################

    Ts = param[0] # Threshold temperature [C]
    CFMAX = param[1] # Degree day factor [mm/C]
    CFR = param[2] # Refreezing factor [-]
    CWH = param[3] # Water holding capacity of snow [-]

    T = len(prec) # number of time samples

    ###########################################################################
    # Initialise variables
    ###########################################################################
    P = np.zeros((T, )) # snowmelt leaving the snowpack/recharge to the soil [mm/Dt]
    rain = np.zeros((T, )); rain[temp >= Ts] = prec[temp >= Ts] # [mm/Dt]
    snow = np.zeros((T, )); snow[temp < Ts] = prec[temp < Ts] # [mm/Dt]
    Ta = temp - Ts; Ta[temp < Ts] = 0 # Active Temperature for snowmelt
    Tn = Ts - temp; Tn[temp >= Ts] = 0 #Active Temperature for refreezing
    m = np.zeros((T, )) # Snowmelt [mm/Dt]
    rfz = np.zeros((T, )) # Refreezing [mm/Dt]
    v = np.zeros((T+1, )) # Snowpack depth [mm]: solid component
    vl = np.zeros((T+1, ))# Snowpack depth [mm]: liquid component

    ###########################################################################
    # Snow pack routine
    ###########################################################################
    for t in range(T):

        m[t] = min(CFMAX*Ta[t], v[t])
        rfz[t] = min(CFR*CFMAX*Tn[t], vl[t])
        # snowpack dynamics: solid component
        v[t+1] = v[t]- m[t] + snow[t] + rfz[t]
        # snowpack dynamics: liquid component
        vl[t+1] = vl[t] + m[t] + rain[t] - rfz[t]
        if vl[t+1] > CWH*v[t+1]: # if the liquid component exceed the snow pack
                                 # holding capacity
            P[t] = vl[t+1] - CWH*v[t+1]
            vl[t+1] = CWH*v[t+1]
        else:
            P[t] = 0

    STATES = np.column_stack((v, vl))
    FLUXES = np.column_stack((rfz, m))

    return P, STATES, FLUXES


@jit
def hbv_sim(param, P, ept, Case, ini):

    """This function simulates the HBV rainfall-runoff model (Seibert, 1997).

    Usage:
        Q_sim, STATES, FLUXES = HBV.hbv_sim(param, P, ept, Case, ini)

    Input:
      param = vector of model parameters                   - numpy.ndarray(9, )
              1. BETA   = Exponential parameter in soil
                          routine [-]
              2. LP     = evapotranspiration limit [-]
              3. FC     = field capacity [mm]
              4. PERC   = maximum flux from Upper to Lower
                          Zone [mm/Dt]
              5. K0     = near surface flow coefficient
                          (ratio) [1/Dt]
              6. K1     = upper Zone outflow coefficient
                          (ratio) [1/Dt]
              7. K2     = lower Zone outflow coefficient
                          (ratio) [1/Dt]
              8. UZL    = near surface flow threshold [mm]
              9. MAXBAS = flow routing coefficient [Dt]
          P = time series of effective precipitation       - numpy.ndarray(T, )
              reaching the ground (i.e. precipitation -
               snow accumulation + snowmelt )
        ept = time series of potential evapotranspiration  - numpy.ndarray(T, )
       Case = flag for preferred path in the Upper Zone    - scalar
              dynamics
            flag=1 -> Preferred path is runoff
            flag=2 -> Preferred path is percolation

    Output:
      Q_sim = time series of simulated flow (in mm)        - numpy.ndarray(T, )
     STATES = time series of simulated storages            - numpy.ndarray(T,3)
              (all in mm)
              1: water content of soil (soil moisture)
              2. water content of upper reservoir of flow
                 routing routine
              3. water content of lower reservoir of flow  - numpy.ndarray(T,5)
                 routing routine
     FLUXES = time series of simulated fluxes
              (all in mm/Dt)
              1: actual evapotranspiration
              2: recharge (water flux from soil moisture
                 accounting module to flow routing module)
              3: percolation (water flux from upper to
                 lower reservoir of the  flow routing module)
              4: runoff from upper reservoir
              5: runoff from lower reservoir

     References:

     Seibert, J.(1997), Estimation of Parameter Uncertainty in the HBV Model,
     Nordic Hydrology, 28(4/5), 247-262.

     Comments:
     * The Capillary flux (from upper tank to soil moisture accounting module)
     is not considered
     * The recharge from the soil to the upper zone is considered to be a
     faster process than evapotranspiration.
     * The preferential path from the upper zone can be modified
               - Case 1: interflow is dominant
               - Case 2: percolation is dominant
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
    BETA = param[0] # Exponential parameter in soil routine [-]
    LP = param[1] # Evapotranspiration limit [-]
    FC = max(np.spacing(1), param[2]) # Field capacity [mm] cannot be zero

    PERC = param[3] # Maximum flux from Upper to Lower Zone [mm/Dt]
    K0 = param[4] # Near surface flow coefficient (ratio) [1/Dt]
    K1 = param[5] # Upper Zone outflow coefficient (ratio) [1/Dt]
    K2 = param[6] # Lower Zone outflow coefficient (ratio) [1/Dt]
    UZL = param[7] # Near surface flow threshold [mm]

    MAXBAS = max(1, int(param[8])) # Flow routing coefficient [Dt]

    T = len(ept) # number of time samples

    ###########################################################################
    # Initialise variables
    ###########################################################################
    EA = np.zeros((T, )) # Actual Evapotranspiration [mm/Dt]
    SM = np.zeros((T+1, )) # Soil Moisture [mm]
    R = np.zeros((T, )) # Recharge (water flow from Soil to Upper Zone) [mm/Dt]
    UZ = np.zeros((T+1, )) # Upper Zone moisture [mm]
    LZ = np.zeros((T+1, )) # Lower Zone moisture [mm]
    RL = np.zeros((T, )) # Recharge to the lower zone [mm]
    Q0 = np.zeros((T, )) # Outflow from Upper Zone [mm/Dt]
    Q1 = np.zeros((T, )) # Outflow from Lower Zone [mm/Dt]

    # Set intial states
    SM[0] = ini[0]
    UZ[0] = ini[1]
    LZ[0] = ini[2]

    ###########################################################################
    # Soil, lower and upper zone routine
    ###########################################################################
    for t in range(T):

        #######################################################################
        # Soil moisture dynamics
        #######################################################################
        R[t] = P[t] * (SM[t]/FC)**BETA  # Compute the value of the recharge to the
        # upper zone (we assumed that this process is faster than evaporation)
        SM_dummy = max(min(SM[t] + P[t] - R[t], FC), 0) # Compute the water balance
        # with the value of the recharge
        R[t] = R[t] + max(SM[t] + P[t] - R[t] - FC, 0) + min(SM[t] + P[t] - R[t], 0)
        # adjust R by an amount equal to the possible negative SM amount or to
        # the possible SM amount above FC

        EA[t] = ept[t] * min(SM_dummy/(FC*LP), 1) # Compute the evaporation
        SM[t+1] = max(min(SM_dummy - EA[t], FC), 0) # Compute the water balance

        EA[t] = EA[t] + max(SM_dummy - EA[t] - FC, 0) + min(SM_dummy - EA[t], 0)
        # adjust EA by an amount equal to the possible negative SM amount or to
        # the possible SM amount above FC

        #######################################################################
        # Upper zone dynamics
        #######################################################################
        if Case == 1:
            # Case 1: Preferred path = runoff from the upper zone
            Q0[t] = max(min(K1*UZ[t] + K0*max(UZ[t] - UZL, 0), UZ[t]), 0)
            RL[t] = max(min(UZ[t] - Q0[t], PERC), 0)

        elif Case == 2:
            # Case 2: Preferred path = percolation
            RL[t] = max(min(PERC, UZ[t]), 0)
            Q0[t] = max(min(K1*UZ[t] + K0*max(UZ[t] - UZL, 0), UZ[t] - RL[t]), 0)
        else:
            raise ValueError('Case must equal to 1 or 2 ')

        UZ[t+1] = UZ[t] + R[t] - Q0[t] - RL[t]

        #######################################################################
        # Lower zone dynamics
        #######################################################################

        Q1[t] = max(min(K2*LZ[t], LZ[t]), 0)
        LZ[t+1] = LZ[t] + RL[t] - Q1[t]

    Q = Q0 + Q1 # total outflow (mm/Dt)

    ###########################################################################
    # FLow routing routine
    ###########################################################################
    #c = mytrimf(np.arange(1, MAXBAS+1, 1), [0, (MAXBAS+1)/2, MAXBAS+1])
    # make list [0, (MAXBAS+1)/2, MAXBAS+1] must be homogeneous (so that the
    # code works for Python 2):
    c = mytrimf(np.arange(1, MAXBAS+1, 1), [0.0, (MAXBAS+1)/2, float(MAXBAS+1)])
    c = c/np.sum(c) # vector of normalized coefficients - (1,MAXBAS)
    Q_sim = np.zeros((T, ))

    for t in range(MAXBAS, T+1):
        #Q_sim[t-1] = c @ Q[t-MAXBAS:t] # does not work for python 2
        Q_sim[t-1] = np.matmul(c, Q[t-MAXBAS:t])

    STATES = np.column_stack((SM, UZ, LZ))
    FLUXES = np.column_stack((EA, R, RL, Q0, Q1))

    return Q_sim, STATES, FLUXES

@jit
def mytrimf(x, param):
    # implements triangular-shaped membership function
    f = np.zeros(x.shape)
    idx = (x > param[0]) & (x <= param[1])
    f[idx] = (x[idx]-param[0]) / (param[1]-param[0])
    idx = (x > param[1]) & (x <= param[2])
    f[idx] = (param[2]-x[idx]) / (param[2]-param[1])

    return f

def hbv_snow_objfun(x, prec, temp, ept, flow, warmup, Case):

    """This function simulates the snow accumulation/melting process
    (via the function ''snow_routine'') and the rainfall-runoff process
    (via the HBV model by Seibert (1997)) and returns 6 objective functions
    (see Kollat et al, 2012).

    Usage:
    f, Q_sim, STATES, FLUXES = HBV.hbv_snow_objfun(param, prec, temp, ept,
                                               flow, warmup, Case)

    Input:
     param = vector of model parameters                   - numpy.ndarray(13, )
             Snow routine parameters:
               1. Ts     = threshold temperature [C]
               2. CFMAX  = degree day factor [mm/C]
               3. CFR    = refreezing factor [-]
               4. CWH    = Water holding capacity of snow [-]
             HBV parameters:
               5. BETA   = Exponential parameter in soil
                          routine [-]
               6. LP     = evapotranspiration limit [-]
               7. FC     = field capacity [mm]
               8. PERC   = maximum flux from Upper to Lower
                           Zone [mm/Dt]
               9. K0     = Near surface flow coefficient
                           (ratio) [1/Dt]
              10. K1     = Upper Zone outflow coefficient
                           (ratio) [1/Dt]
              11. K2     = Lower Zone outflow coefficient
                          (ratio) [1/Dt]
              12. UZL    = Near surface flow threshold
                           [mm]
              13. MAXBAS = Flow routing coefficient
                          [Dt]
      prec = time series of precipitation                  - numpy.ndarray(T, )
      temp = time series of temperature                    - numpy.ndarray(T, )
       ept = time series of evapotranspiration             - numpy.ndarray(T, )
      flow = time series of observed flow                  - numpy.ndarray(T, )
    warmup = number of time steps for model warm-up        - scalar
      Case = flag [see hbv_sim.m for more details]         - scalar

    Output:
         f = vector of objective functions                 - numpy.ndarray(6, )
             1:AME, 2:NSE, 3:BIAS, 4:TRMSE, 5:SFDCE,
             6:RMSE
     Q_sim = time series of simulated flow                 - numpy.ndarray(T, )
    STATES = time series of simulated storages (all in mm) - numpy.ndarray(T,5)
             1: water content of snowpack
                (snow component)
             2: water content of snowpack
                (liquid component)
             3: water content of soil (soil moisture)
             4. water content of upper reservoir of flow
                routing routine
             5. water content of lower reservoir of flow
                routing routine
    FLUXES = time series of simulated fluxes (all in mm/Dt)- numpy.ndarray(T,8)
             1: refreezing
             2: snowmelt
             3: actual evapotranspiration
             4: recharge (water flux from soil moisture
                accounting module to flow routing module)
             5: percolation (water flux from upper to lower
                reservoir of the flow routing module)
             6: runoff from upper reservoir
             7: runoff from lower reservoir

    References:

    Seibert,J.(1997)."Estimation of Parameter Uncertainty in the HBV Model".
    Nordic Hydrology.28(4/5).247-262.

    Kollat,J.B.,Reed,P.M.,Wagener,T.(2012)."When are multiobjective
    calibration trade-offs in hydrologic models meaningful?". Water resources
    research,VOL.48,W03520,doi:10.1029/2011WR011534.

    This function is part of the SAFE Toolbox by F. Pianosi, F. Sarrazin
    and T. Wagener at Bristol University (2015).
    SAFE is provided without any warranty and for non-commercial use only.
    For more details, see the Licence file included in the root directory
    of this distribution.
    For any comment and feedback, or to discuss a Licence agreement for
    commercial use, please contact: francesca.pianosi@bristol.ac.uk
    For details on how to cite SAFE in your publication, please see:
    https://www.safetoolbox.info"""

    # Comments:
    # * Model components: snow routine (optional)- soil moisture, upper zone
    # and lower routine - flow routing routine

    ###########################################################################
    # Check inputs
    ###########################################################################
    M = 13 # number of model parameters
    if not isinstance(x, np.ndarray):
        raise ValueError('"x" must be a numpy.array.')
    if x.dtype.kind != 'f' and x.dtype.kind != 'i' and x.dtype.kind != 'u':
        raise ValueError('"x" must contain floats or integers.')
    Nx = x.shape
    if len(Nx) != 1 or len(x) != M:
        raise ValueError('"x" must have shape (13, ).')

    if not isinstance(prec, np.ndarray):
        raise ValueError('"prec" must be a numpy.array.')
    if prec.dtype.kind != 'f' and prec.dtype.kind != 'i' and prec.dtype.kind != 'u':
        raise ValueError('"prec" must contain floats or integers.')
    Nprec = prec.shape
    if len(Nprec) != 1:
        raise ValueError('"prec" must be of shape (T, ).')
    T = Nprec[0]

    if not isinstance(ept, np.ndarray):
        raise ValueError('"ept" must be a numpy.array.')
    if ept.dtype.kind != 'f' and ept.dtype.kind != 'i' and ept.dtype.kind != 'u':
        raise ValueError('"ept" must contain floats or integers.')
    Nept = ept.shape
    if len(Nept) != 1:
        raise ValueError('"ept" must be of shape (T, ).')
    if len(ept) != T:
        raise ValueError('"ept" and "prec" must have the same number of elements.')

    if not isinstance(temp, np.ndarray):
        raise ValueError('"temp" must be a numpy.array.')
    if temp.dtype.kind != 'f' and temp.dtype.kind != 'i' and temp.dtype.kind != 'u':
        raise ValueError('"temp" must contain floats or integers.')
    Ntemp = ept.shape
    if len(Ntemp) != 1:
        raise ValueError('"temp" must be of shape (T, ).')
    if len(temp) != T:
        raise ValueError('"temp" and "prec" must have the same number of elements.')

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

    if not isinstance(Case, (int, np.int8, np.int16, np.int32, np.int64)):
        raise ValueError('"Case" must be scalar and integer.')

    ###########################################################################
    # Simulate HBV and compute scalar output
    ###########################################################################
    # Initialise variables
    STATES = np.nan * np.ones((T+1, 5))
    FLUXES = np.nan * np.ones((T, 7))
    f = np.nan * np.ones((6, ))

    # Run the model
    ini = [0, 0, 0] # initial states
    P, STATES[:, 0:2], FLUXES[:, 0:2] = snow_routine(x[0:4], temp, prec)
    Q_sim, STATES[:, 2:5], FLUXES[:, 2:7] = hbv_sim(x[4:13], P, ept, Case, ini)

    # Compute objective functions
    Qs = Q_sim[warmup:len(Q_sim)+1]
    Qo = flow[warmup:len(Q_sim)+1]

    N = len(Qs)

    N_67 = int(np.floor(N*0.67)-1)
    N_33 = int(np.floor(N*0.33)-1)
    Qs_sort = np.sort(Qs)
    Qo_sort = np.sort(Qo)

    Lambda = 0.3
    Zs = ((1+Qs)**Lambda-1)/Lambda
    Zo = ((1+Qo)**Lambda-1)/Lambda

    f[0] = np.mean(np.abs(Qs-Qo)) #  AME (absolute mean error)
    f[1] = NSE(Qs, Qo) #  NSE
    f[2] = np.abs(np.mean(Qs - Qo)) # BIAS
    f[3] = np.sqrt(np.mean((Zs - Zo)**2)) # TRMSE (transformed root mean square error)
    f[4] = np.abs((Qs_sort[N_33]-Qs_sort[N_67]) / (Qo_sort[N_33]-Qo_sort[N_67])-1)*100
    # SFDCE (Slope of Flow Duration Curve)
    f[5] = RMSE(Qs, Qo) # RMSE # RMSE

    return f, Q_sim, STATES, FLUXES
