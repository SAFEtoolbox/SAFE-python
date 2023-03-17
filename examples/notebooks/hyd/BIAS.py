# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 13:38:53 2019

@author: sarrazin
"""

import numpy as np
from numpy.matlib import repmat

def BIAS(y_sim, y_obs):

    """Computes the bias (absolute mean error)

    bias = BIAS(Y_sim,y_obs)

    Y_sim = time series of modelled variable     - matrix (N,T)
            (N>1 different time series can be evaluated at once)
    y_obs = time series of observed variable     - vector (1,T)

    bias   = vector of BIAS coefficients           - vector (N,1)

    """

    Nsim = y_sim.shape
    if len(Nsim) > 1:
        N = Nsim[0]
        T = Nsim[1]
    elif len(Nsim) == 1:
        T = Nsim[0]
        N = 1
        y_sim_tmp = np.nan * np.ones((1,T))
        y_sim_tmp[0,:] = y_sim
        y_sim = y_sim_tmp

    Nobs = y_obs.shape
    if len(Nobs) > 1:
        if Nobs[0] != 1:
             raise ValueError('"y_obs" be of shape (T, ) or (1,T).')
        if Nobs[1] != T:
            raise ValueError('the number of elements in "y_obs" must be equal to the number of columns in "y_sim"')
    elif len(Nobs) == 1:
        if Nobs[0] != T:
            raise ValueError('the number of elements in "y_obs" must be equal to the number of columns in "y_sim"')
        y_obs_tmp = np.nan * np.ones((1,T))
        y_obs_tmp[0,:] = y_obs
        y_obs = y_obs_tmp


    bias = abs(np.mean(y_sim - repmat(y_obs, N, 1), axis=1))

    return bias
