# -*- coding: utf-8 -*-
"""
A simple mathematical description of the spread of a flu in a company is the 
so-called the flu model, which divides the (fixed) population of N individuals 
into three "compartments" which may vary as a function of time, t:
V(t) are those vulnerable but not yet infected with the flu;
S(t) is the number of sick individuals;
RI(t) are those individuals who either are immune or have recovered from the 
flu and now have immunity to it.
The model describes the change in the population of each of these compartments 
in terms of two parameters, β and γ. 
β describes the effective contact rate of the flu: an infected individual comes 
into contact with βN other individuals per unit time (of which the fraction 
that are susceptible to contracting the flu is V/N). 
γ is the mean recovery rate: that is, 1/γ is the mean period of time during 
which a sick individual can pass it on.
The differential equations describing this model were first derived by Kermack 
and McKendrick [Proc. R. Soc. A, 115, 772 (1927)]:
dV / dt = -βVS / N
 
dS / dt = βVS / N - γS
dRI / dt = γS
@author: Andres Peñuela
"""
from scipy.integrate import odeint
import numpy as np
from numba import njit # the function jit allows to compile the code and reduced

class population:
    
    def __init__(self, N = 100, I_0 = 10, S_0 = 1):
        # Total population, N.
        self.N = N
        # Initial number of sick individuals
        self.S_0 = S_0
        # Initial number of immune individuals
        self.I_0 = I_0
        # Everyone else, S_0, is susceptible to infection initially.
        self.V_0 = N - S_0 - I_0

        
def simulation(t,population, contact, contagion, recovery, vaccination):
    
    y_0 = population.S_0, population.I_0, population.V_0
    
    def deriv(y, t, N, contact, contagion, recovery, vaccination):
        S, RI, V = y
        dVdt = -contact*contagion * V * S / N - np.min([vaccination, vaccination*V])
        dSdt = contact*contagion * V * S / N - recovery * S #- sigma * S
        dRIdt = recovery * S + np.min([vaccination, vaccination*V])
        
        return dSdt, dRIdt, dVdt
    
    ret = odeint(deriv, y_0, t, args=(population.N, contact, contagion, recovery, vaccination))
    S, RI, V = ret.T
    
    return S, RI, V

@njit(parallel = False) # Numba decorator to speed-up the function below
def model(param,t,N):
    RI_0 = param[0]
    S_0 = 1
    contact = param[1]
    contagion = param[2]
    recovery = 1/param[3]
    vaccination = param[4]
    
    V_0 = N - S_0 - RI_0
    
    T = len(t)
    S  = np.zeros(T)
    RI = np.zeros(T)
    V  = np.zeros(T)
    vaccination_num = np.zeros(T)
    
    S[0] = S_0
    RI[0] =  RI_0
    V[0] = V_0
    
    for i in np.arange(T-1):
        V[i+1] = np.array([V[i] - contact*contagion * S[i] * V[i] / N - vaccination_num[i],0]).max()
        RI[i+1] = RI[i] + recovery * S[i] + vaccination_num[i]
        S[i+1] = np.array([S[i] + contact*contagion * S[i] * V[i] / N - recovery * S[i],N]).min()
        vaccination_num[i+1] = np.array([V[i+1], vaccination]).min()
        
    max_value = np.array(np.max(S))
    vaccine_price = 7 # £ per vaccine
    vaccination_cost =  np.sum(vaccination_num) * vaccine_price# + RI_0 * vaccine_price
    social_dist_price = 200 # Price of implementing social distancing and isolation measures
    social_dist_cost = (2 - contact) * social_dist_price
    masks_price = 50 # Price of the distribution of face masks
    masks_cost = (1 - contagion) * masks_price
    treatment_price = 0 # we set it to zero to reflect that it is not an action but an unknown variable
    treatment_cost = (21 - 1/recovery) * treatment_price
    total_cost = np.array(vaccination_cost + \
                          social_dist_cost + masks_cost + \
                          treatment_cost)
        
    return S, RI, V,max_value,total_cost

@njit(parallel = False) # Numba decorator to speed-up the function below        
def function(param,t,N,output):
    RI_0 = param[0]
    S_0 = 1
    contact = param[1]
    contagion = param[2]
    recovery = 1/param[3]
    vaccination = param[4]
    
    V_0 = N - S_0 - RI_0
    
    T = len(t)
    S  = np.zeros(T)
    RI = np.zeros(T)
    V  = np.zeros(T)
    vaccination_num = np.zeros(T)
    
    S[0] = S_0
    RI[0] =  RI_0
    V[0] = V_0
    
    
    
    for i in np.arange(T-1):
        V[i+1] = np.array([V[i] - contact*contagion * S[i] * V[i] / N - vaccination_num[i],0]).max()
        vaccination_num[i+1] = np.array([V[i+1], vaccination]).min()
        RI[i+1] = RI[i] + recovery * S[i] + vaccination_num[i+1]
        S[i+1] = np.array([S[i] + contact*contagion * S[i] * V[i] / N - recovery * S[i],N]).min()
    
    if output == 0: # max number of sick individuals in a day
        out = np.array(np.max(S))
    elif output == 1: # total cost of the measures
        vaccine_price = 7 # £ per vaccine. Cost before the outbreak
        vaccination_cost = np.sum(vaccination_num) * vaccine_price# + RI_0 * vaccine_price 
        social_dist_price = 200 # Price of implementing social distancing and isolation measures
        social_dist_cost = (2 - contact) * social_dist_price
        masks_price = 50 # Price of the distribution of face masks
        masks_cost = (1 - contagion) * masks_price
        treatment_price = 0 # we set it to zero to reflect that it is not an action but an unknown variable
        treatment_cost = (21 - 1/recovery) * treatment_price
        out = np.array(vaccination_cost + \
                       social_dist_cost + masks_cost + \
                       treatment_cost)
        
#    elif output == 2:
#        out = np.array(RI[-1])
#    elif output == 3:
#        out = np.array(V[-1])
    
#    out = np.array(np.max(S))
    
    return out