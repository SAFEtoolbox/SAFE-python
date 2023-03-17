# -*- coding: utf-8 -*-
"""
This function is a Python implementation of the predator prey model
@author: Andres Peñuela andres.penuela-fernandez@bristol.ac.uk, penyuela@gmail.com
"""
import numpy as np
from numba import njit # the function jit allows to compile the code and reduced

class predator:
    def __init__(self,ini = 1, attack_rate = 0.5, death_rate = 0.7, efficiency_rate = 1.6):
        # Parameters of the predator
        self.ini = ini
        self.attack_rate = attack_rate
        self.death_rate = death_rate
        self.efficiency_rate = efficiency_rate
        
class prey:
    def __init__(self,ini = 1, growth_rate = 1.3):
        # Parameters of the prey
        self.ini = ini
        self.growth_rate = growth_rate
        
class environment:
    def __init__(self, carrying_capacity=5):
        # Parameters of the environment
        self.carrying_capacity = carrying_capacity

def predator_equation(prey_pop, predator_pop, death_rate, efficiency_rate, period): # equation for determining the differential of predator populations 
    d_predator = (-death_rate + efficiency_rate * prey_pop) * predator_pop * period
    return d_predator
        
def prey_equation(prey_pop, predator_pop, carrying_capacity, growth_rate, attack_rate, period): # equation for determining the differential of prey population
    d_prey = (growth_rate * (1 - prey_pop / carrying_capacity) - attack_rate * predator_pop) * prey_pop * period
    return d_prey

def simulation(T,predator,prey,environment):
    prey_pop = np.zeros(T) # array for storing the current prey amounts 
    predator_pop = np.zeros(T) # array for storing the current predator amounts 
    prey_pop[0] = prey.ini
    predator_pop[0] = predator.ini
    
    period = 1/7
    
    # For loop which adds the differentials to the prey and predator populations for a given amount of time
    for t in np.arange(1,T):
        d_prey = prey_equation(prey_pop[t-1],predator_pop[t-1], environment.carrying_capacity, prey.growth_rate, predator.attack_rate, period) # calculates prey differential
        d_predator = predator_equation(prey_pop[t-1], predator_pop[t-1], predator.death_rate,predator.efficiency_rate, period) # calculates predator differential
        prey_pop[t] = np.max([prey_pop[t-1] + d_prey,0])# stores the new values in the arrays for plotting
        predator_pop[t] = np.max([predator_pop[t-1] + d_predator,0])
        
    return predator_pop,prey_pop

def model(param,T):
    # predetor
    predator_ini      = param[0]
    attack_rate       = param[1]
    efficiency_rate   = param[2]
    death_rate        = param[3]

    # prey
    prey_ini          = param[4]
    growth_rate       = 1.5
    # environment
    carrying_capacity = 20
    
    prey_pop = np.zeros(T) # array for storing the current prey amounts 
    predator_pop = np.zeros(T) # array for storing the current predator amounts 
    prey_pop[0] = prey_ini
    predator_pop[0] = predator_ini
    
    period = 1/7
    
    # For loop which adds the differentials to the prey and predator populations for a given amount of time
    for t in np.arange(1,T):
        d_prey = (growth_rate * (1 - prey_pop[t-1] / carrying_capacity) - attack_rate * predator_pop[t-1]) * prey_pop[t-1] * period
        d_predator = (-death_rate + attack_rate * efficiency_rate * prey_pop[t-1]) * predator_pop[t-1] * period
        prey_pop[t] = np.array([prey_pop[t-1] + d_prey,0]).max()# stores the new values in the arrays for plotting
        predator_pop[t] = np.array([predator_pop[t-1] + d_predator,0]).max()
#        if prey_pop[t] < 2 or predator_pop[t] < 2 or prey_pop[t] > carrying_capacity or predator_pop[t] > carrying_capacity:
#            break
        
    return predator_pop,prey_pop


@njit(parallel = False) # Numba decorator to speed-up the function below
def function(param,T,equil_value):
    # predetor
    predator_ini      = param[0]
    attack_rate       = param[1]
    efficiency_rate   = param[2]
    death_rate        = param[3]

    # prey
    prey_ini          = param[4]
    growth_rate       = 1.5
    # environment
    carrying_capacity = 20
    
    prey_pop = np.zeros(T) # array for storing the current prey amounts 
    predator_pop = np.zeros(T) # array for storing the current predator amounts 
    prey_pop[0] = prey_ini
    predator_pop[0] = predator_ini
    
    period = 1/7
    
    # For loop which adds the differentials to the prey and predator populations for a given amount of time
    for t in np.arange(1,T):
        d_prey = (growth_rate * (1 - prey_pop[t-1] / carrying_capacity) - attack_rate * predator_pop[t-1]) * prey_pop[t-1] * period
        d_predator = (-death_rate + attack_rate * efficiency_rate * prey_pop[t-1]) * predator_pop[t-1] * period
        prey_pop[t] = np.array([prey_pop[t-1] + d_prey,0]).max()# stores the new values in the arrays for plotting
        predator_pop[t] = np.array([predator_pop[t-1] + d_predator,0]).max()
#        if prey_pop[t] < 2 or predator_pop[t] < 2 or prey_pop[t] > carrying_capacity or predator_pop[t] > carrying_capacity:
#            break
#     equil_value = 7
    equil_dev = np.array(np.abs(predator_pop[-1]-equil_value)+np.abs(prey_pop[-1]-equil_value))
        
    return equil_dev