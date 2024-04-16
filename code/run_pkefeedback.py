# -*- coding: utf-8 -*-
"""run_pkefeedback

Executes a simple transient package that includes:
    - Solution of point kinetic equations
    - Lumped conduction-convection problem
    - Change in reactivity

Created on Tue Oct 12 22:30:00 2021 @author: Dan Kotlyar
Last updated on Wed Oct 13 10:45:00 2021 @author: Dan Kotlyar
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore")

from pkewithfeedback import SimpleTransient, plot1D


# Store kinetic parameters a designated dictionary
# -----------------------------------------------------------------------------
PKE = {}
beta1gr = 0.0075
PKE["beta"] = beta1gr * np.array([0.033, 0.219, 0.196, 0.395, 0.115, 0.042])
beta = PKE["beta"]
PKE["lamda"] = np.array([0.0124, 0.0305, 0.1110, 0.3011, 1.1400, 3.0100])
PKE["promptL"] = 0.001

# Store thermal properties in a designated dictionary
# -----------------------------------------------------------------------------
TH = {}
TH["C_F"] = 200.00     # J/kg/K
TH["C_M"] = 4000.0     # J/kg/K
TH["M_F"] = 40000.     # kg
TH["M_M"] = 7000.0     # kg
TH["W_M"] = 8000.0     # kg/sec
TH["Tin"] = 550.00     # K
TH["h"] = 4E+6         # J/K/sec

# Define metadata parameters and external reactivity scenario
# -----------------------------------------------------------------------------
P0 = 1500E+6     # Nominal power in Watts
P1 = 3000E+6     # Desired power uprate
aF = -1E-05      # drho/dTf
aM = -10E-05      # drho/dTm (associated with density change)

simulationTime = 120  # seconds
nsteps = 51         # number of time-steps
timepoints = np.linspace(0, simulationTime, nsteps)  # absolute time vector
rhoExtStep = 0.5 * beta1gr * np.ones(nsteps)             # 0.1*Beta

#---------------------------------
# Defining the Objective Function
#---------------------------------
constraint_violation_penalty = 1E+06

def find_final_time(P, P1):
    """Finds the time at which the power is nearly constant (steady state) and equal to
    the desired power P1, if not found, return """
    
    power_uprate_acheived = np.isclose(P, P1, rtol=1e-05, equal_nan=False)
    steady_state_solution = np.isclose(0, np.gradient(P, timepoints), rtol=1e-05, equal_nan=False)
    if np.any(np.logical_and(power_uprate_acheived, steady_state_solution)):
        # Get first index where condition was met (the following points are not relevant)
        first_index = np.argmax(np.logical_and(power_uprate_acheived, steady_state_solution))
        return  (first_index, P[first_index])
    else: # Condition was not met, return a distinctive output
        return (None, None)


def objective_function(rhoExt, P1):
    X, T, rho, rhoF = SimpleTransient(timepoints, P0, rhoExt, PKE, TH, aF, 0.1E-05)
    final_time_index, final_time = find_final_time(X[0, :], P1)

    # Set initial penalty
    penalty = 0
    if final_time_index is None and final_time is None:
        # End point conditions not satisfied, return large value
        return constraint_violation_penalty
    else:
        power_overshoot = np.max(X[0, 0:(final_time_index + 1)]) - P1
        reactivity_overshoot = np.max(rhoExt[0:(final_time_index + 1)]) - beta
        if power_overshoot > 0:
            penalty += constraint_violation_penalty
        if reactivity_overshoot > 0:
            penalty += constraint_violation_penalty
        
        return penalty + final_time


#-------------------
# Minimization Loop
#-------------------

guess = rhoExtStep
optimal_control = minimize(
    objective_function,
    guess,
    options={'verbose': 3, 'gtol': 1e-6, 'xtol': 1e-6, 'maxiter': 1000},
    args=P1
)