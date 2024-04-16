# -*- coding: utf-8 -*-
"""pkewithfeedback

Package that solves point kinetic equations coupled with a lumped T/H model
and linked via an algebraic reactivity relation.

Created on Tue Oct 12 22:30:00 2021 @author: Dan Kotlyar
Last updated on Wed Oct 13 10:45:00 2021 @author: Dan Kotlyar
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm, inv

# Default values
FONT_SIZE = 16  # font size for plotting purposes


def SimpleTransient(timepoints, P0, rhoExt, PKE, TH, aF, aM):
    """Main function to solve a full transient scenario

    Parameters
    ----------
    timepoints : array
        Time points in seconds
    P0 : float
        Nominal power in Watts
    PKE : dict
        Contains all the PKEs parameters (e.g. beta)
    TH : dict
        Contains all the thermal parameters (e.g. heat capacity)
    aF : float
        Fuel temperature reactivity coefficient
    aM : float
        Moderator temperature reactivity coefficient

    Returns
    -------
    X : 2-dim ndarray
        Power and precursors as a function of time
    T : 2-dim ndarray
        Fuel and moderator vector as a function of time
    rho : 1-dim ndarray
        Reactivity as a function of time
    rhoF : 1-dim ndarray
        Feedback reactivity as a function of time

    """

    # Define all the time-steps used to progress time
    timesteps = timepoints[1::] - timepoints[:-1:]
    nsteps = len(timesteps)
    groups = len(PKE["beta"])

    # reset the expected results matrices
    X = np.zeros((groups+1, nsteps + 1))  # [P, C1, ..., C6]
    T = np.zeros((2, nsteps + 1))  # [Tf, Tm]
    rho = np.zeros(nsteps + 1)  # time-dependent reactivity
    drhoF = np.zeros(nsteps + 1)  # time-dependent partial reactivity feedback
    rhoF = np.zeros(nsteps + 1)  # time-dependent feedback reactivity

    # Obtain the initial values at steady-state
    X[:, 0], T[:, 0] = _InitialConditions(P0, PKE, TH)

    # delta external reactivity added at each new time-step
    drhoExt = rhoExt[1::] - rhoExt[:-1:]  # reactivity difference between steps
    rho[0] = rhoExt[0]  # initial reactivity

    for idx, dt in enumerate(timesteps, start=1):

        # Total reactivity of the system
        rho[idx] = rho[idx-1] + drhoExt[idx-1] + drhoF[idx-1]

        # solve PKE
        X[:, idx] = solvePKEs(dt, rho[idx], X[:, idx-1],
                              PKE["beta"], PKE["lamda"], PKE["promptL"])

        # solve lumped conduction-convection
        T[:, idx] = solveTH(T[:, idx-1], X[0, idx], dt, TH)

        # update the partial reactivity based on the variation in temperatures
        drhoF[idx] = feedbackReactivity(
            T[0, idx-1], T[0, idx], T[1, idx-1], T[1, idx], aF, aM)
        rhoF[idx] = rhoF[idx-1] + drhoF[idx] # Only for tracking the cumulative rho
        
        # Look at differential feedback

    return X, T, rho, rhoF


def solvePKEs(dt, rho, X0, beta, lamda, promptL):
    """Solves the point kinetic equations

    Parameters
    ----------
    dt : float
        Time step in seconds
    rho : float
        Reactivity
    beta : 1-dim ndarray
        delayed neutron fractions
    lamda : 1-dim ndarray
        delayed neutron decay constants
    prompt : float
        prompt neutron generation time in seconds
    X0 : 1-dim array
        beginning-of-step vector [P, C1, ..., C6]'

    Returns
    -------
    X1 : 1-dim ndarray
        end-of-step vector [P, C1, ..., C6]
    """

    groupsDN = len(beta)  # Number of delay neutron groups
    betaTot = beta.sum()  # Calculate beta total:

    # reset the PKE matrix
    mtxA = np.zeros((groupsDN+1, groupsDN+1))
    # reactivity at a specific time point

    # build diagonal
    np.fill_diagonal(mtxA, np.append(0, -lamda))
    # build the first row
    mtxA[0, :] = np.append((rho-betaTot)/promptL, lamda)
    # build the first column
    mtxA[1:groupsDN+1, 0] = beta / promptL

    # Obtain the solution after dt
    X1 = np.dot(expm(mtxA*dt), X0)

    return X1


def solveTH(T0, P, dt, TH):
    """Solves two conduction-convection equations

    Parameters
    ----------
    dt : float
        Time step in seconds
    P : float
        Power in Watts
    T0 : 1-dim array
        beginning-of-step vector [Tf, Tm]

    Returns
    -------
    T1 : 1-dim ndarray
        end-of-step vector [P, C1, ..., C6]'
    """

    # reset the T/H matrix
    mtx = np.zeros((2, 2))
    Ident = np.eye(2)

    # build the matrix manually
    b = TH['h']/(TH['M_F']*TH['C_F'])
    c = TH['h']/(TH['M_M']*TH['C_M'])
    d = 2*TH['W_M']/TH['M_M']
    mtx[0, 0] = b
    mtx[0, 1] = -b
    mtx[1, 0] = -c
    mtx[1, 1] = c+d

    # Define the non-homogeneous terms
    D = np.array([P/(TH["M_F"]*TH["C_F"]), TH["Tin"]*2*TH["W_M"]/TH["M_M"]])

    # Obtain the T/H solution after dt
    T1 = np.dot(expm(-mtx*dt), T0) +\
        np.dot(inv(mtx), (Ident - expm(-mtx*dt))).dot(D)

    return T1


def _InitialConditions(P0, PKE, TH):
    """Solves the point kinetic equations"""

    # Obtain the steady-state temperatures
    Tf0 = TH["Tin"] + (1/(2*TH['W_M']*TH['C_M']) + 1/TH["h"])*P0
    Tm0 = TH['Tin'] + P0/(2*TH['W_M']*TH['C_M'])
    T0 = np.array([Tf0, Tm0])

    # Obtain the steady-state precursors
    X0 = P0*np.append(1, PKE["beta"]/PKE["lamda"]/PKE["promptL"])

    return X0, T0


def feedbackReactivity(Tf0, Tf1, Tm0, Tm1, aF, aM):
    """Update the total reactivity following feedback inclusion

    Parameters
    ----------
    Tm0 : float
        Moderator temperature in Kelvin at previous step
    Tf0 : float
        Fuel temperature in Kelvin at previous step
    Tm1 : float
        Current moderator temperature in Kelvin
    Tf1 : float
        Current fuel temperature in Kelvin

    Returns
    -------
    rho1 : float
        Updated reactivity
    """

    drhoFuel = aF*(Tf1 - Tf0)
    drhoMod = aM*(Tm1 - Tm0)
    return drhoMod + drhoFuel


def plot1D(xvals, yvals, xlabel=None, ylabel=None, fontsize=FONT_SIZE,
           marker="--*", markerfill=False, markersize=6):
    """Plot the 1D slab neutron flux distribution.

    The function is meant to be executed after the one-group slab diffusion
    solver is ran.

    Parameters
    ----------
    power : bool, optional
        if ``power`` is `True` or not included the
        power distribution is plotted, otherwise not
    xvals : ndarray
        x-axis values
    yvals : ndarray
        y-axis values
    xlabel : str
        x-axis label with a default ``Length, meters``
    ylabel : str
        y-axis label with a default ``Normalized Flux``
    fontsize : float
        font size value
    markers : str or list of strings
        markers type
    markerfill : bool
        True if the marking filling to be included and False otherwise
    markersize : int or float
        size of the marker with a default of 8.

    """

    if xlabel is None:
        xlabel = "Slab length, meters"
    if ylabel is None:
        ylabel = "Normalized flux"

    if markerfill:
        mfc = "white"  # marker fill color
    else:
        mfc = None

    plt.ylabel(ylabel)
    plt.xlabel(xlabel)

    plt.plot(xvals, yvals, marker, mfc=mfc, ms=markersize)
    plt.grid()
    plt.rc('font', size=fontsize)      # text sizes
    plt.rc('axes', labelsize=fontsize)  # labels
    plt.rc('xtick', labelsize=fontsize)  # tick labels
    plt.rc('ytick', labelsize=fontsize)  # tick labels
