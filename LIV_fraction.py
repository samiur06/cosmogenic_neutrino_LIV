import numpy as np
from sympy import sin, cos 
import sympy as sp
import scipy as scp
import pandas as pd
from cmath import *
import matplotlib.pyplot as plt
from scipy.linalg import eigh as scipy_eigh
from scipy.integrate import odeint
from scipy.optimize import fsolve
import scienceplots
plt.style.context(['science'])
plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "Times New Roman"
import pickle
import timeit
import glob # need to merge arrays
import os # need to make directory
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import chi2
import ast

t0 = timeit.default_timer()

def ini_composition_norm(x):
    arr = np.asarray(x, dtype=float)
    s = arr.sum()
    if s == 0:
        raise ValueError("Cannot normalize: sum of elements is zero.")
    return arr / s

dm21, dm31 = np.array([7.49e-5, 2.534e-3])
theta12, theta13, theta23 =  [np.arcsin(np.sqrt(0.303)), 
                              np.arcsin(np.sqrt(0.02225)), 
                              np.arcsin(np.sqrt(0.572))]

delta = 197*np.pi/180

def pmns_matrix(theta12, theta23, theta13, delta):

    s12, s13, s23 = np.sin(theta12), np.sin(theta13), np.sin(theta23)
    c12, c13, c23 = np.cos(theta12), np.cos(theta13), np.cos(theta23)

    return np.array([
        [c12*c13, s12*c13, s13*np.exp(-1j*delta)],
        [-s12*c23 - c12*s23*s13*np.exp(1j*delta),
         c12*c23 - s12*s23*s13*np.exp(1j*delta),
         s23*c13],
        [s12*s23 - c12*c23*s13*np.exp(1j*delta),
         -c12*s23 - s12*c23*s13*np.exp(1j*delta),
         c23*c13]
    ], dtype=complex)

std_pmns = pmns_matrix(theta12, theta23, theta13, delta)

def H0_flavor(E, dm21, dm31, theta12, theta23, theta13, delta):
    """
    Vacuum Hamiltonian in flavor basis.
    Parameters
    ----------
    E : float or (N,) array
        Neutrino energy in eV
    Returns
    -------
    H0 : (3,3) or (N,3,3) complex ndarray
    """
    E = np.asarray(E, dtype=float)
    scalar_input = E.ndim == 0
    E = np.atleast_1d(E)                   # (N,)
    N = len(E)

    U = pmns_matrix(theta12, theta23, theta13, delta)  # (3,3)

    # mass eigenvalues: shape (N,3)
    mass_diag = np.stack([
        np.zeros(N),
        dm21 / (2.0 * E),
        dm31 / (2.0 * E)
    ], axis=1)                             # (N,3)

    # build diagonal matrices: (N,3,3)
    D = np.zeros((N, 3, 3), dtype=complex)
    idx = np.arange(3)
    D[:, idx, idx] = mass_diag            # fill diagonals
    # U @ D @ U†  batched: (3,3) @ (N,3,3) @ (3,3)
    Uh = U.conj().T                        # (3,3)
    H0 = U @ D @ Uh                        # broadcasts correctly → (N,3,3)

    if scalar_input:
        return H0[0]                       # (3,3)
    return H0                              # (N,3,3)


### eq 6, 2503.15468
def H_LIV_paper(E, d, a_eff=None, c_eff=None):
    """
    Compute H_LIV in flavor space.
    Parameters
    ----------
    E : float or (N,) array
        Neutrino energy in eV
    d : int
        Operator dimension
    a_eff, c_eff : (3,3) complex ndarray
        LIV coefficients in flavor space
    Returns
    -------
    H : (3,3) or (N,3,3) complex ndarray
    """
    d = int(d)
    E = np.asarray(E, dtype=float)
    scalar_input = E.ndim == 0
    E = np.atleast_1d(E)                   # (N,)
    N = len(E)

    E_GeV    = E * 1e-9                    # (N,)
    kaffara  = 1e9
    prefactor = E_GeV ** (d - 3) * kaffara  # (N,)
    pre      = prefactor[:, None, None]    # (N,1,1) for broadcasting with (3,3)

    H = np.zeros((N, 3, 3), dtype=complex)

    if d % 2 == 1:  # CPT-odd
        if a_eff is None:
            a_eff = np.zeros((3, 3), dtype=complex)
        if a_eff.shape != (3, 3):
            raise TypeError("a_eff shape must be (3,3).")
        H += pre * a_eff                   # (N,1,1) * (3,3) → (N,3,3)
    else:  # CPT-even
        if c_eff is None:
            c_eff = np.zeros((3, 3), dtype=complex)
        if c_eff.shape != (3, 3):
            raise TypeError("c_eff shape must be (3,3).")
        H -= pre * c_eff                   # (N,1,1) * (3,3) → (N,3,3)

    # H + H† (Hermitian part), axes=(0,2,1) transposes only the 3x3 part
    H = H + np.conj(H.swapaxes(1, 2))

    if scalar_input:
        return H[0]                        # (3,3)
    return H                               # (N,3,3)


# def prob_avg(E, dm21, dm31, theta12, theta23, theta13, delta,
#             d, a_eff=None, c_eff=None):

#     H0 = H0_flavor(E, dm21, dm31, theta12, theta23, theta13, delta)
#     HLIV = H_LIV_paper(E, d, a_eff=a_eff, c_eff=c_eff)
#     Htot = H0 + HLIV
#     _, evecs = np.linalg.eigh(Htot)
#     V = evecs

#     avgP = np.zeros((3,3))
#     for alpha in range(3):
#         for beta in range(3):
#             avgP[alpha, beta] = np.sum(np.abs(V[alpha, :])**2 * np.abs(V[beta, :])**2)

#     return avgP 

def get_pmns(E, d, a_eff=None, c_eff=None):
    E    = np.asarray(E, dtype=float)
    scalar_input = E.ndim == 0
    E    = np.atleast_1d(E)
    H0   = H0_flavor(E, dm21, dm31, theta12, theta23, theta13, delta)
    HLIV = H_LIV_paper(E, d, a_eff=a_eff, c_eff=c_eff)
    Htot = H0 + HLIV

    # scipy_eigh doesn't support batched (N,3,3) — loop over N
    N = Htot.shape[0]
    evecs = np.empty((N, 3, 3), dtype=complex)
    for i in range(N):
        _, evecs[i] = scipy_eigh(Htot[i])

    if scalar_input:
        return evecs[0]
    return evecs

def prob_avg(E, d, a_eff=None, c_eff=None):
    """
    Returns avgP of shape (3,3) for scalar E, or (N,3,3) for array E.
    avgP[alpha, beta] = sum_i |V[alpha,i]|^2 |V[beta,i]|^2
    """
    E = np.asarray(E, dtype=float)
    scalar_input = E.ndim == 0
    E = np.atleast_1d(E)

    V = get_pmns(E, d, a_eff, c_eff)                        # (N,3,3)

    # |V|^2: (N,3,3)
    absV2 = np.abs(V)**2                                 # (N,3,3)

    # avgP[n,alpha,beta] = sum_i absV2[n,alpha,i] * absV2[n,beta,i]
    # → einsum over the mass index i
    avgP = np.einsum('nai,nbi->nab', absV2, absV2)       # (N,3,3)

    if scalar_input:
        return avgP[0]                                   # (3,3)
    return avgP                                          # (N,3,3)


def prob_avg_redshifted(E, z, d, a_eff=None, c_eff=None):
    """
    Returns avgP of shape (3,3) for scalar E/z, or (N,3,3) for arrays.
    avgP[alpha, beta] = sum_i |Vprod[alpha,i]|^2 * |Vdet[beta,i]|^2
    """
    E = np.asarray(E, dtype=float)
    z = np.asarray(z, dtype=float)
    scalar_input = E.ndim == 0
    E = np.atleast_1d(E)
    z = np.atleast_1d(z)

    Vprod = get_pmns(E * (1 + z), d, a_eff, c_eff)          # (N,3,3)
    Vdet  = get_pmns(E, d, a_eff, c_eff)                    # (N,3,3)

    absProd2 = np.abs(Vprod)**2                          # (N,3,3)
    absDet2  = np.abs(Vdet)**2                           # (N,3,3)

    # avgP[n,alpha,beta] = sum_i absProd2[n,alpha,i] * absDet2[n,beta,i]
    avgP = np.einsum('nai,nbi->nab', absProd2, absDet2)  # (N,3,3)

    if scalar_input:
        return avgP[0]                                   # (3,3)
    return avgP                                          # (N,3,3)


def flavor_fraction(E, dm21, dm31, 
                  theta12, theta23, theta13, delta,
                  d=0, a_eff=None, c_eff=None,
                  src_ratio=[1,2,0], flavor='tau'):
    
    flavor_dict = {'e':0, 'mu':1, 'tau':2}

    prob_avg_matrix = prob_avg(E, d, a_eff, c_eff)

    comp = (prob_avg_matrix 
            @ ini_composition_norm(src_ratio))[flavor_dict[flavor]]

    return comp 

def flavor_fraction_redshifted(E, z, dm21, dm31, 
                  theta12, theta23, theta13, delta,
                  d=0, a_eff=None, c_eff=None,
                  src_ratio=[1,2,0], flavor='tau'):
    
    flavor_dict = {'e':0, 'mu':1, 'tau':2}

    prob_avg_matrix = prob_avg_redshifted(E, z, d, a_eff, c_eff)

    comp = (prob_avg_matrix 
            @ ini_composition_norm(src_ratio))[flavor_dict[flavor]]

    return comp 


###############################################################    

