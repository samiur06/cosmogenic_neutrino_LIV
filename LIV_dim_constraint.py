import numpy as np
from sympy import sin, cos 
import sympy as sp
import scipy as scp
import pandas as pd
from cmath import *
import matplotlib.pyplot as plt
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
import ternary

t0 = timeit.default_timer()

def ini_composition_norm(x):
    arr = np.asarray(x, dtype=float)
    s = arr.sum()
    if s == 0:
        raise ValueError("Cannot normalize: sum of elements is zero.")
    return arr / s

### eq 6, 2503.15468
def H_LIV_paper(E,d,a_eff=None,c_eff=None):
    """
    Compute H_LIV_paper^(d) in flavor space.

    Parameters
    ----------
    E : float
        Neutrino energy
    theta, phi : float
        Direction of momentum (polar, azimuth)
    d : int
        Operator dimension
    a_eff, c_eff : dict
        LIV coefficients indexed as:
        coeff[(l, m)] -> 3x3 complex matrix in flavor space

    Returns
    -------
    H : (3,3) complex ndarray
        LIV Hamiltonian in flavor space
    """
    d = int(d)
    H = np.zeros((3, 3), dtype=complex)

    prefactor = E**(d - 3)
    
    if d % 2 == 1:  # CPT-odd
        if a_eff is None:
            a_eff = np.zeros((3, 3), dtype=complex)
        if np.shape(a_eff)==np.shape(H):
            H += prefactor * a_eff
        else:
            raise TypeError("shapes don't match.")


    else:  # CPT-even
        if c_eff is None:
            c_eff = np.zeros((3, 3), dtype=complex)
        if np.shape(c_eff)==np.shape(H):
            H -= prefactor * c_eff
        else:
            raise TypeError("shapes don't match.")

    return H + H.conj().T

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

def H0_flavor(E, dm21, dm31, theta12, theta23, theta13, delta):
    """Vacuum Hamiltonian in flavor basis."""
    U_pmns = pmns_matrix(theta12, theta23, theta13, delta)
    H0 = U_pmns @ np.diag([0, dm21/(2*E), dm31/(2*E)]) @ U_pmns.conj().T
    return H0 

def prob_avg(E, dm21, dm31, theta12, theta23, theta13, delta,
            d, a_eff=None, c_eff=None):

    H0 = H0_flavor(E, dm21, dm31, theta12, theta23, theta13, delta)
    HLIV = H_LIV_paper(E, d, a_eff=a_eff, c_eff=c_eff)
    Htot = H0 + HLIV
    rescaled_Htot = Htot
#     rescaled_Htot = Htot / np.max(np.abs(Htot)) 
    ## orders of magnitude mismatch of elements in H0 & HLIV
    ## can give error to eigenvector calculation
    ## hence rescaling the matrix, keeping the egvectors same
    _, evecs = np.linalg.eigh(rescaled_Htot)
    V = evecs

    avgP = np.zeros((3,3))
    for alpha in range(3):
        for beta in range(3):
            avgP[alpha, beta] = np.sum(np.abs(V[alpha, :])**2 * np.abs(V[beta, :])**2)

    return avgP 

def chisq(diffBSM, diff0):
    if diffBSM==0 and diff0==0:
        return 0 
    if diffBSM==0 and diff0!=0:
        raise Exception("diffBSM is zero, diff0 is not!!")
        
    frac = diff0/diffBSM
    if diff0==0:
        frac=1
        
    return 2*(diffBSM - diff0 + diff0 * np.log(frac))

neuflux = pd.read_csv("neuflux_1505.04020/neuflux_SFR_proton_only.csv", 
                      header=None, names=["log10Enu", "rate"]) 

E_eV = 10**neuflux["log10Enu"].values
rate = neuflux["rate"].values   # already GeV/(cm^2 s sr)")

# Build interpolation function: input E (eV) 
flux_interp = interp1d(
    E_eV,
    rate,
    kind="linear",       # or "cubic" for smoother
    bounds_error=False,
    fill_value="extrapolate"
)

poemma_area = pd.read_csv("eff_area/poemma_nutau_tau.csv", 
                      header=None, names=["log10Enu", "area"]) 

kmtocm = 1e5
E_eV = 10**poemma_area["log10Enu"].values
eff_area = poemma_area["area"].values * kmtocm**2   # cm^2 sr

poemma_area_interp = interp1d(
    E_eV,
    eff_area,
    kind="linear",       # or "cubic" for smoother
    bounds_error=False,
    fill_value="extrapolate"
)

# ---- Test the interpolator ----
E_test = 1e17  # eV
print(f"area at E = {E_test:.1e} eV : {poemma_area_interp(E_test):.3e} cm^2 sr")

def flavor_fraction(E, dm21, dm31, 
                  theta12, theta23, theta13, delta,
                  d=0, a_eff=None, c_eff=None,
                  src_ratio=[1,2,0], flavor='tau'):
    
    flavor_dict = {'e':0, 'mu':1, 'tau':2}

    prob_avg_matrix = prob_avg(E, dm21, dm31, 
                theta12, theta23, theta13, delta,
                d, a_eff, c_eff)

    comp = (prob_avg_matrix 
            @ ini_composition_norm(src_ratio))[flavor_dict[flavor]]

    return comp 

# Integration limits in eV
E_min = 1e15      # 1 PeV = 1e15 eV
E_max = 100e18      # 100 EeV = 1e20 eV
year = 365*24*3600

def tau_count(d=0, a_eff=None, c_eff=None, src_ratio=[1,2,0]):
    # Define the integrand_flux_area
    def integrand_flux_area(E, d, a_eff, c_eff, src_ratio):
        EnuGeV = E * 1e-9
        # flux = rate/E^2 = 1/(GeV cm^2 s sr)
        # area = cm^2 sr
        # flux * area = GeV^-1 s^-1
        unit_refiner = (1/EnuGeV**2 ) * 1e-9 
        #factor explanaton: 1/E^2, 1/GeV to 1/eV

        dm21, dm31 = np.array([7.49e-5, 2.534e-3])*1e-18
        theta12, theta13, theta23 =  [np.arcsin(np.sqrt(0.303)), 
                                      np.arcsin(np.sqrt(0.02225)), 
                                      np.arcsin(np.sqrt(0.572))]

        delta = 197*np.pi/180

        ftau_fraction = flavor_fraction(E, dm21, dm31, 
                  theta12, theta23, theta13, delta,
                  d, a_eff, c_eff,
                  src_ratio, flavor='tau')
        
        return flux_interp(E) * ftau_fraction * poemma_area_interp(E) * unit_refiner    # 1/(s)

    # Perform the integral
    result, err = quad(integrand_flux_area, E_min, E_max, 
                       args=(d, a_eff, c_eff, src_ratio), limit=500, epsrel=1e-4)#, epsabs=0, epsrel=1e-4)

    # tobs = 0.2*5 years, pg. 16 , https://arxiv.org/pdf/1902.11287
    # full 360 degree i.e. 2pi azimuth # hence no factor 30/360
    # 0.2 for duty cycle, 1/3 for tau flavor
    return result * 5*year*0.20


### compute tau count
d=4

"""
guesses = np.logspace(np.log10(1e-75), np.log10(1e-30), 45)
tau_count_track = {}

for src_ratio in [[1,0,0],[0,1,0],[1,2,0]]:
    srctext = ''.join(f"{x}" for x in src_ratio)
    tau_count_track[srctext] = {}

    for alpha in range(3):
        for beta in range(alpha,3):
            
            tau_vals = []
            tau_vals_std = []
            for sol in guesses:
                LIVmatrix = np.zeros((3, 3), dtype=complex)
                LIVmatrix[alpha, beta] = sol  # GeV^{-n}
                tau_vals.append([sol,
                                 tau_count(d=d,
                                           a_eff=LIVmatrix, 
                                           c_eff=LIVmatrix,
                                           src_ratio=src_ratio)
                                ])
                tau_vals_std.append([sol,tau_count(src_ratio=src_ratio)])

            tau_vals = np.array(tau_vals)
            tau_vals_std = np.array(tau_vals_std)
            tau_count_track[srctext][alpha,beta] = {'LIV':tau_vals,
                                    'std':tau_vals_std}
"""

d=4
guesses = np.logspace(np.log10(1e-62), np.log10(1e-57), 20)
tau_count_track = {}

for src_ratio in [[1,0,0],[0,1,0],[1,2,0]]:
    srctext = ''.join(f"{x}" for x in src_ratio)
    tau_count_track[srctext] = {}

    for alpha in range(3):
        for beta in range(alpha,3):
            
            tau_vals = []
            tau_vals_std = []
            for sol in guesses:
                LIVmatrix = np.zeros((3, 3), dtype=complex)
                LIVmatrix[alpha, beta] = sol  # GeV^{-n}
                tau_vals.append([sol,
                                 tau_count(d=d,
                                           a_eff=LIVmatrix, 
                                           c_eff=LIVmatrix,
                                           src_ratio=src_ratio)
                                ])
                tau_vals_std.append([sol,tau_count(src_ratio=src_ratio)])

            tau_vals = np.array(tau_vals)
            tau_vals_std = np.array(tau_vals_std)
            tau_count_track[srctext][alpha,beta] = {'LIV':tau_vals,
                                    'std':tau_vals_std}


### solve chi2 eq. from tau count and save it to LIVparam dictionary
LIVparam = {}
LIVparam[d]={}

for src_ratio in [[1,0,0],[0,1,0],[1,2,0]]:
    srctext = ''.join(f"{x}" for x in src_ratio)
    print(srctext)
    LIVparam[d][srctext] = {}
    for (alpha, beta), data in tau_count_track[srctext].items():
        print(f"{alpha, beta}")
        tau_vals = data['LIV']   # shape (N,2)
        c_vals = tau_vals[:,0]
        NBSM_vals = tau_vals[:,1]
        N0 = tau_count(src_ratio=src_ratio)
        target = chi2.ppf(0.95, 1)

        # compute diff array
        diff_vals = np.abs(NBSM_vals - N0)

        # mask out zero-diff points
        mask = diff_vals != 0

        c_vals_f = c_vals[mask]
        diff_vals_f = diff_vals[mask]

        # build interpolator on filtered data
        diff_of_c = interp1d(c_vals_f, diff_vals_f, 
                             kind='cubic', fill_value="extrapolate")
        
#         print(chisq(diff_of_c(2.3256822905487024e-57), 0))
        def makezero(c):
            return chisq(diff_of_c(c), 0) - target

#         ### vanilla, less precise 
#         chi_vals = np.array([chisq(x, 0) for x in diff_vals_f])
#         idx = np.argmin(np.abs(chi_vals - target))
#         c_solution = c_vals_f[idx]
#         print(f"{alpha, beta}, c =", c_solution,
#               np.round(makezero(c_solution),3))

#         ### brentq, needs +,- sign change with c_min, c_max
#         c_min = c_vals_f.min()
#         c_max = c_vals_f.max()
#         c_solution = brentq(makezero, c_min, c_max)
        
        ### fsolve, needs good guess 
        c_solution = 0
        for c_guess in [1e-57, 1e-58, 1e-59, 1e-60, 1e-61, 1e-62]:
            csol = fsolve(makezero, c_guess)[0]
#             print(makezero(c_solution), makezero(csol))
            if np.abs(makezero(csol))<0.001 and csol>0:
                c_solution = csol
#             print(makezero(csol),csol,c_solution)

        print(f"{alpha, beta}, c =", c_solution,
          np.round(makezero(c_solution),4))

        LIVparam[d][srctext][f"{alpha,beta}"] = c_solution

#         print("NBSM =", NBSM_of_c(c_solution))



### make plot
d = 4
src_ratio_all = [[1,0,0],[0,1,0],[1,2,0]]
srctexts = ["100", "010", "120"]
colors = ["tab:blue", "tab:orange", "tab:green"]

# get alpha,beta keys from one entry
alphabeta_keys = list(LIVparam[d][srctexts[0]].keys())  # 6 of them

# prepare data matrix: rows = alpha,beta, cols = srctext
vals = np.zeros((len(alphabeta_keys), len(srctexts)))

for i, ab in enumerate(alphabeta_keys):
    for j, sr in enumerate(srctexts):
        vals[i, j] = LIVparam[d][sr][ab]

# bar positions
x = np.arange(len(alphabeta_keys))
width = 0.25

plt.figure(figsize=(8,5))

flavlabel = [r'$e$',r'$\mu$',r'$\tau$']
y_top = 1e-1

for j in range(len(srctexts)):
    print(len(np.log10(vals[:, j])), np.shape(np.log10(vals[:, j])))
    plt.bar(x + j*width, np.log10(vals[:, j]) - np.log10(y_top),
            color=colors[j], width=width)

# x-ticks: alpha,beta labels
plt.xticks(x + width, 
          [f"{flavlabel[k[0]]}{flavlabel[k[1]]}"
           for k in (ast.literal_eval(x) for x in alphabeta_keys)],
          fontsize=14)

# plt.yscale("log")
plt.ylim(np.log10(1e-40), -70)
# plt.ylim(np.log10(y_top), -90)


plt.gca().invert_yaxis()

plt.ylabel(r"$\log{c_{\alpha\beta}/\,[\mathrm{GeV}^{-1}]}$", fontsize=14)
plt.xlabel(r"$(\alpha,\beta)$", fontsize=14)
plt.title(rf"$d={d}$", fontsize=14)

# legend (srctext)
src_legend = [fr"$(\nu_e,\nu_\mu,\nu_\tau)_S={tuple(src_ratio_all[0])}$", 
              fr"$(\nu_e,\nu_\mu,\nu_\tau)_S={tuple(src_ratio_all[1])}$", 
              fr"$(\nu_e,\nu_\mu,\nu_\tau)_S={tuple(src_ratio_all[2])}$"]

plt.legend(src_legend, frameon=False, fontsize=14)

plt.tick_params(top=False, bottom=False, right=True,
                direction="in", which="both")
plt.minorticks_on()

plt.tight_layout()
# plt.savefig("figures/param_dim4.pdf", dpi=300)
plt.show()
