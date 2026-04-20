import numpy as np
from sympy import sin, cos 
import sympy as sp
import scipy as scp
import pandas as pd
from cmath import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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
from scipy.interpolate import make_interp_spline
from itertools import combinations
from itertools import product

import ast
import ternary
import awkward as ak

import uproot 

from LIV_fraction import *
from area_info import *

t0 = timeit.default_timer()

### common functions

# ══════════════════════════════════════════════════════════════════
# 1.  CONSTANTS & COSMOLOGY  (Planck 2013, used in SimProp)
# ══════════════════════════════════════════════════════════════════
H0       = 67.3                     # km/s/Mpc
Omega_m  = 0.315
Omega_L  = 1.0 - Omega_m

Mpc_cm   = 3.085677581e24           # 1 Mpc  → cm
yr_s     = 3.1557600e7              # 1 yr   → s
c_cm_s   = 2.99792458e10            # c      in cm/s
H0_s     = H0 * 1e5 / Mpc_cm       # H0     in s⁻¹

eV_erg   = 1.60217657e-12           # 1 eV   → erg

def H_over_H0(z):
    """Dimensionless Hubble factor E(z) = H(z)/H0."""
    return np.sqrt(Omega_m * (1 + z)**3 + Omega_L)

def dtdz(z):
    """
    |dt/dz| in seconds  — the SimProp weighting kernel.
    From eq. (B.2): dt/dz = 1 / [H0 (1+z) E(z)]
    """
    return 1.0 / (H0_s * (1 + z) * H_over_H0(z))

# ══════════════════════════════════════════════════════════════════
# 2.  EVOLUTION  S(z) 
# ══════════════════════════════════════════════════════════════════

def S_noevolution(z):
    z   = np.atleast_1d(np.asarray(z, dtype=float))
    out = np.ones(np.shape(z))
    
    return out

def S_AGN(z):
    z   = np.atleast_1d(np.asarray(z, dtype=float))
    out = np.zeros_like(z)

    m1 = (z >= 0.0) & (z <= 1.7)
    m2 = (z > 1.7)  & (z <= 2.7)
    m3 = (z > 2.7)  & (z <= 6.0)

    out[m1] = (1 + z[m1])**5
    out[m2] = 2.7**5
    out[m3] = 2.7**5 * 10**(2.7 - z[m3])

    return out

def S_SFR(z):
    z   = np.atleast_1d(np.asarray(z, dtype=float))
    out = np.zeros_like(z)
    m1 = z <  1.0
    m2 = (z >= 1.0) & (z < 4.0)
    m3 = z >= 4.0
    out[m1] = (1 + z[m1])**3.4
    out[m2] = 2**3.7 * (1 + z[m2])**(-0.3)
    out[m3] = 2**3.7 * 5**3.2 * (1 + z[m3])**(-3.5)
    return out


### SimProp fluxes 

# ===============================
# CONSTANTS
# ===============================
eV_erg = 1.60218e-12
Mpc_cm = 3.0857e24
yr_s   = 3.154e7
mp_eV  = 0.938e9
Gamma0 = 1e8
R_cut_EV = 1e4

# ══════════════════════════════════════════════════════════════════
# SCENARIOS: (spectral_idx, ...) — auto-selects spectral_idx
# ══════════════════════════════════════════════════════════════════
SCENARIOS = {
    'no': ('2.60', 2.6, 1.5e46),
    'SFR':          ('2.50', 2.5, 4.5e45),
    'AGN':          ('2.40', 2.4, 3.5e45),
}

_EVOL = {
    'no'  : S_noevolution,
    'AGN' : S_AGN,
    'SFR' : S_SFR,
}


### tau count from flux in one run
def compute_tau_counts(
    cosmo_evolution, SCENARIOS, _EVOL,
    c_cm_s,
    d, a_eff, c_eff,
    exp,
    det_flav=2,                         # 0=e, 1=mu, 2=tau (user-specified)
    data_dir="SimProp-v2r4/src/data_proton",
    npz_dir="data/flux_array",
    n_bins=24, log_E_min=9.0, log_E_max=21.0,
):
    """
    tau counts with LIV with flavor transition 
    from neutrino production redshift to neutrino detection redshift (i.e. z=0)
    """
    _, spectral_idx_val, _ = SCENARIOS[cosmo_evolution]
    g = spectral_idx_val

    # ── Load MC arrays ────────────────────────────────────────────────────────
    data_tag = os.path.basename(data_dir.rstrip("/"))
    arr           = np.load(os.path.join(npz_dir, f"{data_tag}_{cosmo_evolution}_per_neu.npz"))
    inj_z_per_neu = arr["inj_z_per_neu"]
    inj_E_per_neu = arr["inj_E_per_neu"]
    flav_per_neu  = arr["flav_per_neu"]
    cosmo_weight  = arr["cosmo_weight"]
    neu_E_flat    = arr["neu_E_flat"]
    zOri_per_neu  = arr["zOri_per_neu"]      
    N_neu         = len(inj_z_per_neu)

    # ── N_protons and injection energy bounds (for Q0 norm and C) ─────────────
    file_list = [
        f"{data_dir}_{cosmo_evolution}/{f}:summary;1"
        for f in sorted(os.listdir(f"{data_dir}_{cosmo_evolution}"))
        if f.endswith(".root")
    ]
    injE_arr     = ak.to_numpy(uproot.concatenate(file_list, ["injEnergy"], library="ak")["injEnergy"])
    N_protons    = len(injE_arr)
    E_min, E_max = injE_arr.min(), injE_arr.max()

    # ── Spectral reweighting ──────────────────────────────────────────────────
    C = (1.0 / np.log(E_max / E_min)) if np.isclose(g, 1.0) \
        else (1.0 - g) / (E_max**(1.0 - g) - E_min**(1.0 - g))

    # ── Q0 spectrum ───────────────────────────────────────────────────────────
    def Q_shape(E, gamma_inj):
        Gamma = E / mp_eV
        return np.where(Gamma < Gamma0, Gamma**(-2.0), Gamma**(-gamma_inj)) \
               * np.exp(-E / (R_cut_EV * 1e18))

    def make_Q0(scenario):
        _, gamma_inj, L0_cgs = SCENARIOS[scenario]
        L0_si  = L0_cgs / Mpc_cm**3 / yr_s
        I, _   = quad(
            lambda lE: np.log(10) * (10**lE)**2 * Q_shape(10**lE, gamma_inj),
            np.log10(E_min), np.log10(E_max), limit=500, epsrel=1e-12,
        )
        A_norm = L0_si / (I * eV_erg)
        return lambda E_eV: A_norm * Q_shape(E_eV, gamma_inj)

    Q0 = make_Q0(cosmo_evolution)

    # ── Source flavour index per neutrino event (0=e, 1=mu, 2=tau) ───────────
    flavinc = np.abs(flav_per_neu).astype(int) - 1   # MC convention: 1/2/3 → 0/1/2

    # ── Oscillation probability: P(nu_{flavinc[i]} → nu_{det_flav}) ──────────
    # neu_E_flat: MC neutrino energies — correct input to prob
    prob_matrix  = prob_avg_redshifted(neu_E_flat, zOri_per_neu, d, a_eff, c_eff)  # (N,3,3)
    prob_per_neu = prob_matrix[np.arange(N_neu), flavinc, det_flav]                 # (N,)
    # ── Per-event weights ─────────────────────────────────────────────────────
    w = (
        (c_cm_s / (4.0 * np.pi))
        * (1.0 / N_protons)
        * _EVOL[cosmo_evolution](inj_z_per_neu)
        * cosmo_weight
        * 10.0                              # Delta_z
        * Q0(inj_E_per_neu)
        / (C * inj_E_per_neu**(-g))
        * prob_per_neu  # flavor transition with neutrino production redshift
    )                                       # (N,)

    # ── Bin into flux J [cm⁻² s⁻¹ sr⁻¹ eV⁻¹] ────────────────────────────────
    log_edges = np.linspace(log_E_min, log_E_max, n_bins + 1)
    E_cents   = 10**(0.5 * (log_edges[:-1] + log_edges[1:]))
    dE_eV     = E_cents * np.log(10) * (log_edges[1] - log_edges[0])
    bin_idx   = np.digitize(np.log10(neu_E_flat), log_edges) - 1
    valid     = (bin_idx >= 0) & (bin_idx < n_bins)

    J_bins = np.array([
        np.sum(w[valid & (bin_idx == k)]) / dE_eV[k]
        for k in range(n_bins)
    ])                                      # (n_bins,)  [cm⁻² s⁻¹ sr⁻¹ eV⁻¹]
    E2J_GeV = E_cents**2 * J_bins * 1e-9

    # ── Effective area at bin centres ─────────────────────────────────────────
    _area_map = {
        "poemma":      poemma_area_interp,
        # "ICgen2radio": ICgen2radio_area_interp,
        "grand200k":   grand_area_interp,
    }
    if exp not in _area_map:
        raise ValueError(f"Unknown experiment '{exp}'. Choose from {list(_area_map)}")

    A_eff_bins = np.maximum(_area_map[exp](E_cents), 0.0)   # (n_bins,) [cm² sr]
    # ── Count integrand at bin centres ────────────────────────────────────────
    # Original integrand: J(E) * (1/EnuGeV²) * 1e-9 * A_eff(E)  [s⁻¹ eV⁻¹]
    # Equivalent compact form: J(E) * A_eff(E) / E²  * unit_factor
    unit_factor   = 1e9                              # (1 GeV/eV)² / (1 GeV/eV) → eV → GeV cancel
    integrand_bins = J_bins * A_eff_bins # (n_bins,) [s⁻¹ eV⁻¹]
    # integrand_bins = (E2J_GeV/ E_cents**2 ) * A_eff_bins * unit_factor   # (n_bins,) [s⁻¹ eV⁻¹]

    # print(integrand_bins)
    # ── Integrate in log-energy (trapezoidal, exact for piecewise log-log) ────
    # ∫ f(E) dE = ∫ f(E)·E·ln10  d(log₁₀E)
    # evaluated at bin centres with spacing Δ(log₁₀E) = log_edges[1]-log_edges[0]
    integrand_logE = integrand_bins * E_cents * np.log(10)  # Jacobian absorbed

    # Use only bins with valid (non-zero) flux
    good   = (J_bins > 0) & np.isfinite(integrand_logE)
    counts = np.trapz(
        integrand_logE[good],
        np.log10(E_cents[good]),
    )

    return counts * experimental_factor(exp)

## test SM counts
for flux_dummy in ['no', 'SFR']:
    for exp_dummy in ['grand200k','poemma']:
        d_dummy = 6
        LIVmatrix = np.zeros((3, 3), dtype=complex)
        LIVmatrix[1, 0] = 1e-58  # GeV^{-n}
        count_dummy = compute_tau_counts(
            flux_dummy, SCENARIOS, _EVOL,
            c_cm_s,
            d=d_dummy, a_eff=None, c_eff=None,
            exp=exp_dummy,
            det_flav=2,                         # 0=e, 1=mu, 2=tau (user-specified)
            n_bins=50, log_E_min=15.0, log_E_max=20.0)


        """
        ### LIV effect test
        count_dummy = compute_tau_counts(
            flux_dummy, SCENARIOS, _EVOL,
            c_cm_s,
            d=d_dummy, a_eff=LIVmatrix, c_eff=LIVmatrix,
            exp=exp_dummy,
            det_flav=2,                         # 0=e, 1=mu, 2=tau (user-specified)
            n_bins=50, log_E_min=15.0, log_E_max=20.0)
        """

        print(flux_dummy, exp_dummy, count_dummy)

dt = (timeit.default_timer()-t0)
print(f"time processed:{dt} s")


# fluxtype=input("Enter fluxtype (SFR/no): ")
# exp=input("Enter experiment (poemma/ICgen2radio/grand200k): ")
# d=int(input("Enter dimension: "))
# print((fluxtype, exp, d))
# det_tau = 2


###guess list  
upper_guess = 40 + 10*(d-3)
lower_guess = 20 + 10*(d-3) # or + 5*(d-3) for poemma weak fluxes
srctext = 'simprop'

### 1. perform tau count
"""
guesses = np.logspace(np.log10(10**( - lower_guess )), np.log10(10**( - upper_guess )), 20) 
tau_count_track = {}
tau_count_track[srctext] = {}
print("==========================\ncomputing the number of tau...\n==========================")
std_temp = compute_tau_counts(
    fluxtype, SCENARIOS, _EVOL,
    c_cm_s,
    d=d, a_eff=None, c_eff=None,
    exp=exp,
    det_flav=det_tau,                         # 0=e, 1=mu, 2=tau (user-specified)
    n_bins=50, log_E_min=15.0, log_E_max=20.0)

# for alpha in range(1):
#     for beta in range(alpha,1):
for alpha in range(3):
    for beta in range(alpha,3):
        print(alpha,beta)
        tau_vals = []
        tau_vals_std = []
        for idxsol, sol in enumerate(guesses):
            LIVmatrix = np.zeros((3, 3), dtype=complex)
            LIVmatrix[alpha, beta] = sol  # GeV^{-n}
            bsm_count = compute_tau_counts(
                fluxtype, SCENARIOS, _EVOL,
                c_cm_s,
                d=d, a_eff=LIVmatrix, c_eff=LIVmatrix,
                exp=exp,
                det_flav=det_tau,                         # 0=e, 1=mu, 2=tau (user-specified)
                n_bins=50, log_E_min=15.0, log_E_max=20.0)

            tau_vals.append([sol,bsm_count])
            tau_vals_std.append([sol, std_temp])
            print([idxsol, sol, std_temp, bsm_count])
        tau_vals = np.array(tau_vals)
        tau_vals_std = np.array(tau_vals_std)
        tau_count_track[srctext][alpha,beta] = {'LIV':tau_vals,
                                'std':tau_vals_std}

dt = (timeit.default_timer()-t0)
print(f"time processed:{np.round(dt,2)} s")

print(tau_count_track)
with open(f'data/tau_count/flux{fluxtype}_{exp}_dim{d}.pkl', 'wb') as f:
    pickle.dump(tau_count_track, f)
"""

### 2. computing sensitivity

## defining chi sq function
def chisq(NBSM, N0):
    if NBSM==0 and N0==0:
        return 0 
    if NBSM==0 and N0!=0:
        raise Exception("NBSM is zero, N0 is not!!")
        
    frac = N0/NBSM
    if N0==0:
        frac=1 # evading log(0)
    # print(NBSM,N0,frac)
    return 2*(NBSM - N0 + N0 * np.log(frac))


#### 2A. compute 1D LIVparam sensitivity 
"""
print("==========================\nsolving for parameters\n==========================")

## opening tau count file
with open(f'data/tau_count/flux{fluxtype}_{exp}_dim{d}.pkl', 'rb') as f:
    tau_count_track = pickle.load(f)

LIVparam = {d:{}}

# for src_ratio in src_scan:#[[1,0,0]]:#,[0,1,0],[1,2,0]]:
srctext = 'simprop'
LIVparam[d][srctext] = {}
for (alpha, beta), data in tau_count_track[srctext].items():
    # print(f"{srctext, alpha, beta}")
    tau_vals = data['LIV']   # shape (N,2)
    c_vals = tau_vals[:,0]
    NBSM_vals = tau_vals[:,1]
    N0 = data['std'][:,1][0] # take any value for N0

    # build interpolator on filtered data
    NBSM_function = interp1d(c_vals, NBSM_vals, 
                         kind='linear', fill_value="extrapolate")

    def makezero(cexp):
        c = 10**cexp
        target = chi2.ppf(0.90, 1)
        return chisq(NBSM_function(c), N0) - target        
    ### brentq
    # upper_guess = 40 + 10*(d-3)
    # lower_guess = 20 + 10*(d-3) # or + 5*(d-3) for poemma weak fluxes

    lower_bound_exp = -120 
    upper_bound_exp = -20
    
    # ---- Find a valid bracket automatically ----
    exp_grid = np.linspace(- (40 + 10*(d-3)), - (20 + 10*(d-3)), 1000+100*(d-3))
    fvals = np.array([makezero(e) for e in exp_grid])

    bracket_found = False
    for i in range(len(exp_grid)-1):
        if fvals[i] * fvals[i+1] < 0:
            lower_bound_exp = exp_grid[i]
            upper_bound_exp = exp_grid[i+1]
            bracket_found = True
            break

    if not bracket_found:
        print("No sign change found in scan range.")

    # print(lower_bound_exp, upper_bound_exp)

    try:
        # brentq takes the function and the two ends of the bracket
        cexp_sol = brentq(makezero, lower_bound_exp, upper_bound_exp, xtol=1e-12)
        
        c_solution = 10**cexp_sol
        # solution is within the guess range. This removes absurd extrapolated c_solution. 
        if not (min(c_vals) <= c_solution <= max(c_vals)):
            c_solution = 1e100 # arbitrary large value, not sensitive
        
        print(f"Root found at c = {c_solution:.2e}, Verification (should be ~0): {makezero(cexp_sol)}")

    except ValueError as e:
        print(f"Root find failed: {e}")
        print("This usually means makezero(a) and makezero(b) have the same sign.")
        print("Check if your target is within the range of your chisq values.")
    if not bracket_found:
        c_solution=1e100 # arbitrary large value, not sensitive

    LIVparam[d][srctext][f"{alpha,beta}"] = c_solution

dt = (timeit.default_timer()-t0)
print(f"time processed:{np.round(dt,2)} s")

with open(f'data/param/flux{fluxtype}_{exp}_dim{d}.pkl', 'wb') as f:
    pickle.dump(LIVparam, f)

with open(f'data/param/flux{fluxtype}_{exp}_dim{d}.pkl', 'rb') as f:
    LIVparam = pickle.load(f)
"""


### 3. perform 2D tau count
upper_guess = 66
lower_guess = 50
pos_guesses = np.logspace(-lower_guess, -upper_guess, 12)  # simplified
neg_guesses = -pos_guesses[::-1]  # mirror, reversed so it goes most-neg to least-neg
guesses = np.concatenate([neg_guesses, pos_guesses])  # e.g. [-1e-40,...,-1e-50, 1e-50,...,1e-40]
# print(guesses)
tau_count_track = {}
tau_count_track[srctext] = {}
"""
print("==========================\ncomputing the number of tau...\n==========================")

# Generate all 15 index pairs, numbered 1-15
indices = [(alpha, beta) for alpha in range(3) for beta in range(alpha, 3)]
all_pairs = list(combinations(indices, 2))  # list of 15 pairs

# --- user input ---
pair_id = int(input("Enter pair index (1-15): "))
assert 1 <= pair_id <= 15, "Must be between 1 and 15"

(a1,b1), (a2,b2) = all_pairs[pair_id - 1]
print(f"Computing for ({a1},{b1}) x ({a2},{b2})")

std_temp = compute_tau_counts(
    fluxtype, SCENARIOS, _EVOL,
    c_cm_s,
    d=d, a_eff=None, c_eff=None,
    exp=exp,
    det_flav=det_tau,                         # 0=e, 1=mu, 2=tau (user-specified)
    n_bins=50, log_E_min=15.0, log_E_max=20.0)

# --- compute ---
tau_vals = []
tau_vals_std = []
for idxsol, (sol1, sol2) in enumerate(product(guesses, guesses)):
    LIVmatrix = np.zeros((3, 3), dtype=complex)
    LIVmatrix[a1, b1] = sol1
    LIVmatrix[a2, b2] = sol2
    bsm_count = compute_tau_counts(
        fluxtype, SCENARIOS, _EVOL,
        c_cm_s,
        d=d, a_eff=LIVmatrix, c_eff=LIVmatrix,
        exp=exp,
        det_flav=det_tau,
        n_bins=50, log_E_min=15.0, log_E_max=20.0)
    tau_vals.append([sol1, sol2, bsm_count])
    tau_vals_std.append([sol1, sol2, std_temp])
    print([idxsol, sol1, sol2, std_temp, bsm_count])

tau_vals = np.array(tau_vals)
tau_vals_std = np.array(tau_vals_std)

tau_count_track[srctext][(a1,b1),(a2,b2)] = {'LIV': tau_vals, 'std': tau_vals_std}

# --- save with pair info in filename ---
fname = f'data/tau_count/flux{fluxtype}_{exp}_dim{d}_pair{pair_id:02d}_{a1}{b1}_{a2}{b2}.pkl'
with open(fname, 'wb') as f:
    pickle.dump(tau_count_track, f)
print(f"Saved: {fname}")
dt = (timeit.default_timer()-t0)
print(f"time processed:{np.round(dt,2)} s")

"""

### 4. blindspot count
"""

fluxtype='SFR'
exp='grand200k'
d=6

# ══════════════════════════════════════════════════════════════════
# SM + THREE BLIND-SPOT LIV CASES
# ══════════════════════════════════════════════════════════════════

## emu, etau
# emu_etau_picked = [5.861599453065788e-61, 4.155947048987179e-60]

## mumu, tautau
mumu_tautau_picked = [1.986926102161816e-60, 4.8546516928066826e-61]

# ## etau, mumu
etau_mumu_picked = [1.1701526723938257e-60, 4.110724912959438e-61]

liv_cases = {
    'std': {               # SM
        'indices': [(0, 0)],
        'values':  [0],
    },
    # 'case1_01': {               # [0,1] only
    #     'indices': [(0, 1)],
    #     'values':  [emu_etau_picked[0]],
    # },
    # 'case2_02': {               # [0,2] only
    #     'indices': [(0, 2)],
    #     'values':  [emu_etau_picked[1]],
    # },
    # 'case3_01_02': {            # [0,1] and [0,2]
    #     'indices': [(0, 1), (0, 2)],
    #     'values':  emu_etau_picked,
    # },
    # mmu, tautau
    'case1_11': {               # [0,1] only
        'indices': [(1, 1)],
        'values':  [mumu_tautau_picked[0]],
    },
    'case2_22': {               # [0,2] only
        'indices': [(2, 2)],
        'values':  [mumu_tautau_picked[1]],
    },
    'case3_11_22': {            # [0,1] and [0,2]
        'indices': [(1, 1), (2, 2)],
        'values':  mumu_tautau_picked,
    },
    # ## etau,mumu
    # 'case1_02': {               # [0,1] only
    #     'indices': [(0, 2)],
    #     'values':  [etau_mumu_picked[0]],
    # },
    # 'case2_11': {               # [0,2] only
    #     'indices': [(1, 1)],
    #     'values':  [etau_mumu_picked[1]],
    # },
    # 'case3_02_11': {            # [0,1] and [0,2]
    #     'indices': [(0, 2), (1, 1)],
    #     'values':  etau_mumu_picked,
    # },
    
}

 
# ══════════════════════════════════════════════════════════════════
# 10 ENERGY INTERVALS  [15.0–15.5, 15.5–16.0, ..., 19.5–20.0]
# ══════════════════════════════════════════════════════════════════
energy_edges = np.arange(16.25, 20.01, 0.25)          # 20 bins
# energy_edges = np.arange(18., 18.56, 0.25)          # 20 bins
intervals    = list(zip(energy_edges[:-1], energy_edges[1:]))
 
# ══════════════════════════════════════════════════════════════════
# OUTPUT DIRECTORY
# ══════════════════════════════════════════════════════════════════
outdir = 'data/blindspot'
os.makedirs(outdir, exist_ok=True)
 
# ══════════════════════════════════════════════════════════════════
# COMPUTE
# ══════════════════════════════════════════════════════════════════
results = {}   # results[case_name][interval_idx] = count

std_count = [0.0,0.0,6.51879277,19.22672871,41.55039162,71.54320836,
88.37323341,92.58280576,86.46133404,75.774091,
47.28331302,29.10346358,19.7879381,4.31679087,11.49785068]

if len(std_count)!=len(intervals):
    raise TypeError("std_count and intervals are not of the same shape, either update interval accordingly the std_count or compute new std_count")

t0 = timeit.default_timer()

for case_name, case_def in liv_cases.items():
    print(f"\n{'='*60}\nCase: {case_name}\n{'='*60}")
    results[case_name] = {}
 
    # build the LIV matrix once per case
    LIVmatrix = np.zeros((3, 3), dtype=complex)
    for (a, b), val in zip(case_def['indices'], case_def['values']):
        LIVmatrix[a, b] = val
 
    # for i, (e_min, e_max) in enumerate(intervals):
    #      results['std'][i] = {
    #         'log_E_min': e_min,
    #         'log_E_max': e_max,
    #         'count':     std_count[i],
    #     }

    for i, (e_min, e_max) in enumerate(intervals):
        print(f"  interval {i+1:2d}/{len(intervals)} : log_E [{e_min:.2f}, {e_max:.2f}]", end='  ', flush=True)
 
        count = compute_tau_counts(
            fluxtype, SCENARIOS, _EVOL,
            c_cm_s,
            d=d,
            a_eff=LIVmatrix,
            c_eff=LIVmatrix,
            exp=exp,
            det_flav=det_tau,
            n_bins=50,
            log_E_min=e_min,
            log_E_max=e_max,
        )

        results[case_name][i] = {
            'log_E_min': e_min,
            'log_E_max': e_max,
            'count':     count,
        }
        print(f"count = {count}")
 
dt = timeit.default_timer() - t0
print(f"\nTotal time: {dt:.1f} s")
 
# # # ══════════════════════════════════════════════════════════════════
# # # SAVE
# # # ══════════════════════════════════════════════════════════════════
outfile = os.path.join(outdir, f'blindspot_{fluxtype}_{exp}_dim{d}_11_22.pkl')
with open(outfile, 'wb') as f:
    pickle.dump(results, f)
 
print(f"\nSaved → {outfile}")

"""



