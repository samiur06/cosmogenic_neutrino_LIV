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
# SCENARIOS: (spectral_idx string, spectral_idx, emissivity) 
# — auto-selects spectral_idx and emissivity for the flux
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
    injE_arr_min_max = arr["injE_arr_min_max"]
    N_protons = arr["N_protons"]    

    E_min, E_max = injE_arr_min_max.min(), injE_arr_min_max.max()

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
    # neu_E_flat: neutrino energy at Earth
    # zOri_per_neu: redshift at neutrino production
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
    integrand_bins = J_bins * A_eff_bins # (n_bins,) [s⁻¹ eV⁻¹]

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

## example
for flux_test in ['no', 'SFR']:
    for exp_test in ['grand200k','poemma']:
        d_test = 6
        LIVmatrix = np.zeros((3, 3), dtype=complex)
        LIVmatrix[1, 0] = 1e-58  # GeV^{-n}
        
        SM_count_test = compute_tau_counts(
            flux_test, SCENARIOS, _EVOL,
            c_cm_s,
            d=d_test, a_eff=None, c_eff=None,
            exp=exp_test,
            det_flav=2,                         # 0=e, 1=mu, 2=tau (user-specified)
            n_bins=50, log_E_min=15.0, log_E_max=20.0)

        ### LIV effect test
        liv_count_test = compute_tau_counts(
            flux_test, SCENARIOS, _EVOL,
            c_cm_s,
            d=d_test, a_eff=LIVmatrix, c_eff=LIVmatrix,
            exp=exp_test,
            det_flav=2,                         # 0=e, 1=mu, 2=tau (user-specified)
            n_bins=50, log_E_min=15.0, log_E_max=20.0)


        print(flux_test, exp_test, SM_count_test, liv_count_test)

dt = (timeit.default_timer()-t0)
print(f"time processed:{dt} s")


