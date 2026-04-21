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

import ast
import ternary
import awkward as ak

import uproot 

from LIV_fraction import *

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


### saves necessary data for each neutrino from the root files
def save_per_neu_arrays(cosmo_evolution, data_dir="SimProp-v2r4/src/data_proton", out_dir="data/flux_array"):
    """
    Load SimProp ROOT files and save neutrino info as a single .npz file 
    named after data_dir and cosmo_evolution.
    """
    base_dir = f"{data_dir}_{cosmo_evolution}"
    root_files = sorted(f for f in os.listdir(base_dir) if f.endswith(".root"))

    # ── Summary tree → one row per neutrino ─────────────────────────────────
    arrays = uproot.concatenate(
        [f"{base_dir}/{f}:summary;1" for f in root_files],
        ["event", "neuEnergy", "neuFlav", "injRedshift", "injEnergy", "injZ"],
        library="ak",
    )

    injE_arr = ak.to_numpy(arrays["injEnergy"]) # all the inj energies
    N_protons = len(injE_arr)    

    broadcast    = lambda field: ak.to_numpy(ak.flatten(ak.broadcast_arrays(arrays[field], arrays["neuEnergy"])[0]))
    neu_E_flat   = ak.to_numpy(ak.flatten(arrays["neuEnergy"]))
    flav_per_neu = ak.to_numpy(ak.flatten(arrays["neuFlav"]))
    inj_z_per_neu = broadcast("injRedshift")
    inj_E_per_neu = broadcast("injEnergy")
    evt_flat      = broadcast("event")

    # ── Nuc tree → zOri matched on (evt, Flav) with intmult == 0 ────────────
    nuc = uproot.concatenate(
        [f"{base_dir}/{f}:nuc;1" for f in root_files],
        ["evt", "Flav", "intmult", "zOri"],
        library="ak",
    )
    mask_free = ak.to_numpy(nuc["intmult"]) == 0
    df_nuc = pd.DataFrame({
        "evt":  ak.to_numpy(nuc["evt"] [mask_free]),
        "Flav": ak.to_numpy(nuc["Flav"][mask_free]),
        "zOri": ak.to_numpy(nuc["zOri"][mask_free]),
    }).drop_duplicates(subset=["evt", "Flav"])

    lookup      = df_nuc.set_index(["evt", "Flav"])["zOri"]
    zOri_per_neu = lookup[list(zip(evt_flat, flav_per_neu))].to_numpy()
     
    # ── Cosmo weight ─────────────────────────────────────────────────────────
    cosmo_weight = dtdz(inj_z_per_neu)

    # ── Energy mask ──────────────────────────────────────────────────────────
    mask = (neu_E_flat > 1e15) & (neu_E_flat < 1e20)
    print(f"BEFORE mask: len(cosmo_weight)={len(cosmo_weight):,}")
    neu_E_flat, flav_per_neu, inj_z_per_neu, inj_E_per_neu, cosmo_weight, zOri_per_neu = (
        a[mask] for a in (neu_E_flat, flav_per_neu, inj_z_per_neu, inj_E_per_neu, cosmo_weight, zOri_per_neu)
    )
    print(f"AFTER  mask: len(cosmo_weight)={len(cosmo_weight):,}")

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    data_tag = os.path.basename(data_dir.rstrip("/"))
    out_path = os.path.join(out_dir, f"{data_tag}_{cosmo_evolution}_per_neu.npz")
    np.savez(out_path,
        inj_z_per_neu = inj_z_per_neu,
        inj_E_per_neu = inj_E_per_neu,
        flav_per_neu  = flav_per_neu,
        cosmo_weight  = cosmo_weight,
        neu_E_flat    = neu_E_flat,
        zOri_per_neu  = zOri_per_neu,  
        # injE_arr_min_max = [injE_arr.min(), injE_arr.max()],
        # N_protons = N_protons,    
    )

    print(f"Saved  N_protons={N_protons:,}  N_neu={len(neu_E_flat):,}  →  {out_path}")
    return out_path

### save neutrino data files for various cosmological source cases
### uncomment if you want to extract them from SimProp ROOT files and save them
### ROOT files location must be correct, as taken in base_dir variable inside the function save_per_neu_arrays  
# """
cosmo_keys  = ['no']#, 'SFR']
for ic, evol in enumerate(cosmo_keys):
    save_per_neu_arrays(evol)
    print(f"time processed:{np.round(timeit.default_timer()-t0,2)} s")
# """

