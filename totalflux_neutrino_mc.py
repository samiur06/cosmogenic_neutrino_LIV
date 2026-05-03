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
import awkward as ak

import uproot 

# from LIV_fraction import *

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

### computes flux E^2 J integrating over injection redshift from the root files
def run_flux_pipeline(cosmo_evolution, SCENARIOS, _EVOL,
                      dtdz, c_cm_s, n_bins=24,
                      log_E_min=9.0, log_E_max=21.0,
                      data_dir="SimProp-v2r4/src/data_proton", out_dir="data/total_neutrino_flux"):
    """
    creates all-flavor neutrino flux
    """
    spectral_idx_str, spectral_idx_val, _ = SCENARIOS[cosmo_evolution]

    # ── Load ──────────────────────────────────────────────────────
    branches = ["nNeu", "neuEnergy", "neuFlav", "injRedshift", "injEnergy", "injZ"]
    file_list = [
            f"{data_dir}_{cosmo_evolution}/{f}:summary;1"
            for f in sorted(os.listdir(f"{data_dir}_{cosmo_evolution}"))
            if f.endswith(".root")
        ]
    arrays        = uproot.concatenate(file_list, branches, library="ak")
    injE_arr      = ak.to_numpy(arrays["injEnergy"])
    neu_E_flat    = ak.to_numpy(ak.flatten(arrays["neuEnergy"]))
    inj_z_per_neu, inj_E_per_neu = [
        ak.to_numpy(ak.flatten(ak.broadcast_arrays(arrays[b], arrays["neuEnergy"])[0]))
        for b in ("injRedshift", "injEnergy")
    ]
    N_protons = len(injE_arr)
    print(f"Loaded  N_events={len(ak.to_numpy(arrays['nNeu'])):,}  "
          f"N_neu={len(neu_E_flat):,}  [{cosmo_evolution}, γ={spectral_idx_str}]")

    # ===============================
    # SHAPE  (unnormalised)
    # ===============================
    def Q_shape(E, Z, gamma_inj):
        E     = np.asarray(E, dtype=float)
        Gamma = E / mp_eV
        spec  = np.where(Gamma < Gamma0, Gamma**(-2), Gamma**(-gamma_inj))
        return spec * np.exp(-E / (Z * R_cut_EV * 1e18))

    # ===============================
    # BUILD Q0_spectrum FOR ONE SCENARIO
    # ===============================
    def make_Q0_spectrum(scenario, Z=1):
        """
        Returns a normalised injection-spectrum function for the given scenario.
        Q0_spectrum(E_eV) in cm^{-3} s^{-1} eV^{-1}
        """
        _, gamma_inj, L0_cgs = SCENARIOS[scenario]
        L0_si = L0_cgs / Mpc_cm**3 / yr_s          # erg cm^{-3} s^{-1}

        I, _ = quad(lambda lE: np.log(10) * (10**lE)**2 * Q_shape(10**lE, Z, gamma_inj),
                    np.log10(E_sim_min_eV), np.log10(E_sim_max_eV),
                    limit=500, epsrel=1e-12)

        A_norm = L0_si / (I * eV_erg)
        return lambda E_eV: A_norm * Q_shape(E_eV, Z, gamma_inj)

    # ===============================
    # BUILD & TEST ALL SCENARIOS
    # ===============================
    E_sim_min_eV = injE_arr.min()
    E_sim_max_eV = injE_arr.max()

    Q0_funcs = {}
    for name, (_, gamma_inj, L0_cgs) in SCENARIOS.items():
        Q0_funcs[name] = make_Q0_spectrum(name)
    
    # ── Weights ───────────────────────────────────────────────────
    E_min, E_max = injE_arr.min(), injE_arr.max()
    g = spectral_idx_val
    C = (1.0 / np.log(E_max / E_min)) if np.isclose(g, 1.0) \
        else (1.0 - g) / (E_max**(1.0 - g) - E_min**(1.0 - g))

    w = (
        (c_cm_s / (4.0 * np.pi))
        * (1.0 / N_protons)
        * _EVOL[cosmo_evolution](inj_z_per_neu)
        * dtdz(inj_z_per_neu)
        * 10.0                                        # Delta_z
        * Q0_funcs[cosmo_evolution](inj_E_per_neu)
        / (C * inj_E_per_neu**(-g))
    )

    # ── Flux ──────────────────────────────────────────────────────
    log_edges = np.linspace(log_E_min, log_E_max, n_bins + 1)
    E_cents   = 10**(0.5 * (log_edges[:-1] + log_edges[1:]))
    dE_eV     = E_cents * np.log(10) * (log_edges[1] - log_edges[0])
    bin_idx   = np.digitize(np.log10(neu_E_flat), log_edges) - 1
    valid     = (bin_idx >= 0) & (bin_idx < n_bins)

    J_bins  = np.array([np.sum(w[valid & (bin_idx == k)]) / dE_eV[k] for k in range(n_bins)])
    E2J_GeV = E_cents**2 * J_bins * 1e-9
    print(f"Peak E²J ({cosmo_evolution}): {np.max(E2J_GeV):.3e} GeV cm⁻² s⁻¹ sr⁻¹")

    # ── Save ──────────────────────────────────────────────────────
    fname = f"{out_dir}/flux_{cosmo_evolution}_g{spectral_idx_str}.npy"
    np.save(fname, np.column_stack([E_cents, E2J_GeV]))
    print(f"Saved {fname}\n")

    return None 


cosmo_keys  = ['no', 'SFR']
### snippet to create flux from the function 
"""

for ic, evol in enumerate(cosmo_keys):
    run_flux_pipeline(evol, SCENARIOS, _EVOL, dtdz, c_cm_s)

"""

### ══════════════════════════════════════════════════════════════════
# LOAD & PLOT  (no evolution / SFR ) with other experimental sensitivities
# ══════════════════════════════════════════════════════════════════

SENS_DIR = "NeuExpSensitivity"
GeV2eV   = 1e9

cosmo_keys  = ['no', 'SFR']#, 'AGN']
cosmo_label = ['no evolution', 'SFR evolution', 'AGN evolution']
cosmo_color = ['crimson', 'green', 'blue']

fig, ax = plt.subplots(figsize=(7,5))

# ── Flux curves ───────────────────────────────────────────────────


for ic, evol in enumerate(cosmo_keys):
#     ic, evol = 2, 'AGN'
    spectral_idx = SCENARIOS[evol][0]
    fname = f"data/total_neutrino_flux/flux_{evol}_g{spectral_idx}.npy"
    E_cents, E2J_GeV = np.load(fname).T

    mask = E2J_GeV > 0
    x, y = E_cents[mask]/GeV2eV, E2J_GeV[mask]
    x_smooth = np.logspace(np.log10(x.min()), np.log10(x.max()), 300)
    spl      = make_interp_spline(np.log10(x), np.log10(y), k=3)
    y_smooth = 10**spl(np.log10(x_smooth))
    ax.plot(x_smooth, y_smooth, color=cosmo_color[ic], lw=2, label=cosmo_label[ic])

# ── Sensitivity curves (dashed) ───────────────────────────────────

csv_files = sorted(glob.glob(os.path.join(SENS_DIR, "*.csv")))

label_list = [#'ANITA', 
              'ARA', 'IC-12p6yrdata','Auger', 'BEACON',  
               'POEMMA','GRAND', 'ARIANNA', 'PUEO', 
              'TAMBO', 'Trinity', 'Gen2-radio_optical']

PALETTE = [
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#000000",  # black
    "#009E73",  # bluish green
    "#E69F00",  # orange
    "#999933",  # olive
    "#44AA99",  # teal
    "#882255",  # wine
    "#AA4499",  # purple
    "#117733",  # dark green
    "#0072B2",  # blue
    "#56B4E9",  # sky blue
]
colors = PALETTE
LABEL_MAP = {
    "IC-12p6yrdata": "IceCube",
    # add more remaps here if needed
}

CONSTRAINTS = {"Auger", "IC-12p6yrdata", "ANITA"}
xmin, xmax = 6e15 / GeV2eV, 10**19.4 / GeV2eV   # eV → GeV
ymin, ymax = 1e-10, 1e-7

for i, label in enumerate(label_list):
    matches = [f for f in csv_files if os.path.splitext(os.path.basename(f))[0] == label]
    if not matches:
        raise FileNotFoundError(f"No CSV file found for label '{label}'")
    f = matches[0]
    d = np.loadtxt(f, delimiter=",", comments="#")
    E, F = d[:, 0], d[:, 1]

    display_label = LABEL_MAP.get(label, label)
    is_constraint = label in CONSTRAINTS
    ax.plot(E, F, color=colors[i], lw=1.4,
            ls="-" if is_constraint else "--")
    if is_constraint:
        ax.fill_between(E, F, F * 1e4, color=colors[i], alpha=0.15)

    # ── Only consider points inside the plot range ────────────────
    inside = np.where(
        (E >= xmin) & (E <= xmax) &
        (F >= ymin) & (F <= ymax)
    )[0]

    if len(inside) < 2:
        continue   # skip label if curve barely visible

    # Pick label position within the inside segment
    frac = (i % 12) / 12
    idx  = inside[int(frac * (len(inside) - 1))]
    idx  = np.clip(idx, 1, len(E) - 1)

# ax.set(xscale='log', yscale='log')
ax.set(xscale='log', yscale='log', xlim=(xmin, xmax), ylim=(ymin, ymax),
       xlabel=r'$E$  [GeV]', ylabel=r'All-flavor neutrino flux $E^2 \Phi(E)$  [GeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]')
ax.xaxis.label.set_size(12.5)
ax.yaxis.label.set_size(12.5)

ax.legend(fontsize=12.5, frameon=False, ncol=1)
ax.tick_params(axis='both', which='both', direction='in',
               top=True, right=True, bottom=True, left=True,
               labelsize=12)
ax.tick_params(axis='both', which='major', length=6, width=1.5)
ax.tick_params(axis='both', which='minor', length=3, width=1.0)

annotations = {
    "ARA":        (2244567848.379701,    1.1810717055349855e-08),
    "IceCube":    (11284702.978554811,    1.051120913898513e-08),
    "Auger":      (2109454111.585128,    6.806390270707948e-08),
    "BEACON":     (22295724.31684166,    2.545146566306185e-9),
    "POEMMA":     (38508392.56985678,    9.559612755705455e-9),
    "GRAND":      (66665812.9682111,     2.6910014443109861e-10),
    "ARIANNA":    (170295724.31684166,    4.785382458429474e-09),
    "PUEO":       (510295724.31684166,    5.229098637671127e-08),
    "TAMBO":      (6.5e6,    5.57018445600869e-10), #5957279.614088885
    "Trinity":    (1584111399.89501002,     2.4428364646881043e-09),
    "IceCube Gen2": (474456784.379701,     6.91532591651244e-10),
}

for i, (display_label, (x, y)) in enumerate(annotations.items()):
    if display_label == "TAMBO":
        ax.annotate(display_label,
                    xy=(x, y),
                    xytext=(0, 0), textcoords="offset points",
                    fontsize=12, color=colors[i],
                    ha='left', va='bottom',
                    clip_on=False)
    else:
        ax.annotate(display_label,
                    xy=(x, y),
                    xytext=(23, -15), textcoords="offset points",
                    fontsize=12, color=colors[i],
                    ha='left', va='bottom',
                    clip_on=True)

ax.xaxis.set_tick_params(which='both', top=True)
ax.yaxis.set_tick_params(which='both', right=True)

plt.tight_layout()
# plt.savefig('figures/fitted_flux_with_sensitivities.pdf', dpi=400, bbox_inches='tight')
plt.show()

