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
    'SFR':          ('2.50', 2.5, 6.0e45 * 0.75),
    'AGN':          ('2.40', 2.4, 3.5e45),
}

_EVOL = {
    'no'  : S_noevolution,
    'AGN' : S_AGN,
    'SFR' : S_SFR,
}

def run_flux_pipeline(cosmo_evolution, SCENARIOS, _EVOL,
                      dtdz, c_cm_s, n_bins=24,
                      log_E_min=9.0, log_E_max=21.0,
                      data_dir="SimProp-v2r4/src/data_proton", out_dir="."):
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
    )

    N_protons = len(ak.to_numpy(arrays["injEnergy"]))
    print(f"Saved  N_protons={N_protons:,}  N_neu={len(neu_E_flat):,}  →  {out_path}")
    return out_path

### save neutrino data files for various cosmological source cases
"""
cosmo_keys  = ['no', 'SFR', 'AGN']
for ic, evol in enumerate(cosmo_keys):
    save_per_neu_arrays(evol)
    print(f"time processed:{np.round(timeit.default_timer()-t0,2)} s")
"""

### creates neutrino flux E^2 J  for each flavor
def run_flavored_flux_pipeline(cosmo_evolution, SCENARIOS, _EVOL,
    dtdz, c_cm_s, 
    d, a_eff, c_eff, n_bins=24,
    log_E_min=9.0, log_E_max=21.0,
    data_dir="SimProp-v2r4/src/data_proton",
    npz_dir="data/flux_array", out_dir="."):
    
    # creates flavored neutrino flux with flavor transition 
    # from neutrino production redshift to neutrino detection redshift (i.e. z=0)
    
    spectral_idx_str, spectral_idx_val, _ = SCENARIOS[cosmo_evolution]

    # ── Load per-neutrino arrays from npz ─────────────────────────
    data_tag = os.path.basename(data_dir.rstrip("/"))
    npz_path = os.path.join(npz_dir, f"{data_tag}_{cosmo_evolution}_per_neu.npz")
    arr           = np.load(npz_path)
    inj_z_per_neu = arr["inj_z_per_neu"]
    inj_E_per_neu = arr["inj_E_per_neu"]
    flav_per_neu  = arr["flav_per_neu"]
    cosmo_weight  = arr["cosmo_weight"]
    neu_E_flat    = arr["neu_E_flat"]
    zOri_per_neu  = arr["zOri_per_neu"]      

    N_neu         = len(inj_z_per_neu)
    print(f"Loaded  N_neu={N_neu:,}  from {npz_path}  [{cosmo_evolution}, γ={spectral_idx_str}]")

    # ── We still need injE_arr (per-proton) for N_protons and E_sim bounds ──
    # Load only the per-proton branch — much cheaper than full uproot load
    file_list = [
        f"{data_dir}_{cosmo_evolution}/{f}:summary;1"
        for f in sorted(os.listdir(f"{data_dir}_{cosmo_evolution}"))
        if f.endswith(".root")
    ]
    injE_arr  = ak.to_numpy(
        uproot.concatenate(file_list, ["injEnergy"], library="ak")["injEnergy"]
    )
    N_protons     = len(injE_arr)
    E_sim_min_eV  = injE_arr.min()
    E_sim_max_eV  = injE_arr.max()
    print(f"N_protons={N_protons:,}  E_sim=[{E_sim_min_eV:.2e}, {E_sim_max_eV:.2e}] eV")

    # ── Flavor index (0=e, 1=mu, 2=tau) ──────────────────────────
    # flav_per_neu uses convention 1=e, 2=mu, 3=tau
    flavinc = np.abs(flav_per_neu).astype(int) - 1

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
        _, gamma_inj, L0_cgs = SCENARIOS[scenario]
        L0_si = L0_cgs / Mpc_cm**3 / yr_s

        I, _ = quad(lambda lE: np.log(10) * (10**lE)**2 * Q_shape(10**lE, Z, gamma_inj),
                    np.log10(E_sim_min_eV), np.log10(E_sim_max_eV),
                    limit=500, epsrel=1e-12)

        A_norm = L0_si / (I * eV_erg)
        return lambda E_eV: A_norm * Q_shape(E_eV, Z, gamma_inj)

    Q0_funcs = {name: make_Q0_spectrum(name) for name in SCENARIOS}

    # ── Spectral reweighting ──────────────────────────────────────
    E_min, E_max = injE_arr.min(), injE_arr.max()
    g = spectral_idx_val
    C = (1.0 / np.log(E_max / E_min)) if np.isclose(g, 1.0) \
        else (1.0 - g) / (E_max**(1.0 - g) - E_min**(1.0 - g))

    # ── Oscillation probabilities (N,3,3) → slice to (N,) ────────
    prob_matrix = prob_avg_redshifted(
        neu_E_flat, zOri_per_neu,
        d, a_eff, c_eff)                                              # (N, 3, 3)
    
    for det_flav in range(3):
        prob_per_neu = prob_matrix[np.arange(N_neu), flavinc, det_flav]   # (N,)

        # ── Weights ───────────────────────────────────────────────────
        w = (
            (c_cm_s / (4.0 * np.pi))
            * (1.0 / N_protons)
            * _EVOL[cosmo_evolution](inj_z_per_neu)
            * cosmo_weight # dtdz(inj_z_per_neu)
            * prob_per_neu # flavor transition with neutrino production redshift
            * 10.0                                     # Delta_z
            * Q0_funcs[cosmo_evolution](inj_E_per_neu)
            / (C * inj_E_per_neu**(-g))
        )
        assert w.shape == (N_neu,), f"Weight shape mismatch: {w.shape}"

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
        fname = f"{out_dir}/flux_{cosmo_evolution}_g{spectral_idx_str}_det_flav{det_flav}.npy"
        np.save(fname, np.column_stack([E_cents, E2J_GeV]))
        print(f"Saved {fname}\n")

    return None
#############################################


### creates E^2 J fluxes for various cosmological source cases
"""
# cosmo_keys  = ['no', 'SFR', 'AGN']
# LIVmatrix = np.zeros((3, 3), dtype=complex)
# LIVmatrix[0,1] = 1e-35  # GeV^{-n}
# d = 4

# for ic, evol in enumerate(cosmo_keys[1:2]):
#     print(evol)
#     run_flavored_flux_pipeline(evol, SCENARIOS, _EVOL, dtdz, c_cm_s, 
#               #                       d, a_eff=LIVmatrix, c_eff=LIVmatrix, n_bins=24,
#                       log_E_min=9.0, log_E_max=21.0,
#                       data_dir="SimProp-v2r4/src/data_proton",
#                       npz_dir="data/flux_array",
#                       out_dir=".")
#     print(f"time processed:{np.round(timeit.default_timer()-t0,2)} s")
"""

#############################################
### SANITY FLUX
#############################################
# ── Load per-flavor npy files ─────────────────────────────────────────────────
# """
flav_data = {}
for flav in range(3):
    d = np.load(f"flux_SFR_g2.50_det_flav{flav}.npy")   # columns: [E_cents, E2J_GeV]
    flav_data[flav] = d

E_cents = flav_data[0][:, 0]   # same for all flavors

# ── Sum over all flavors ──────────────────────────────────────────────────────
E2J_total = sum(flav_data[f][:, 1] for f in range(3))

# ── Load CSV reference ────────────────────────────────────────────────────────
csv = pd.read_csv("flux_simprop/flux_SFR_g2.50.csv")
# csv columns: log10Enu (log10 E/eV), rate (GeV cm^-2 s^-1 sr^-1)
E_csv   = 10**csv["log10Enu"].values   # eV
E2J_csv = csv["rate"].values           # GeV cm^-2 s^-1 sr^-1

# ── Numerical comparison: interpolate csv onto npy E grid ────────────────────
E2J_csv_interp = np.interp(E_cents, E_csv, E2J_csv, left=np.nan, right=np.nan)

ratio = E2J_total / E2J_csv_interp
valid = np.isfinite(ratio)

print("=" * 55)
print(f"{'Bin':>4}  {'E [eV]':>10}  {'npy sum':>12}  {'csv':>12}  {'ratio':>8}")
print("-" * 55)
for i in np.where(valid)[0]:
    print(f"{i:>4}  {E_cents[i]:>10.2e}  {E2J_total[i]:>12.3e}  "
          f"{E2J_csv_interp[i]:>12.3e}  {ratio[i]:>8.4f}")
print("=" * 55)
print(f"Mean ratio (npy_sum / csv): {np.nanmean(ratio):.4f}")
print(f"Max  ratio deviation from 1: {np.nanmax(np.abs(ratio - 1)):.4f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
flav_labels = {0: r"$\nu_e$", 1: r"$\nu_\mu$", 2: r"$\nu_\tau$"}
flav_colors = {0: "steelblue", 1: "tomato", 2: "mediumseagreen"}

fig, axes = plt.subplots(2, 1, figsize=(7, 5),
                         gridspec_kw={"height_ratios": [2, 1]})

ax = axes[0]
for flav in range(3):
    ax.plot(E_cents, flav_data[flav][:, 1],
            color=flav_colors[flav], lw=1.5, ls="--",
            label=f"npy {flav_labels[flav]}")

ax.plot(E_cents, E2J_total,
        color="black", lw=2.0, ls='--' , label="npy total (sum)")
ax.plot(E_csv, E2J_csv,
        color="orange", lw=2.0, ls="-.", label="SimProp csv")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1e15,1e20)
ax.set_ylim(1e-11,5e-7)
ax.set_xlabel(r"$E_\nu$ [eV]", fontsize=13)
ax.set_ylabel(r"$E^2 J$ [GeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]", fontsize=13)
ax.set_title("SFR  $\gamma=2.50$ — per-flavor and total neutrino flux", fontsize=13)
ax.legend(fontsize=12)
ax.grid(True, which="both", alpha=0.3)

# ── Ratio panel ───────────────────────────────────────────────────────────────
ax2 = axes[1]
ax2.plot(E_cents[valid], ratio[valid], color="black", lw=1.5, label="total / csv")
ax2.axhline(1.0, color="orange", lw=1.5, ls="--")
ax2.set_xscale("log")
ax2.set_xlabel(r"$E_\nu$ [eV]", fontsize=13)
ax2.set_ylabel("npy sum / csv", fontsize=12)
ax2.set_xlim(1e15,1e20)
ax2.set_ylim(0, 2)
ax2.grid(True, which="both", alpha=0.3)
ax2.legend(fontsize=12)

plt.tight_layout()
# plt.savefig("flux_SFR_flavoredSum_sanity.pdf", dpi=200)
print("\nPlot saved: flux_SFR_flavoredSum_sanity.pdf")
plt.show()
# """


