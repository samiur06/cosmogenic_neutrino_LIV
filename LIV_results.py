import numpy as np
from sympy import sin, cos 
import sympy as sp
import scipy as scp
import pandas as pd
from cmath import *
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
from matplotlib.colors import LogNorm
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from scipy.integrate import odeint
from scipy.optimize import fsolve
from scipy.ndimage import label, binary_dilation
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

t0 = timeit.default_timer()

exp=input("Enter experiment (poemma/ICgen2radio/grand200k): ")
print((exp))

flux_label = ['no evolution', 'SFR evolution']#, 'AGN evolution']

# get alpha,beta keys from one entry
fixed_srctext = 'simprop' #'1.00.00.0'#'0.01.00.0'#'1.02.00.0'

alphabeta_keys = ['(0, 0)', '(0, 1)', '(0, 2)', '(1, 1)', '(1, 2)', '(2, 2)']  # 6 of them
x = np.arange(len(alphabeta_keys))
width = 0.30
flavlabel = [r'$e$',r'$\mu$',r'$\tau$']

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

### dimensions and exp for 'simprop' flux
"""
plt.figure(figsize=(8,5))
y_top = 1e-1
cmap = plt.get_cmap("tab10")  # very distinguishable

# === ADD BEFORE THE MAIN LOOP ===
no_sensitivity = {
    'grand200k': {
        'no':  [(0,0),(1,2)],        # mu tau
        'SFR': [(1,2)],        # mu tau
    },
    'poemma': {
        'no':  [(0,0),(0,2),(1,2)],  # ee, mu tau
        'SFR': [(0,0),(1,2)],  # ee, mu tau
    },
}

line_flux = [(0,(3,1,1,1,1,1)), (0,(1,1))] #(0,(1,2))]
# line_flux = ['--',(0,(1,0.5)),(0,(1,2))]
drange = np.arange(3,9)
# to simplify making the legend for dimensions
for d in drange:
    base_color = cmap((d-3)/5)  # normalize d in [0,1]
    # some lines that doesn't appear in the plot
    plt.hlines(
        10, 
        x[0],   # left end
        x[1],   # right end
        color=base_color,
        linewidth=2,
        linestyle='-'
    )
# to simplify making the legend for flux types
for idx_ls, ls in enumerate(line_flux):
    plt.hlines(
        10*idx_ls, 
        x[0],   # left end
        x[1],   # right end
        color='black',
        linewidth=2,
        linestyle=ls
    )
for d in drange:
    print(d)
    base_color = cmap((d-3)/5)  # normalize d in [0,1]
    colors = [base_color[:3] + (0.4 + 0.6*j/len(fixed_srctext),) 
              for j in range(len(fixed_srctext))]
    
    for iflux, flux_var in enumerate(['no','SFR']):
        with open(f'data/param/flux{flux_var}_{exp}_dim{d}.pkl', 'rb') as f:
            LIVparam = pickle.load(f)
        # get alpha,beta keys from one entry
        alphabeta_keys = list(LIVparam[d][fixed_srctext].keys())  # 6 of them
        # prepare data matrix: rows = alpha,beta, cols = srctext
        vals = np.zeros((len(alphabeta_keys), 1))
        for i, ab in enumerate(alphabeta_keys):
            for j, sr in enumerate([fixed_srctext]):
                vals[i, j] = LIVparam[d][sr][ab]
        y = np.log10(vals[:, 0]) - np.log10(y_top)

        for idx, (xi, yi) in enumerate(zip(x + iflux*width, y)):
            flagged = no_sensitivity.get(exp, {}).get(flux_var, [])
            ab = ast.literal_eval(alphabeta_keys[idx])
            if ab not in flagged:
                plt.hlines(
                    yi,
                    xi - width/2,   # left end
                    xi + width/2,   # right end
                    color=base_color,
                    linewidth=2,
                    linestyle=line_flux[iflux]
                )

            if d==8 and ab in flagged:
                plt.text(
                    xi, -40,
                    'no projected sensitivity',
                    color='red', fontsize=12,
                    ha='center', va='top',
                    rotation=90,
                    fontweight='bold'
                )


            # Vertical line (like a bar) when d=max , largest bar
            if d==8:
                if yi>0 or ab in flagged:
                    set_dip=-75
                else:
                    set_dip=yi
                for xedge in [xi - width/2,xi + width/2]:
                    plt.vlines(
                        xedge,
                        set_dip,   # bottom
                        -20,       # top
                        color='black',#base_color,
                        linewidth=1.5,
                        linestyle='solid'#line_flux[iflux]
                    )

# x-ticks: alpha,beta labels
plt.xticks(x + 0.5*width, 
          [f"{flavlabel[k[0]]}{flavlabel[k[1]]}"
           for k in (ast.literal_eval(x) for x in alphabeta_keys)],
          fontsize=14)
# plt.yscale("log")
plt.ylim(np.log10(1e-20), -100)
plt.gca().invert_yaxis()
plt.ylabel(r"$\log _{10}{(\mathring{{\kappa}}^{{({d})}}_{{\alpha\beta}}/\,\mathrm{GeV}^{4-d})}$", fontsize=14)
plt.xlabel(r"$(\alpha,\beta)$", fontsize=14)


legend_tags = [f'd={x}' for x in drange] + flux_label
plt.legend(legend_tags, 
    frameon=False, fontsize=14,
    ncol=3,
    loc='lower left', 
    bbox_to_anchor=(0.0, 0.0))

# total_ratio = int(sum(src_ratio))
# s = ",".join(
#     f"{int(v)}/{total_ratio}" if (v and total_ratio != 1) 
#     else str(int(v))
#     for v in src_ratio
#     )
# label = rf"$f_S=({s})$"
# plt.text(0.97, 0.07, label, transform=plt.gca().transAxes,
#         ha="right", va="top", fontsize=14)

plt.tick_params(top=False, bottom=False, right=True,
                direction="in", which="both")
plt.minorticks_on()
plt.tight_layout()
plt.savefig(f"figures/param_{exp}_src{fixed_srctext}.pdf", dpi=300)
plt.show()
print(f"time processed:{np.round(timeit.default_timer(),2)} s")
"""

### plot count vs LIV fixed dim d 
# """
flux_color = ["#0072B2",  # blue
          "#D55E00",  # vermillion (orange-red)
          "#009E73"]  # bluish green
line_flux = ['--',(0,(1,0.5)),'-']#(0,(1,2))]
drange = np.arange(3,9)
cmap = plt.get_cmap("tab10")  # very distinguishable
flavlabel = [r"e", r"\mu", r"\tau"]

fig, axes = plt.subplots(1, 1, figsize=(10, 8), sharex=True, sharey=True)

# to simplify making the legend for std
axes.hlines(
    10000000, 
    10,   # left end
    20,   # right end
    color='black',
    linewidth=2,
    linestyle=':',
    label='std'
)

cmin=10
cmax=0
for d in [6]:#drange:
    colors = plt.cm.Greys(np.linspace(0.28, 0.95, 6))
    linestyles = ["-", "--", "-.", (0,(1,1)), (0,(3,1,1,1)), (0,(5,2))]

    base_color = cmap((6-3)/5)  # normalize d in [0,1]
    # for iflux, flux_var in enumerate(flux_list):
    iflux, flux_var = 1, 'SFR'

    with open(f'data/tau_count/flux{flux_var}_{exp}_dim{d}.pkl', 'rb') as f:
        tau_count_track = pickle.load(f)

        srctext = 'simprop'

    ab_list = list(tau_count_track[srctext].keys())
    for iax, (alpha, beta) in enumerate(ab_list):
        ax = axes#[iax]

        data = tau_count_track[srctext][(alpha,beta)]
        tau_vals = data['LIV']   # shape (N,2)
        c_vals = tau_vals[:,0]
        NBSM_vals = tau_vals[:,1]

        # keep only positive values (log-safe)
        mask = (c_vals > 0) & (NBSM_vals > 0)
        c_vals = c_vals[mask]
        NBSM_vals = NBSM_vals[mask]

        # fit in log-log space
        logc = np.log10(c_vals)
        logN = np.log10(NBSM_vals)

        # build linear interpolator (no extrapolation on the right)
        f = interp1d(logc, logN, kind='linear', bounds_error=False)
        # define extrapolation grid
        c_ext = c_vals #np.logspace(-150, -30, 500)
        logc_ext = np.log10(c_ext)

        logN_ext = f(logc_ext)
        # for values outside c grid, keep constant
        logN_ext[logc_ext < logc.min()] = logN[logc.argmin()]
        logN_ext[logc_ext > logc.max()] = logN[logc.argmax()]

        NBSM_ext = 10**logN_ext

        # plot extrapolated curve
        ax.plot(
            # c_ext, NBSM_ext,
            c_vals, NBSM_vals,
            linestyle=linestyles[iax],#line_flux[iflux],
            color=colors[iax],
            linewidth=2,
            label=rf"$\mathring{{\kappa}}^{{({d})}}_{{{flavlabel[alpha]}{flavlabel[beta]}}}$"
        )
        ax.set_xlabel(rf"$\mathring{{\kappa}}^{{({d})}}_{{\alpha\beta}}\,[\mathrm{{GeV}}^{{{4-d}}}]$", fontsize=30)
    
        # horizontal N0 line 
        N0 = data['std'][:,1][0]
        # N0 = tau_count(exp, src_ratio=src_ratio, fluxtype=flux_var)
        ax.plot(
            c_vals, data['std'][:,1],
            linestyle=':',
            color='k',#flux_color[iflux],
            linewidth=2,
            # label=f'std'
        )   

ax.set_xscale('log')
# ax.set_xlim(cmin, cmax)
# ax.set_xlim(min(c_ext), max(c_ext))
ax.set_xlim(1e-64, 9e-51)
ax.set_ylim(min(NBSM_vals)-1, max(NBSM_vals)+40)

ax.set_ylabel(r"$N_{\nu_\tau}(\mathring{\kappa}^{(d)}_{\alpha\beta})$", fontsize=30)
ax.tick_params(top=True, bottom=True, right=True,
                direction="in", which="both")
ax.tick_params(axis='both', which='major', labelsize=24, length=8, width=1.5)  # major ticks
ax.tick_params(axis='both', which='minor', labelsize=16, length=4, width=1.0)  # minor ticks

ax.minorticks_on()

# legend only once
ax.legend(frameon=False, fontsize=29,
    ncol=1,
    loc='lower left', )
    # bbox_to_anchor=(0.0, 0.0))

# plt.suptitle(f"d = {d}", fontsize=16)

plt.tight_layout()
plt.savefig(f"figures/tau_count_d{d}_{exp}_src{srctext}.pdf", dpi=300)
plt.show()
print(f"time processed:{np.round(timeit.default_timer(),2)} s")
# """

### plot chi2 vs LIV fixed dim d uniform
"""
flux_color = ["#0072B2",  # blue
          "#D55E00",  # vermillion (orange-red)
          "#009E73"]  # bluish green
line_flux = ['--',(0,(1,0.5)),'-']#(0,(1,2))]
drange = np.arange(3,9)
cmap = plt.get_cmap("tab10")  # very distinguishable
flavlabel = [r"e", r"\mu", r"\tau"]
fig, axes = plt.subplots(1, 1, figsize=(10, 8), sharex=True, sharey=True)
ax = axes

cmin=10
cmax=0
for d in [6]:#drange:
    colors = plt.cm.Greys(np.linspace(0.28, 0.95, 6))
    linestyles = ["-", "--", "-.", (0,(1,1)), (0,(3,1,1,1)), (0,(5,2))]

    base_color = cmap((6-3)/5)  # normalize d in [0,1]
    # for iflux, flux_var in enumerate(flux_list):
    iflux, flux_var = 1, 'SFR'
    fluxtype = flux_var
    srctext = 'simprop'

    LIVparam = {d:{}}

    with open(f'data/tau_count/flux{fluxtype}_{exp}_dim{d}.pkl', 'rb') as f:
        totNBSM_dict = pickle.load(f)

    LIVparam[d][srctext] = {}

    for iax, (alpha,beta) in enumerate(totNBSM_dict[srctext].keys()):

        data = totNBSM_dict[srctext][alpha,beta]
        tau_vals = data['LIV']   # shape (N,2)
        c_vals = tau_vals[:,0]
        NBSM_vals = tau_vals[:,1]
        N0 = data['std'][:,1][0]
        ### build interpolator using inter1d
        # NBSM_function = interp1d(c_vals, NBSM_vals, 
        #                      kind='linear', fill_value="extrapolate")

        # target = chi2.ppf(0.90, 1)

        # c_ext = np.logspace(np.log10(c_vals.min()), np.log10(c_vals.max()), 100)
        # chisq_vals = np.array([chisq(float(NBSM_function(c)), N0) for c in c_ext])

        ### build interpolator using PchipInterpolator
        from scipy.interpolate import PchipInterpolator

        c_vals_clean, idx = np.unique(c_vals, return_index=True)
        NBSM_vals_clean = NBSM_vals[idx]

        NBSM_function = PchipInterpolator(c_vals_clean, NBSM_vals_clean)
        target = chi2.ppf(0.90, 1)
        c_ext = np.logspace(np.log10(c_vals_clean.min()), np.log10(c_vals_clean.max()), 1000)
        chisq_vals = np.array([chisq(float(NBSM_function(c)), N0) for c in c_ext])

        ax.plot(
            c_ext, chisq_vals,
            linestyle=linestyles[iax],
            color=colors[iax],
            linewidth=2,
            label=rf"$\mathring{{\kappa}}^{{({d})}}_{{{flavlabel[alpha]}{flavlabel[beta]}}}$")


ax.axhline(target, color='k', linestyle=':', linewidth=3.0, label=f'90\% CL')


ax.set_xlabel(rf"$\mathring{{\kappa}}^{{({d})}}_{{\alpha\beta}}\,[\mathrm{{GeV}}^{{{4-d}}}]$", fontsize=30)

ax.set_xscale('log')
ax.set_yscale('log')
# ax.set_xlim(1e-64, max(c_ext))
ax.set_xlim(1e-64, 9e-51)
ax.set_ylim(9e-2, 6e5)
ax.set_ylabel(r"$-2\log { L }(\mathring{\kappa}^{(d)}_{\alpha\beta})$", fontsize=30)
ax.tick_params(top=True, bottom=True, right=True,
                direction="in", which="both")
ax.tick_params(axis='both', which='major', labelsize=24, length=8, width=1.5)  # major ticks
ax.tick_params(axis='both', which='minor', labelsize=16, length=4, width=1.0)  # minor ticks
ax.minorticks_on()

# legend only once
ax.legend(frameon=False, fontsize=30,
    ncol=2,
    loc='upper left', )
    # bbox_to_anchor=(0.0, 0.0))

plt.tight_layout()
# plt.savefig(f"figures/chi2_d{d}_{exp}_src{srctext}.pdf", dpi=300)
plt.show()
print(f"time processed:{np.round(timeit.default_timer(),2)} s")
"""

### plot 2D count for 15 pairs 
"""
all_pairs = [
    ((0,0),(0,1)), ((0,0),(0,2)), ((0,0),(1,1)), ((0,0),(1,2)), ((0,0),(2,2)),
    ((0,1),(0,2)), ((0,1),(1,1)), ((0,1),(1,2)), ((0,1),(2,2)), ((0,2),(1,1)),
    ((0,2),(1,2)), ((0,2),(2,2)), ((1,1),(1,2)), ((1,1),(2,2)), ((1,2),(2,2)),
]

srctext = 'simprop'

idx_to_greek = {0: "e", 1: r"\mu", 2: r"\tau"}

fig, axes = plt.subplots(3, 5, figsize=(19, 8), constrained_layout=True)

for pair_id in range(1, 16):
    (a1, b1), (a2, b2) = all_pairs[pair_id - 1]
    ax = axes.flatten()[pair_id - 1]

    fname = f"data/tau_count/fluxSFR_grand200k_dim6_pair{pair_id:02d}_{a1}{b1}_{a2}{b2}.pkl"
    with open(fname, "rb") as f:
        tau_count_track = pickle.load(f)

    data = tau_count_track[srctext][(a1,b1),(a2,b2)] 

    liv = np.array(data["LIV"])   # shape (N, 3): [sol1, sol2, bsm_count]
    std = np.array(data["std"])   # shape (N, 3): [sol1, sol2, std_count]
    x = np.unique(liv[:,0])
    y = np.unique(liv[:,1])

    nx, ny = len(x), len(y)

    Z = np.full((nx, ny), np.nan)

    # fill grid safely
    for xi, yi, zi in liv:
        ix = np.where(x == xi)[0][0]
        iy = np.where(y == yi)[0][0]
        Z[ix, iy] = zi/std[0,2] # dividing by the standard value, randomly choosing 1st one

    X, Y = np.meshgrid(x, y)

    linthresh_x = np.percentile(np.abs(x[x!=0]),5)
    linthresh_y = np.percentile(np.abs(y[y!=0]),5)

    norm = TwoSlopeNorm(vmin=np.nanmin(Z), vcenter=np.nanmean(Z), vmax=np.nanmax(Z))
    pm = ax.pcolormesh(X, Y, Z.T, cmap="turbo", norm=norm, shading="nearest")
    cbar = fig.colorbar(pm, ax=ax, fraction=0.046, pad=0.04)
    # cbar.set_ticks([np.nanmin(Z), 0.5*(np.nanmin(Z)+np.nanmax(Z)), np.nanmax(Z)])
    # cbar.set_ticks([0.2,0.4,0.6,0.8,1.0])
    cbar.ax.tick_params(labelsize=6)
    ax.set_xscale('symlog', linthresh=linthresh_x, linscale=1.0)  # wider linear window
    ax.set_yscale('symlog', linthresh=linthresh_y, linscale=1.0)


    def make_ticks(arr):
        pos = arr[arr > 0]
        exps = np.unique(np.round(np.log10(pos)).astype(int))

        # pick two representative exponents
        idx = np.linspace(0, len(exps)-1, 2).astype(int)
        e1, e2 = exps[idx]

        return [-10.0**e2, -10.0**e1, 10.0**e1, 10.0**e2]

    # ax.set_xticks(make_ticks(x))
    # ax.set_yticks(make_ticks(y))
    ticks = [-1e-50, -1e-60, 0, 1e-60, 1e-50]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    def symlog_fmt(val, pos):
        if val == 0: return '0'
        exp = int(np.round(np.log10(abs(val))))
        return rf'$10^{{{exp}}}$' if val > 0 else rf'$-10^{{{exp}}}$'

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(symlog_fmt))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(symlog_fmt))
    g1 = f"{idx_to_greek[a1]}{idx_to_greek[b1]}"
    g2 = f"{idx_to_greek[a2]}{idx_to_greek[b2]}"
    ax.set_xlabel(rf"$\mathring{{\kappa}}^{{(6)}}_{{{g1}}}$  [GeV$^{{-2}}$]", fontsize=12)
    ax.set_ylabel(rf"$\mathring{{\kappa}}^{{(6)}}_{{{g2}}}$  [GeV$^{{-2}}$]", fontsize=12)
    ax.set_title(f"Pair {pair_id:02d}", fontsize=12)
    ax.minorticks_on()
ax.tick_params(axis='both', which='both',
               direction='in',
               top=True, right=True,
               labelsize=8)
    # print(f"time processed:{np.round(timeit.default_timer()-t0,2)} s")

# fig.suptitle(r"$N_{\nu_{\tau}}$ — dim 6, flux SFR", fontsize=13)
# fig.suptitle(r"$N_\mathrm{BSM}/N_\mathrm{std}$ — dim 6, flux SFR", fontsize=13)
# plt.savefig("figures/density_grid.pdf", dpi=200, bbox_inches="tight")
plt.show()
"""

### plot 2D chi sqr for 15 pairs 
"""
all_pairs = [
    ((0,0),(0,1)), ((0,0),(0,2)), ((0,0),(1,1)), ((0,0),(1,2)), ((0,0),(2,2)),
    ((0,1),(0,2)), ((0,1),(1,1)), ((0,1),(1,2)), ((0,1),(2,2)), ((0,2),(1,1)),
    ((0,2),(1,2)), ((0,2),(2,2)), ((1,1),(1,2)), ((1,1),(2,2)), ((1,2),(2,2)),
]

srctext = 'simprop'

idx_to_greek = {0: "e", 1: r"\mu", 2: r"\tau"}

fig, axes = plt.subplots(3, 5, figsize=(19, 8), constrained_layout=True)

# === BUILD 1D CONSTRAINT LOOKUP (add before the pair loop) ===
with open(f'data/param/fluxSFR_grand200k_dim6.pkl', 'rb') as f:
    LIVparam_1d = pickle.load(f)

single_constraints = {}  # key: (a,b) tuple, value: positive constraint value
for ab_key in LIVparam_1d[6][srctext]:
    ab = ast.literal_eval(ab_key) if isinstance(ab_key, str) else ab_key
    single_constraints[ab] = LIVparam_1d[6][srctext][ab_key]

# by-hand mu-tau, making it blind because it is! 
single_constraints[1,2] = 1e-50 # for dim 6

for pair_id in range(1, 16):
    (a1, b1), (a2, b2) = all_pairs[pair_id - 1]
    ax = axes.flatten()[pair_id - 1]

    fname = f"data/tau_count/fluxSFR_grand200k_dim6_pair{pair_id:02d}_{a1}{b1}_{a2}{b2}.pkl"
    with open(fname, "rb") as f:
        tau_count_track = pickle.load(f)

    data = tau_count_track[srctext][(a1,b1),(a2,b2)] 

    liv = np.array(data["LIV"])   # shape (N, 3): [sol1, sol2, bsm_count]
    std = np.array(data["std"])   # shape (N, 3): [sol1, sol2, std_count]
    x = np.unique(liv[:,0])
    y = np.unique(liv[:,1])

    nx, ny = len(x), len(y)

    Z = np.full((nx, ny), np.nan)
    N0 = std[0,2]

    # fill grid safely
    for xi, yi, zi in liv:
        ix = np.where(x == xi)[0][0]
        iy = np.where(y == yi)[0][0]
        NBSM = zi
        Z[ix, iy] = chisq(NBSM, N0) # dividing by the standard value, randomly choosing 1st one

    X, Y = np.meshgrid(x, y)

    linthresh_x = np.percentile(np.abs(x[x!=0]),5)
    linthresh_y = np.percentile(np.abs(y[y!=0]),5)
    
    Z = np.where(Z <= 0, np.nan, Z)  # or replace 0 with a small positive number
    # norm = LogNorm(vmin=np.nanmin(Z), vmax=np.nanmax(Z))
    # pm = ax.pcolormesh(X, Y, Z.T, cmap="turbo", norm=norm, shading="nearest")
    # fig.colorbar(pm, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=6)

    # 90% CL contour
    threshold = chi2.ppf(0.90, df=2)  # 4.605
    # Filled contour (inside = allowed region at 90% CL)
    ax.contourf(X, Y, Z.T, levels=[0, threshold], colors=['blue'], alpha=0.1)
    ax.contour(X, Y, Z.T, levels=[threshold], colors=['blue'], 
               linestyles=['-'], linewidths=2.5)

    # === ADD 1D CONSTRAINT BOX ===
    cx = single_constraints.get((a1,b1), None)
    cy = single_constraints.get((a2,b2), None)
    # Filled 1D constraint box
    if cx is not None and cy is not None:
        from matplotlib.patches import Rectangle
        rect = Rectangle((-cx, -cy), 2*cx, 2*cy,
                         linewidth=2.5, edgecolor='red',
                         facecolor='purple', alpha=0.25,
                         linestyle='-', zorder=3)
        ax.add_patch(rect)

    ax.set_xscale('symlog', linthresh=linthresh_x, linscale=1.0)  # wider linear window
    ax.set_yscale('symlog', linthresh=linthresh_y, linscale=1.0)

    ticks = [-1e-50, -1e-55, -1e-60, 0, 1e-60, 1e-55, 1e-50]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    def symlog_fmt(val, pos):
        if val == 0: return '0'
        exp = int(np.round(np.log10(abs(val))))
        return rf'$10^{{{exp}}}$' if val > 0 else rf'$-10^{{{exp}}}$'

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(symlog_fmt))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(symlog_fmt))
    g1 = f"{idx_to_greek[a1]}{idx_to_greek[b1]}"
    g2 = f"{idx_to_greek[a2]}{idx_to_greek[b2]}"
    # ax.set_xlim(-1e-55,1e-55)
    # ax.set_ylim(-1e-55,1e-55)
    ax.set_xlabel(rf"$\mathring{{\kappa}}^{{(6)}}_{{{g1}}}$  [GeV$^{{-2}}$]", fontsize=12)
    ax.set_ylabel(rf"$\mathring{{\kappa}}^{{(6)}}_{{{g2}}}$  [GeV$^{{-2}}$]", fontsize=12)
    ax.set_title(f"Pair {pair_id:02d}", fontsize=12)
    ax.minorticks_on()
ax.tick_params(axis='both', which='both',
               direction='in',
               top=True, right=True,
               labelsize=8)
    # print(f"time processed:{np.round(timeit.default_timer()-t0,2)} s")

# fig.suptitle(r"$\chi^2$ — dim 6, SFR flux", fontsize=13)
# plt.savefig("figures/chi2_grid.pdf", dpi=200, bbox_inches="tight")
# plt.savefig("figures/compare_2D_1D.pdf", dpi=200, bbox_inches="tight")
plt.show()
"""

### plot selected 2D chi sqr for 4 pairs 
"""
all_pairs = [
    ((0,0),(0,1)), ((0,0),(0,2)), ((0,0),(1,1)), ((0,0),(1,2)), ((0,0),(2,2)),
    ((0,1),(0,2)), ((0,1),(1,1)), ((0,1),(1,2)), ((0,1),(2,2)), ((0,2),(1,1)),
    ((0,2),(1,2)), ((0,2),(2,2)), ((1,1),(1,2)), ((1,1),(2,2)), ((1,2),(2,2)),
]

srctext = 'simprop'

idx_to_greek = {0: "e", 1: r"\mu", 2: r"\tau"}

fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=False)

# === BUILD 1D CONSTRAINT LOOKUP (add before the pair loop) ===
with open(f'data/param/fluxSFR_grand200k_dim6.pkl', 'rb') as f:
    LIVparam_1d = pickle.load(f)

single_constraints = {}  # key: (a,b) tuple, value: positive constraint value
for ab_key in LIVparam_1d[6][srctext]:
    ab = ast.literal_eval(ab_key) if isinstance(ab_key, str) else ab_key
    single_constraints[ab] = LIVparam_1d[6][srctext][ab_key]

# by-hand mu-tau, making it blind because it is! 
single_constraints[1,2] = 1e-50 # for dim 6

for idx_pair, pair_id in enumerate([6,7,12,14]):
    (a1, b1), (a2, b2) = all_pairs[pair_id - 1]
    ax = axes.flatten()[idx_pair]

    fname = f"data/tau_count/fluxSFR_grand200k_dim6_pair{pair_id:02d}_{a1}{b1}_{a2}{b2}.pkl"
    with open(fname, "rb") as f:
        tau_count_track = pickle.load(f)

    data = tau_count_track[srctext][(a1,b1),(a2,b2)] 

    liv = np.array(data["LIV"])   # shape (N, 3): [sol1, sol2, bsm_count]
    std = np.array(data["std"])   # shape (N, 3): [sol1, sol2, std_count]
    x = np.unique(liv[:,0])
    y = np.unique(liv[:,1])

    nx, ny = len(x), len(y)

    Z = np.full((nx, ny), np.nan)
    N0 = std[0,2]

    # fill grid safely
    for xi, yi, zi in liv:
        ix = np.where(x == xi)[0][0]
        iy = np.where(y == yi)[0][0]
        NBSM = zi
        Z[ix, iy] = chisq(NBSM, N0) # dividing by the standard value, randomly choosing 1st one

    X, Y = np.meshgrid(x, y)

    linthresh_x = np.percentile(np.abs(x[x!=0]),5)
    linthresh_y = np.percentile(np.abs(y[y!=0]),5)
    
    Z = np.where(Z <= 0, np.nan, Z)  # or replace 0 with a small positive number
    # 90% CL contour
    threshold = chi2.ppf(0.90, df=2)  # 4.605



    # only for finding connected regions
    allowed = (Z <= threshold) & np.isfinite(Z)

    labels, num_features = label(allowed)

    ix0 = np.argmin(np.abs(x))
    iy0 = np.argmin(np.abs(y))

    target_label = labels[ix0, iy0]

    if target_label == 0:
        print("origin outside contour")

    allowed_main = (labels == target_label)

    # IMPORTANT:
    # slightly enlarge the region so contour crossing exists
    allowed_main = binary_dilation(allowed_main, iterations=1)

    # use ORIGINAL Z values here
    Z_main = np.where(allowed_main, Z, np.nan)

    # Filled contour (inside = allowed region at 90% CL)
    ax.contourf(X, Y, Z_main.T, levels=[0, threshold], colors=['blue'], alpha=0.1)
    ax.contour( X, Y, Z_main.T, levels=[threshold],    colors=['blue'],
                      linestyles=['-'], linewidths=2)

    # === ADD 1D CONSTRAINT BOX ===
    cx = single_constraints.get((a1,b1), None)
    cy = single_constraints.get((a2,b2), None)
    # Filled 1D constraint box
    if cx is not None and cy is not None:
        from matplotlib.patches import Rectangle
        rect = Rectangle((-cx, -cy), 2*cx, 2*cy,
                         linewidth=3, edgecolor='black',
                         facecolor='none',
                         linestyle='-', zorder=3)
        ax.add_patch(rect)

    # === ADD STAR MARKER ONLY FOR pair_id = 6 ===
    if pair_id == 14:
        mumu_tautau_picked = [1.986926102161816e-60, 4.8546516928066826e-61]

        ax.scatter(
            mumu_tautau_picked[0],
            mumu_tautau_picked[1],
            marker='*',
            s=200,                 # star size
            facecolor='limegreen', # green fill
            edgecolor='darkgreen',     # thin black outline
            linewidth=0.9,
            zorder=5
        )

        # -----------------------------
        # inset zoom
        # -----------------------------
        axins = inset_axes(
            ax,
            width="35%",
            height="35%",
            loc="center"
        )

        # same contour inside inset
        axins.contourf(
            X, Y, Z_main.T,
            levels=[0, threshold],
            colors=['blue'],
            alpha=0.12
        )

        axins.contour(
            X, Y, Z_main.T,
            levels=[threshold],
            colors='blue',
            linewidths=1.2
        )

        # same 1D constraint box inside inset
        if cx is not None and cy is not None:
            rect2 = Rectangle(
                (-cx, -cy), 2*cx, 2*cy,
                linewidth=2,
                edgecolor='black',
                facecolor='none',
                linestyle='-',
                zorder=3
            )
            axins.add_patch(rect2)

        # star inside inset
        axins.scatter(
            mumu_tautau_picked[0],
            mumu_tautau_picked[1],
            marker='*',
            s=350,
            facecolor='limegreen',
            edgecolor='darkgreen',
            linewidth=0.9,
            zorder=5
        )

        # zoom range you requested
        axins.set_xlim(1e-61, 5e-60)
        axins.set_ylim(1e-61, 5e-60)

        axins.set_xscale('log')
        axins.set_yscale('log')

        axins.tick_params(
            axis='both',
            which='major',
            labelsize=11,
            direction='in',
            top=True,
            right=True
        )

        # optional connecting box + lines
        mark_inset(
            ax,
            axins,
            loc1=2,
            loc2=4,
            fc="none",
            ec="gray",
            lw=1
        )




    ax.set_xscale('symlog', linthresh=1e-70, linscale=1.0)   # wider linear window
    ax.set_yscale('symlog', linthresh=1e-70, linscale=1.0)

    ticks = [-1e-50, -1e-55, -1e-60, -1e-65, 0, 1e-65, 1e-60, 1e-55, 1e-50]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    def symlog_fmt(val, pos):
        if val == 0: return '0'
        exp = int(np.round(np.log10(abs(val))))
        return rf'$10^{{{exp}}}$' if val > 0 else rf'$-10^{{{exp}}}$'

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(symlog_fmt))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(symlog_fmt))
    g1 = f"{idx_to_greek[a1]}{idx_to_greek[b1]}"
    g2 = f"{idx_to_greek[a2]}{idx_to_greek[b2]}"
    ax.set_xlim(-5e-59,5e-59)
    ax.set_ylim(-5e-59,5e-59)
    ax.set_xlabel(rf"$\mathring{{\kappa}}^{{(6)}}_{{{g1}}}$  [GeV$^{{-2}}$]", fontsize=20)
    ax.set_ylabel(rf"$\mathring{{\kappa}}^{{(6)}}_{{{g2}}}$  [GeV$^{{-2}}$]", fontsize=20)
    # ax.set_title(f"Pair {pair_id:02d}", fontsize=12)
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.minorticks_off()
    ax.tick_params(axis='both', which='major',
                   direction='in',
                   top=True, right=True, bottom=True, left=True,
                   labelsize=18,
                   length=5, width=0.9)

fig.subplots_adjust(
    left=0.08,
    right=0.97,
    bottom=0.06,
    top=0.98,
    wspace=0.35,   # horizontal separation
    hspace=0.35    # vertical separation
)

plt.savefig("figures/selected_compare_2D_1D.pdf", dpi=200, bbox_inches="tight")
plt.show()
# 
"""

### plot remaining 11 pairs: 2D chi sqr for 11 pairs 
"""
all_pairs = [
    ((0,0),(0,1)), ((0,0),(0,2)), ((0,0),(1,1)), ((0,0),(1,2)), ((0,0),(2,2)),
    ((0,1),(0,2)), ((0,1),(1,1)), ((0,1),(1,2)), ((0,1),(2,2)), ((0,2),(1,1)),
    ((0,2),(1,2)), ((0,2),(2,2)), ((1,1),(1,2)), ((1,1),(2,2)), ((1,2),(2,2)),
]

srctext = 'simprop'

idx_to_greek = {0: "e", 1: r"\mu", 2: r"\tau"}

fig, axes = plt.subplots(4, 3, figsize=(18, 20), constrained_layout=False)

# === BUILD 1D CONSTRAINT LOOKUP ===
with open(f'data/param/fluxSFR_grand200k_dim6.pkl', 'rb') as f:
    LIVparam_1d = pickle.load(f)

single_constraints = {}
for ab_key in LIVparam_1d[6][srctext]:
    ab = ast.literal_eval(ab_key) if isinstance(ab_key, str) else ab_key
    single_constraints[ab] = LIVparam_1d[6][srctext][ab_key]

# by-hand mu-tau, making it blind because it is!
single_constraints[1,2] = 1e-50  # for dim 6

# All 15 pair IDs (1-indexed), excluding 6, 7, 12, 15
excluded = {6,7,12,14}
selected_pair_ids = [i for i in range(1, 16) if i not in excluded]  # 11 pairs

axes_flat = axes.flatten()

for idx_pair, pair_id in enumerate(selected_pair_ids):
    (a1, b1), (a2, b2) = all_pairs[pair_id - 1]
    ax = axes_flat[idx_pair]

    fname = f"data/tau_count/fluxSFR_grand200k_dim6_pair{pair_id:02d}_{a1}{b1}_{a2}{b2}.pkl"
    with open(fname, "rb") as f:
        tau_count_track = pickle.load(f)

    data = tau_count_track[srctext][(a1,b1),(a2,b2)]

    liv = np.array(data["LIV"])
    std = np.array(data["std"])
    x = np.unique(liv[:,0])
    y = np.unique(liv[:,1])

    nx, ny = len(x), len(y)
    Z = np.full((nx, ny), np.nan)
    N0 = std[0,2]

    for xi, yi, zi in liv:
        ix = np.where(x == xi)[0][0]
        iy = np.where(y == yi)[0][0]
        NBSM = zi
        Z[ix, iy] = chisq(NBSM, N0)

    X, Y = np.meshgrid(x, y)

    linthresh_x = np.percentile(np.abs(x[x!=0]), 5)
    linthresh_y = np.percentile(np.abs(y[y!=0]), 5)

    Z = np.where(Z <= 0, np.nan, Z)

    threshold = chi2.ppf(0.90, df=2)
    ax.contourf(X, Y, Z.T, levels=[0, threshold], colors=['blue'], alpha=0.1)
    ax.contour(X, Y, Z.T, levels=[threshold], colors=['blue'],
               linestyles=['-'], linewidths=2)

    # === ADD 1D CONSTRAINT BOX ===
    cx = single_constraints.get((a1,b1), None)
    cy = single_constraints.get((a2,b2), None)
    if cx is not None and cy is not None:
        from matplotlib.patches import Rectangle
        rect = Rectangle((-cx, -cy), 2*cx, 2*cy,
                         linewidth=3, edgecolor='black',
                         facecolor='none',
                         linestyle='-', zorder=3)
        ax.add_patch(rect)

    # ax.set_xscale('symlog', linthresh=linthresh_x, linscale=1.0)
    # ax.set_yscale('symlog', linthresh=linthresh_y, linscale=1.0)
    ax.set_xscale('symlog', linthresh=1e-70, linscale=1.0)
    ax.set_yscale('symlog', linthresh=1e-70, linscale=1.0)

    ticks = [-1e-60, -1e-65, 0, 1e-65, 1e-60]# [-1e-50, -1e-55,-1e-60,-1e-65, 0, 1e-65, 1e-60, 1e-55, 1e-50]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    def symlog_fmt(val, pos):
        if val == 0: return '0'
        exp = int(np.round(np.log10(abs(val))))
        return rf'$10^{{{exp}}}$' if val > 0 else rf'$-10^{{{exp}}}$'

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(symlog_fmt))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(symlog_fmt))

    g1 = f"{idx_to_greek[a1]}{idx_to_greek[b1]}"
    g2 = f"{idx_to_greek[a2]}{idx_to_greek[b2]}"
    ax.set_xlim(-1e-56, 1e-56)
    ax.set_ylim(-1e-56, 1e-56)
    ax.set_xlabel(rf"$\mathring{{\kappa}}^{{(6)}}_{{{g1}}}$  [GeV$^{{-2}}$]", fontsize=20)
    ax.set_ylabel(rf"$\mathring{{\kappa}}^{{(6)}}_{{{g2}}}$  [GeV$^{{-2}}$]", fontsize=20)
    # ax.minorticks_on()
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.minorticks_off()
    ax.tick_params(axis='both', which='major',
                   direction='in',
                   top=True, right=True, bottom=True, left=True,
                   labelsize=18,
                   length=5, width=0.9)

# Hide the unused 12th subplot (4x3 = 12 slots, only 11 used)
axes_flat[11].set_visible(False)

fig.subplots_adjust(
    left=0.08,
    right=0.97,
    bottom=0.06,
    top=0.98,
    wspace=0.35,   # horizontal separation
    hspace=0.35    # vertical separation
)
# plt.savefig("figures/remaining_2D_1D_11pairs.pdf", dpi=200, bbox_inches="tight")
plt.show()
"""

### blindspot count histogram
"""
fluxtype='SFR'
exp='grand200k'
d=6

results={}
param1, param2 = 11, 22
infile = os.path.join('data', 'blindspot', f'blindspot_{fluxtype}_{exp}_dim{d}_{param1}_{param2}.pkl')
with open(infile, 'rb') as f:
    results = pickle.load(f)

# infile_v2 = os.path.join('data', 'blindspot', f'blindspot_{fluxtype}_{exp}_dim{d}_v2.pkl')
# with open(infile_v2, 'rb') as f:
#     results.update(pickle.load(f))

# ══════════════════════════════════════════════════════════════════
# CASE METADATA  (labels & LIV values for the legend)
# ══════════════════════════════════════════════════════════════════

## emu, etau
emu_etau_picked = [5.861599453065788e-61, 4.155947048987179e-60]

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

 
case_meta = {
    # 'case1_01': {
    #     'label': r'BP1: $\mathring{{\kappa}}^{{(6)}}_{{{e\mu}}}$', 
    # },
    # 'case2_02': {
    #     'label': r'BP2: $\mathring{{\kappa}}^{{(6)}}_{{{e\tau}}}$', 
    # },
    # 'case3_01_02': {
    #     'label': r'BP3: $\mathring{{\kappa}}^{{(6)}}_{{{e\mu}}}$ + $\mathring{{\kappa}}^{{(6)}}_{{{e\tau}}}$', 
    # },
    'case1_11': {
        'label': r'BP1: $\mathring{{\kappa}}^{{(6)}}_{{{\mu\mu}}}$', 
    },
    'case2_22': {
        'label': r'BP2: $\mathring{{\kappa}}^{{(6)}}_{{{\tau\tau}}}$', 
    },
    'case3_11_22': {
        'label': r'BP3: $\mathring{{\kappa}}^{{(6)}}_{{{\mu\mu}}}$, $\mathring{{\kappa}}^{{(6)}}_{{{\tau\tau}}}$', 
    },
    'std':{
        'label':r'std',
    },
}
 
# ══════════════════════════════════════════════════════════════════
# COLOR-BLIND FRIENDLY PALETTE  (Wong 2011 / IBM)
# ══════════════════════════════════════════════════════════════════
COLORS = {
    'std':         'darkgray', 
    'case1_01':    'red',
    'case2_02':    'orange',
    'case3_01_02': 'darkgreen',
    'case1_11':    'red',
    'case2_22':    'orange',
    'case3_11_22': 'darkgreen',
}

LINESTYLES = {
    'std':          '-',         # solid
    'case1_01':     '--',        # dashed
    'case2_02':     (0, (3, 1, 1, 1)),   # dash-dot
    'case3_01_02':  (0, (1, 1)),         # dotted
    'case1_11':     '--',        # dashed
    'case2_22':     (0, (3, 1, 1, 1)),   # dash-dot
    'case3_11_22':  (0, (1, 1)),         # dotted
}


# ══════════════════════════════════════════════════════════════════
# EXTRACT DATA
# ══════════════════════════════════════════════════════════════════
# energy bin centres (log10 scale)
energy_edges = np.arange(16.25, 20.01, 0.25) - 9   # eV to GeV       # 20 bins
n_intervals = len(energy_edges)-1
bin_centres  = 0.5 * (energy_edges[:-1] + energy_edges[1:])
bin_width    = energy_edges[1] - energy_edges[0]    # in log10(E/eV)
 
# ══════════════════════════════════════════════════════════════════
# FIGURE  — 3 subplots stacked vertically, sharing x-axis
# ══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
 
for case_name, meta in case_meta.items():
    color = COLORS[case_name]
    counts = np.array([
        results[case_name][i]['count'] for i in range(n_intervals)
    ])
    print(case_name, counts)
    ls = LINESTYLES[case_name]
    rgba_fill = mcolors.to_rgba(color, alpha=0.16)

    # # Fill only — no border at all
    if case_name=='std':
        ax.bar(
            energy_edges[:-1],
            counts,
            width=bin_width,
            align='edge',
            color=rgba_fill,
            edgecolor='none',
            alpha=0.3,
            linewidth=0,
            # label=meta['label'],
        )

    # Outline: step draws only horizontal tops + left/right outer walls
    # np.append closes the last bin edge
    ax.step(
        energy_edges,
        np.append(counts, counts[-1]),  # repeat last value to close right edge
        where='post',
        color=color,
        linewidth=3,
        linestyle=ls,
        label=meta['label']
    )


ax.set_xlabel(r'$\log_{10}(E\,/\,\mathrm{GeV})$', fontsize=17)
ax.set_ylabel('Tau neutrino events', fontsize=17)
ax.set_xlim(energy_edges[1], energy_edges[-1])                          # starts at second edge to match 16.5 cutoff
ax.set_ylim(0, 125.0)
energy_xticks = np.arange(energy_edges[1], energy_edges[-1]+0.01, 0.5)
ax.set_xticks(energy_xticks)
ax.set_xticklabels([f'{e:.1f}' for e in energy_xticks])
ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, left=True, bottom=True, labelsize=14)

## swatchlegend
# handles = [
#     mpatches.Patch(facecolor=COLORS[case_name], edgecolor='none', label=meta['label'])
#     for case_name, meta in case_meta.items()
# ]
# ax.legend(handles=handles, fontsize=12, loc='upper left', frameon=False)

ax.legend(fontsize=14.5, loc='upper left', ncol=1, frameon=False)

fig.savefig(f'figures/blindspot_hist_{fluxtype}_{exp}_dim{d}_{param1}_{param2}.pdf', dpi=150, bbox_inches='tight')
plt.show()

"""



    