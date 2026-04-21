import numpy as np
from sympy import sin, cos 
import sympy as sp
import scipy as scp
import pandas as pd
from cmath import *
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
from matplotlib.colors import TwoSlopeNorm
from matplotlib.colors import LogNorm

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

t0 = timeit.default_timer()

from LIV_fraction import * 
from LIV_tau_count import * 

exp=input("Enter experiment (poemma/ICgen2radio/grand200k): ")

# flavlabel = [r'$e$',r'$\mu$',r'$\tau$']
flavlabel = [r"e", r"\mu", r"\tau"]

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

uniform_src=False
src_ratio = [1.0,2.0,0.0] # [1.0,0.0,0.0] [0.0,1.0,0.0] [1.0,2.0,0.0]
# fixed_srctext = '1.00.00.0'#'1.00.00.0'#'0.01.00.0'#'1.02.00.0'
flux_list = ['pess','mod','opt']
flux_label = ['no evolution', 'SFR evolution', 'AGN evolution']
# line_flux = ['-','--',':']
flux_color = ["#0072B2",  # blue
          "#D55E00",  # vermillion (orange-red)
          "#009E73"]  # bluish green
line_flux = ['--',(0,(1,0.5)),'-']#(0,(1,2))]
drange = np.arange(3,9)
cmap = plt.get_cmap("tab10")  # very distinguishable

### plot count vs LIV for varying dim d
"""
for d in [5]:#drange:
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    # to simplify making the legend for std
    axes[0].hlines(
        10000000, 
        10,   # left end
        20,   # right end
        color='black',
        linewidth=2,
        linestyle=':',
        label='std'
    )

    # to simplify making the legend for flux types
    for iflux, flux_var in enumerate(flux_list):
        axes[0].hlines(
            10000000*iflux, 
            10,   # left end
            20,   # right end
            color=flux_color[iflux],#'black',
            linewidth=2,
            linestyle=line_flux[iflux], 
            label=flux_label[iflux]
        )    


    std_N0_list = []
    # for iflux, flux_var in enumerate([flux_list[0]]):
    for iflux, flux_var in enumerate(flux_list):
     
        if uniform_src:
            with open(f'data/tau_count/uniformx/flux{flux_var}_{exp}_dim{d}.pkl', 'rb') as f:
                tau_count_track = pickle.load(f)
            
            srctext = 'uniformx'

        else:
            with open(f'data/tau_count/flux{flux_var}_{exp}_dim{d}.pkl', 'rb') as f:
                tau_count_track = pickle.load(f)

            srctext = ''.join(f"{x}" for x in src_ratio)

        ab_list = list(tau_count_track[srctext].keys())
        for iax, (alpha, beta) in enumerate(ab_list):
            ax = axes[iax]

            data = tau_count_track[srctext][(alpha,beta)]
            tau_vals = data['LIV']   # shape (N,2)
            c_vals = tau_vals[:,0]
            NBSM_vals = tau_vals[:,1]

            ax.plot(
                c_vals, NBSM_vals,
                linestyle=line_flux[iflux],
                color=flux_color[iflux],
                linewidth=2
            )

            # horizontal N0 line
            N0 = data['std'][:,1][0]
            print(d,flux_var,N0)
            # N0 = tau_count(exp, src_ratio=src_ratio, fluxtype=flux_var)
            # print(d,flux_var,N0)
            ax.plot(
                c_vals, data['std'][:,1],
                linestyle=':',
                color=flux_color[iflux],
                linewidth=2
            )   

            std_N0_list.append(N0)

            ax.set_xscale('log')
            ax.set_xlim(min(c_vals), max(c_vals))
            # last flux is opt, so opt-> is set to range
            # print(N0)
            # ax.set_yscale('log')
            # ax.set_ylim(1e-1, N0*100)
            ax.set_ylim(0, N0*1.4)
            # ax.set_ylim(min(NBSM_vals)-1, max(NBSM_vals)+1)
            # print(min(NBSM_vals)-1, max(NBSM_vals)+1)
            # ax.set_title(rf"$\alpha={alpha},\ \beta={beta}$")
            # ax.set_xlabel(rf"$\kappa_{{{flavlabel[alpha]}{flavlabel[beta]}}}\,[\mathrm{{GeV}}^{{{4-d}}}]$", fontsize=16)
            # ax.set_xlabel(rf"$\kappa_{{{flavlabel[alpha]}{flavlabel[beta]}}}\,[\mathrm{{GeV}}^{{4-d}}]$" ,fontsize=16)
            # ax.set_xlabel(rf"${alpha},{beta}$", fontsize=16)
            # ax.set_ylabel(r"$N_{\nu_\tau}(\mathring{\kappa}^{(d)}_{\alpha\beta})$", fontsize=16)
            ax.tick_params(top=True, bottom=True, right=True,
                            direction="in", which="both")
            ax.tick_params(axis='both', which='major', labelsize=12)  # major ticks
            ax.tick_params(axis='both', which='minor', labelsize=8)  # minor ticks
            ax.minorticks_on()

    for iax, (alpha, beta) in enumerate(ab_list):
        ax = axes[iax]
        ax.text(min(c_vals)*4e2, N0*1.22, rf"$\mathring{{\kappa}}^{{({d})}}_{{{flavlabel[alpha]}{flavlabel[beta]}}}$", fontsize=20)
    # legend only once
    axes[0].legend(loc="center left", frameon=False, fontsize=16)
    fig.supxlabel(rf"$\mathring{{\kappa}}^{{({d})}}_{{\alpha\beta}}\,[\mathrm{{GeV}}^{{{4-d}}}]$", fontsize=16)
    fig.supylabel(r"$N_{\nu_\tau}(\mathring{\kappa}^{(d)}_{\alpha\beta})$", fontsize=16)    
    
    total_ratio = int(sum(src_ratio))
    s = ",".join(
        f"{int(v)}/{total_ratio}" if (v and total_ratio != 1) 
        else str(int(v))
        for v in src_ratio
        )

    label = rf"$f_S=({s})$"
    axes[0].text(min(c_vals)*4e1, 1500, label, fontsize=20)

    plt.tight_layout()
    # plt.savefig(f"figures/tau_count_{exp}_src{srctext}_dim{d}.pdf", dpi=300)
    plt.show()
    # print(f"time processed:{np.round(timeit.default_timer(),2)} s")
"""

### debugging plot count for dim d
"""
for d in [5]: #drange:
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    # to simplify making the legend for std
    axes[0].hlines(
        10000000, 
        10,   # left end
        20,   # right end
        color='black',
        linewidth=2,
        linestyle=':',
        label='std'
    )

    # to simplify making the legend for flux types
    for iflux, flux_var in enumerate(flux_list):
        axes[0].hlines(
            10000000*iflux, 
            10,   # left end
            20,   # right end
            color='black',
            linewidth=2,
            linestyle=line_flux[iflux], 
            label=flux_label[iflux]
        )    


    for iflux, flux_var in enumerate(flux_list):
     
        if uniform_src:
            with open(f'data/tau_count/uniformx/flux{flux_var}_{exp}_dim{d}.pkl', 'rb') as f:
                tau_count_track = pickle.load(f)
            
            srctext = 'uniformx'

        else:
            with open(f'data/tau_count/flux{flux_var}_{exp}_dim{d}.pkl', 'rb') as f:
                tau_count_track = pickle.load(f)

            srctext = ''.join(f"{x}" for x in src_ratio)

        ab_list = list(tau_count_track[srctext].keys())
        for iax, (alpha, beta) in enumerate(ab_list):
            ax = axes[iax]

            data = tau_count_track[srctext][(alpha,beta)]
            tau_vals = data['LIV']   # shape (N,2)
            c_vals = tau_vals[:,0]
            NBSM_vals = tau_vals[:,1]

            ax.plot(
                c_vals, NBSM_vals,
                linestyle=line_flux[iflux],
                color=flux_color[iflux],
                linewidth=2
            )

            NBSM_function = interp1d(c_vals, NBSM_vals, 
                             kind='linear', fill_value="extrapolate")

            ctemp = np.logspace(-120, -60, 1000)
            ax.plot(
                ctemp, NBSM_function(ctemp),
                linestyle=':',
                color='k', #flux_color[iflux],
                linewidth=2
            )

            # horizontal N0 line
            N0 = data['std'][:,1][0]
            # N0 = tau_count(exp, src_ratio=src_ratio, fluxtype=flux_var)
            ax.plot(
                c_vals, data['std'][:,1],
                linestyle=':',
                color=flux_color[iflux],
                linewidth=2
            )   

            ax.set_xscale('log')
            ax.set_xlim(min(c_vals), max(c_vals))
            # last flux is opt, so opt-> is set to range
            # print(N0)
            ax.set_ylim(N0*0.95, 4180)
            ax.axhline(y=4054.5, linestyle='--')
            # ax.set_ylim(0, N0*1.2)
            # ax.set_ylim(min(NBSM_vals)-1, max(NBSM_vals)+1)
            # print(min(NBSM_vals)-1, max(NBSM_vals)+1)
            # ax.set_yscale('log')
            # ax.set_title(rf"$\alpha={alpha},\ \beta={beta}$")
            ax.set_xlabel(rf"$\kappa_{{{flavlabel[alpha]}{flavlabel[beta]}}}\,[\mathrm{{GeV}}^{{4-d}}]$" ,fontsize=16)
            # ax.set_xlabel(rf"${alpha},{beta}$", fontsize=16)
            ax.set_ylabel(r"$N_{\nu_\tau}(\mathring{\kappa}^{(d)}_{\alpha\beta})$", fontsize=16)
            ax.tick_params(top=True, bottom=True, right=True,
                            direction="in", which="both")
            ax.tick_params(axis='both', which='major', labelsize=12)  # major ticks
            ax.tick_params(axis='both', which='minor', labelsize=8)  # minor ticks
            ax.minorticks_on()

    # legend only once
    axes[0].legend(frameon=False, fontsize=16)

    plt.suptitle(f"d = {d}", fontsize=16)

    plt.tight_layout()
    # plt.savefig(f"figures/tau_count_{exp}_src{srctext}_dim{d}_debugg.pdf", dpi=300)
    plt.show()
    print(f"time processed:{np.round(timeit.default_timer(),2)} s")
"""

### plot count vs fixed LIV alpha beta
"""
uniform_src=False # True False
fixed_iax=3 #0 to 5
# fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
fig, axes = plt.subplots(1, 1, figsize=(10, 8), sharex=True, sharey=True)
# axes = axes.flatten()

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

# # to simplify making the legend for flux types
# for iflux, flux_var in enumerate(flux_list):
#     axes.hlines(
#         10000000*iflux, 
#         10,   # left end
#         20,   # right end
#         color='black',
#         linewidth=2,
#         linestyle=line_flux[iflux], 
#         label=flux_label[iflux]
#     )    

cmin=10
cmax=0
for d in drange:
    base_color = cmap((d-3)/5)  # normalize d in [0,1]
    # for iflux, flux_var in enumerate(flux_list):
    iflux, flux_var = 2, 'opt'
    if uniform_src:
        with open(f'data/tau_count/uniformx/flux{flux_var}_{exp}_dim{d}.pkl', 'rb') as f:
            tau_count_track = pickle.load(f)
        
        srctext = 'uniformx'

    else:
        with open(f'data/tau_count/flux{flux_var}_{exp}_dim{d}.pkl', 'rb') as f:
            tau_count_track = pickle.load(f)

        srctext = ''.join(f"{x}" for x in src_ratio)

    ab_list = list(tau_count_track[srctext].keys())
    for iax, (alpha, beta) in enumerate(ab_list):
        if iax!=fixed_iax:
            continue
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
        c_ext = np.logspace(-150, -30, 500)
        logc_ext = np.log10(c_ext)

        logN_ext = f(logc_ext)
        # for values outside c grid, keep constant
        logN_ext[logc_ext < logc.min()] = logN[logc.argmin()]
        logN_ext[logc_ext > logc.max()] = logN[logc.argmax()]

        NBSM_ext = 10**logN_ext

        # plot extrapolated curve
        ax.plot(
            c_ext, NBSM_ext,
            linestyle='--',#line_flux[iflux],
            color=base_color,
            linewidth=2,
            label=f'd={d}'
        )

        ax.set_xlabel(rf"$\kappa_{{{flavlabel[alpha]}{flavlabel[beta]}}}\,[\mathrm{{GeV}}^{{4-d}}]$" ,fontsize=16)
    
    # horizontal N0 line outside d loop so it's done only once
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
ax.set_xlim(min(c_ext), max(c_ext))
# last flux is opt, so opt-> is set to range
# print(N0)
# ax.set_ylim(N0*0.95, 4180)

code = "".join(map(str, map(int, src_ratio)))

# ylim for (fixed_iax, src_ratio)
ylim_map = {
    "poemma": {
        (2,"010"):(0,100),
        (3,"120"):(0,85),
        (4,"010"):(80,125),
    },
    "grand200k": {
        (1,"120"):(0,5000),
        (4,"010"):(4400,6400),
    }
}

ax.set_ylim(*ylim_map[exp].get((fixed_iax, code),
           (0,100) if exp=="poemma" else (0,5000)))

# ax.set_ylim(min(NBSM_vals)-1, max(NBSM_vals)+1)
# print(min(NBSM_vals)-1, max(NBSM_vals)+1)
# ax.set_yscale('log')
ax.set_ylabel(r"$N_{\nu_\tau}(\mathring{\kappa}^{(d)}_{\alpha\beta})$", fontsize=16)
ax.tick_params(top=True, bottom=True, right=True,
                direction="in", which="both")
ax.tick_params(axis='both', which='major', labelsize=12)  # major ticks
ax.tick_params(axis='both', which='minor', labelsize=8)  # minor ticks
ax.minorticks_on()

# legend only once
ax.legend(frameon=False, fontsize=16,
    ncol=2,
    loc='upper left', 
    bbox_to_anchor=(0.0, 1.0))

if uniform_src:
    label = r"$f_S=(x,1-x,0)$"
else:
    s = ",".join(f"{int(v)}/{int(sum(src_ratio))}" if v else "0"
                 for v in src_ratio)
    label = rf"$f_S=({s})$"

ax.text(0.97, 0.97, label, transform=ax.transAxes,
        ha="right", va="top", fontsize=16)
# plt.suptitle(f"d = {d}", fontsize=16)

plt.tight_layout()
# plt.savefig(f"figures/tau_count_LIV{fixed_iax}_{exp}_src{srctext}.pdf", dpi=300)
plt.show()
print(f"time processed:{np.round(timeit.default_timer(),2)} s")
"""


### plot count vs LIV fixed dim d 
"""
uniform_src=True # True False
# fixed_iax=2 #0 to 5
# fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
fig, axes = plt.subplots(1, 1, figsize=(10, 8), sharex=True, sharey=True)
# axes = axes.flatten()

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

# # to simplify making the legend for flux types
# for iflux, flux_var in enumerate(flux_list):
#     axes.hlines(
#         10000000*iflux, 
#         10,   # left end
#         20,   # right end
#         color='black',
#         linewidth=2,
#         linestyle=line_flux[iflux], 
#         label=flux_label[iflux]
#     )    

cmin=10
cmax=0
for d in [6]:#drange:
    colors = plt.cm.Greys(np.linspace(0.28, 0.95, 6))
    linestyles = ["-", "--", "-.", (0,(1,1)), (0,(3,1,1,1)), (0,(5,2))]

    base_color = cmap((6-3)/5)  # normalize d in [0,1]
    # for iflux, flux_var in enumerate(flux_list):
    iflux, flux_var = 2, 'opt'
    if uniform_src:
        with open(f'data/tau_count/uniformx/flux{flux_var}_{exp}_dim{d}.pkl', 'rb') as f:
            tau_count_track = pickle.load(f)
        
        srctext = 'uniformx'

    else:
        with open(f'data/tau_count/flux{flux_var}_{exp}_dim{d}.pkl', 'rb') as f:
            tau_count_track = pickle.load(f)

        srctext = ''.join(f"{x}" for x in src_ratio)

    ab_list = list(tau_count_track[srctext].keys())
    for iax, (alpha, beta) in enumerate(ab_list):
        # if iax!=fixed_iax:
        #     continue
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
ax.set_xlim(min(c_ext), max(c_ext))
# last flux is opt, so opt-> is set to range
# print(N0)
# ax.set_ylim(N0*0.95, 4180)
if exp=='poemma' and uniform_src==True:
    ax.set_ylim(0, 83) # poemma 
elif exp=='poemma' and uniform_src!=True:
    ax.set_ylim(0, 120) # poemma 
elif exp=='grand200k' and uniform_src==True:
    ax.set_ylim(0, 4300) #grand200k
elif exp=='grand200k' and uniform_src!=True:
    ax.set_ylim(0, 6800) #grand200k
# ax.set_ylim(min(NBSM_vals)-1, max(NBSM_vals)+1)
# print(min(NBSM_vals)-1, max(NBSM_vals)+1)
# ax.set_yscale('log')

ax.set_ylabel(r"$N_{\nu_\tau}(\mathring{\kappa}^{(d)}_{\alpha\beta})$", fontsize=30)
ax.tick_params(top=True, bottom=True, right=True,
                direction="in", which="both")
ax.tick_params(axis='both', which='major', labelsize=24)  # major ticks
ax.tick_params(axis='both', which='minor', labelsize=16)  # minor ticks
ax.minorticks_on()

# legend only once
ax.legend(frameon=False, fontsize=30,
    ncol=2,
    loc='lower left', )
    # bbox_to_anchor=(0.0, 0.0))
if uniform_src:
    label = r"$f_S=(x,1-x,0)$"
else:
    s = ",".join(f"{int(v)}/{int(sum(src_ratio))}" if v else "0"
                 for v in src_ratio)
    label = rf"$f_S=({s})$"

ax.text(min(c_ext)*1e1, 4300*0.80, label, fontsize=30)
# plt.suptitle(f"d = {d}", fontsize=16)

plt.tight_layout()
# plt.savefig(f"figures/tau_count_d{d}_{exp}_src{srctext}.pdf", dpi=300)
plt.show()
print(f"time processed:{np.round(timeit.default_timer(),2)} s")
"""

### plot chi2 vs LIV fixed dim d uniform
"""
uniform_src=True # True False
# fixed_iax=2 #0 to 5
# fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
fig, axes = plt.subplots(1, 1, figsize=(10, 8), sharex=True, sharey=True)
# axes = axes.flatten()
ax = axes

# to simplify making the legend for std
axes.hlines(
    10000000, 
    10,   # left end
    20,   # right end
    color='black',
    linewidth=2,
    linestyle=':',
    label=f'$\chi^2 _{{0.05}}$\n(df=1)'
)

cmin=10
cmax=0
for d in [6]:#drange:
    colors = plt.cm.Greys(np.linspace(0.28, 0.95, 6))
    linestyles = ["-", "--", "-.", (0,(1,1)), (0,(3,1,1,1)), (0,(5,2))]

    base_color = cmap((6-3)/5)  # normalize d in [0,1]
    # for iflux, flux_var in enumerate(flux_list):
    iflux, flux_var = 2, 'opt'
    if not uniform_src:
        print('not uniform')
        break
        
    srctext = 'uniformx'

    LIVparam = {d:{}}

    with open(f'data/tau_count/uniformx/flux{fluxtype}_{exp}_dim{d}.pkl', 'rb') as f:
        totNBSM_dict = pickle.load(f)

    LIVparam[d]['uniformx'] = {}

    for iax, (alpha,beta) in enumerate(totNBSM_dict['uniformx'].keys()):

        data = totNBSM_dict['uniformx'][alpha,beta]
        tau_vals = data['LIV']   # shape (N,2)
        c_vals = tau_vals[:,0]
        NBSM_vals = tau_vals[:,1]
        N0 = data['std'][:,1][0]
        # build interpolator on filtered data
        NBSM_function = interp1d(c_vals, NBSM_vals, 
                             kind='linear', fill_value="extrapolate")

        target = chi2.ppf(0.95, 1)

        c_ext = np.logspace(np.log10(c_vals.min()), np.log10(c_vals.max()), 100)
        chisq_vals = np.array([chisq(float(NBSM_function(c)), N0) for c in c_ext])

        ax.plot(
            c_ext, chisq_vals,
            linestyle=linestyles[iax],
            color=colors[iax],
            linewidth=2,
            label=rf"$\mathring{{\kappa}}^{{({d})}}_{{{flavlabel[alpha]}{flavlabel[beta]}}}$")

ax.set_xlabel(rf"$\mathring{{\kappa}}^{{({d})}}_{{\alpha\beta}}\,[\mathrm{{GeV}}^{{{4-d}}}]$", fontsize=30)

# horizontal N0 line 
# N0 = data['std'][:,1][0]
# N0 = tau_count(exp, src_ratio=src_ratio, fluxtype=flux_var)
ax.plot(
    c_ext, target*np.ones(len(c_ext)),
    linestyle=':',
    color='k',#flux_color[iflux],
    linewidth=2,
)   

ax.set_xscale('log')
# ax.set_xlim(cmin, cmax)
ax.set_xlim(min(c_ext), max(c_ext))
# last flux is opt, so opt-> is set to range
# print(N0)
# ax.set_ylim(N0*0.95, 4180)
# if exp=='poemma' and uniform_src==True:
#     ax.set_ylim(0, 83) # poemma 
# elif exp=='poemma' and uniform_src!=True:
#     ax.set_ylim(0, 120) # poemma 
# elif exp=='grand200k' and uniform_src==True:
#     ax.set_ylim(0, 4300) #grand200k
# elif exp=='grand200k' and uniform_src!=True:
#     ax.set_ylim(0, 6800) #grand200k
# ax.set_ylim(min(NBSM_vals)-1, max(NBSM_vals)+1)
# print(min(NBSM_vals)-1, max(NBSM_vals)+1)
ax.set_yscale('log')
ax.set_ylim(1e-10, 1e6)
ax.set_ylabel(r"$\chi^2(\mathring{\kappa}^{(d)}_{\alpha\beta})$", fontsize=30)
ax.tick_params(top=True, bottom=True, right=True,
                direction="in", which="both")
ax.tick_params(axis='both', which='major', labelsize=24)  # major ticks
ax.tick_params(axis='both', which='minor', labelsize=16)  # minor ticks
ax.minorticks_on()

# legend only once
ax.legend(frameon=False, fontsize=30,
    ncol=2,
    loc='lower right', )
    # bbox_to_anchor=(0.0, 0.0))
if uniform_src:
    label = r"$f_S=(x,1-x,0)$"
else:
    s = ",".join(f"{int(v)}/{int(sum(src_ratio))}" if v else "0"
                 for v in src_ratio)
    label = rf"$$" + "\n" + rf"$f_S=({s})$"

# take y lim log difference and take 0.75 to be consistent with N vs LIV plot
ax.text(min(c_ext)*1e1, 10**((6-(-10))*0.75-9), label, fontsize=30) 
# plt.suptitle(f"d = {d}", fontsize=16)

plt.tight_layout()
# plt.savefig(f"figures/chi2_d{d}_{exp}_src{srctext}.pdf", dpi=300)
plt.show()
print(f"time processed:{np.round(timeit.default_timer(),2)} s")
"""


### plot tau fraction vs E for fixed d, fixed src varying alpha beta
"""
d=6

dm21, dm31 = np.array([7.49e-5, 2.534e-3])
theta12, theta13, theta23 =  [np.arcsin(np.sqrt(0.303)), 
                              np.arcsin(np.sqrt(0.02225)), 
                              np.arcsin(np.sqrt(0.572))]

delta = 197*np.pi/180

for src_ratio in [[0,1,0],[1,2,0]]:
    flavtrack = {}
    srctext = ''.join(f"{x}" for x in src_ratio)

    for alpha in range(3):
        for beta in range(alpha,3):
            if beta==alpha:
                continue
            LIVmatrix = np.zeros((3, 3), dtype=complex)
            if d==4:
                LIVmatrix[alpha, beta] = 1e-36  # GeV^{}
            elif d==6:
                LIVmatrix[alpha, beta] = 1e-54  # GeV^{-2}
            flavtracktemp = []

            Erange = np.concatenate(
                [np.logspace(4,np.log10(2e7),40,endpoint=False), 
                 np.logspace(np.log10(2e7),np.log10(9e8),80,endpoint=False), 
                 np.logspace(np.log10(9e8),11,40)])

            for E in Erange:
                flavtracktemp.append([E,
                                  flavor_fraction(E*1e9, dm21, dm31, 
                                                  theta12, theta23, theta13, delta,
                                                  d=d, 
                                                  a_eff=LIVmatrix, 
                                                  c_eff=LIVmatrix,
                                                  src_ratio=src_ratio, flavor='tau')] ) 
            flavtrack[alpha,beta] = flavtracktemp
            if alpha==1 and beta==2:
                print(flavtrack[alpha,beta])

    plt.figure()
    flavlabel = [r'$e$',r'$\mu$',r'$\tau$']
    for (alpha, beta), en_taufrac in flavtrack.items():
        en_taufrac = np.array(en_taufrac)
        en = en_taufrac[:,0]
        taufrac = en_taufrac[:,1]
        plt.plot(en, taufrac, label=f"{flavlabel[alpha]}{flavlabel[beta]}")

    plt.xscale("log")   # since your c is like 1e-75 ...
    plt.xlabel("neutrino energy, E [GeV]", fontsize=16)
    plt.ylabel(r"tau neutrino fraction, $f_{\tau,\oplus}$", fontsize=16)
    if d==4:
        plt.title(fr"$f_S={tuple(src_ratio)}$, $d={d}$, $|\kappa_{{\alpha\beta}}|=10^{{-36}}\ $", fontsize=16)
    elif d==6:
        plt.title(fr"$f_S={tuple(src_ratio)}$, $d={d}$, $|\kappa_{{\alpha\beta}}|=10^{{-55}}\ \mathrm{{GeV}}^{{-2}}$", fontsize=16)
    plt.legend(frameon=False, ncol=2, loc="lower left", fontsize=16)
    # plt.xlim(Erange[0],Erange[-1])
    plt.xlim(1e5,1e10)
#     plt.grid(True)
    # shaded regions
    plt.axvspan(1e4, 1e7, color='gray', alpha=0.1)   # first shade
    plt.axvspan(1e7, 1e11, color='gray', alpha=0.35)  # second shade

    plt.tick_params(top=True, bottom=True, left=True, right=True,
                    which='both', direction='in')
    plt.tick_params(axis='both', which='major', labelsize=12)  # major ticks
    plt.tick_params(axis='both', which='minor', labelsize=8)  # minor ticks

    plt.tight_layout()
    # plt.savefig(f'figures/tau_frac_E_dim{d}_src{srctext}.pdf',dpi=300)
    plt.show()
"""

### plot tau fraction vs E for fixed d, fixed src diag + off-diag
# """
d=6# 4 5 6
redshifted=True

dm21, dm31 = np.array([7.49e-5, 2.534e-3])
theta12, theta13, theta23 =  [np.arcsin(np.sqrt(0.303)), 
                              np.arcsin(np.sqrt(0.02225)), 
                              np.arcsin(np.sqrt(0.572))]

delta = 197*np.pi/180

for src_ratio in [[1,2,0],[0,1,0]]:
    flavtrack = {}
    srctext = ''.join(f"{x}" for x in src_ratio)

    for alpha in range(3):
        for beta in range(alpha,3):
            LIVmatrix = np.zeros((3, 3), dtype=complex)
            if d==4:
                LIVmatrix[alpha, beta] = 1e-36  # GeV^{}
            elif d==5:
                LIVmatrix[alpha, beta] = 1e-45  # GeV^{-1}
            elif d==6:
                LIVmatrix[alpha, beta] = 1e-54  # GeV^{-2}
            flavtracktemp = []

            Erange = np.concatenate(
                [np.logspace(4,np.log10(2e7),40,endpoint=False), 
                 np.logspace(np.log10(2e7),np.log10(9e8),80,endpoint=False), 
                 np.logspace(np.log10(9e8),11,40)])

            for E in Erange:
                if redshifted==True:
                    redz = 1
                    flavtracktemp.append([E,
                                      flavor_fraction_redshifted(E*1e9, redz, dm21, dm31, 
                                                      theta12, theta23, theta13, delta,
                                                      d=d, 
                                                      a_eff=LIVmatrix, 
                                                      c_eff=LIVmatrix,
                                                      src_ratio=src_ratio, flavor='tau')] ) 

                else:
                    flavtracktemp.append([E,
                                      flavor_fraction(E*1e9, dm21, dm31, 
                                                      theta12, theta23, theta13, delta,
                                                      d=d, 
                                                      a_eff=LIVmatrix, 
                                                      c_eff=LIVmatrix,
                                                      src_ratio=src_ratio, flavor='tau')] ) 

            flavtrack[alpha,beta] = flavtracktemp

    flavlabel = [r'$e$', r'$\mu$', r'$\tau$']

    diag_colors = ["#0072B2", "#D55E00", "#009E73"]   # blue, vermillion, green
    offdiag_colors = ["#CC79A7", "#E69F00", "#000000"] # purple, orange, sky blue
    
    for mode in ["diag", "offdiag"]:
        plt.figure()
        colors = diag_colors if mode == "diag" else offdiag_colors
        ic = 0
        for (alpha, beta), en_taufrac in flavtrack.items():
            # select which elements to plot
            if mode == "diag" and alpha != beta:
                continue
            elif mode == "offdiag" and alpha == beta:
                continue

            en_taufrac = np.array(en_taufrac)
            en = en_taufrac[:,0]
            taufrac = en_taufrac[:,1]

            plt.plot(en, taufrac,
                 color=colors[ic],
                 label=f"{flavlabel[alpha]}{flavlabel[beta]}",
                 linewidth=2)

            ic += 1

        plt.xscale("log")
        plt.xlabel("Neutrino energy, E [GeV]", fontsize=20)
        plt.ylabel(r"Tau neutrino fraction, $f_{\tau,\oplus}$", fontsize=20)


        total_ratio = int(sum(src_ratio))
        s = ",".join(
            f"{int(v)}/{total_ratio}" if (v and total_ratio != 1) 
            else str(int(v))
            for v in src_ratio
            )

        print(type(s),s)

        if d == 4:
            label = (
                fr"$f_S={tuple(s)}$"
                "\n"
                fr"$\mathring{{\kappa}}^{{({d})}}_{{\alpha\beta}}=10^{{-36}}$"
            )
        elif d == 5:
            label = (
                fr"$f_S={tuple(s)}$"
                "\n"
                fr"$\mathring{{\kappa}}^{{({d})}}_{{\alpha\beta}}=10^{{-45}}\ \mathrm{{GeV}}^{{-1}}$"
            )
        elif d == 6:
            label = (
                fr"$f_S=({s})$"
                "\n"
                fr"$\mathring{{\kappa}}^{{({d})}}_{{\alpha\beta}}=10^{{-54}}\ \mathrm{{GeV}}^{{-2}}$"
            )

        plt.legend(frameon=False, ncol=1, loc="lower left", fontsize=18)
        plt.xlim(1e5, 1e10)
        plt.ylim(-0.02,0.5)

        # plt.axvspan(1e4, 1e6, color='gray', alpha=0.1)
        # plt.axvspan(1e6, 1e11, color='gray', alpha=0.35)

        plt.tick_params(top=True, bottom=True, left=True, right=True,
                        which='both', direction='in')
        plt.tick_params(axis='both', which='major', labelsize=16, length=6, width=1.5)
        plt.tick_params(axis='both', which='minor', labelsize=8,  length=3, width=1.0)

        plt.text(1.5*1e5, 0.40, label, fontsize=18)
        plt.tight_layout()
        if redshifted==True:
            plt.savefig(f'figures/tau_frac_E_dim{d}_src{srctext}_{mode}_redshift{redz}.pdf', dpi=500)
        else:
            plt.savefig(f'figures/tau_frac_E_dim{d}_src{srctext}_{mode}.pdf', dpi=500)
        plt.show()
# """


### plot tau fraction wrt redshift with 5 energies fixed and 6 panels for 6 LIVparam
"""
dm21, dm31 = np.array([7.49e-5, 2.534e-3])
theta12, theta13, theta23 = [np.arcsin(np.sqrt(0.303)),
                              np.arcsin(np.sqrt(0.02225)),
                              np.arcsin(np.sqrt(0.572))]
delta = 197 * np.pi / 180
src_ratio = [1, 2, 0]
redz_range = np.linspace(0.01, 10, 100)

energies_GeV = [1e7, 1e8, 1e9, 1e10, 1e11]
elabels      = ["10 PeV", "100 PeV", "1 EeV", "10 EeV", "100 EeV"]
ecolors    = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
elinstyles = ['--', (0,(6,2,1,2)), (0,(4,2,1,2,1,2)), (0,(2,2)), (0,(1,1))]

flavlabel = [r'$e$', r'$\mu$', r'$\tau$']
pairs = [(alpha, beta) for alpha in range(3) for beta in range(alpha, 3)]

fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True, sharex=True)
axes = axes.flatten()

srctext = ''.join(f"{x}" for x in src_ratio)
total_ratio = int(sum(src_ratio))
s = ",".join(
    f"{int(v)}/{total_ratio}" if (v and total_ratio != 1) else str(int(v))
    for v in src_ratio
)

for idx, (alpha, beta) in enumerate(pairs):
    ax = axes[idx]

    LIVmatrix = np.zeros((3, 3), dtype=complex)
    LIVmatrix[alpha, beta] = 1e-54

    for E_GeV, lbl, col, ls in zip(energies_GeV, elabels, ecolors, elinstyles):
        ftau = [
            flavor_fraction_redshifted(
                E_GeV * 1e9, z, dm21, dm31,
                theta12, theta23, theta13, delta,
                d=6, a_eff=LIVmatrix, c_eff=LIVmatrix,
                src_ratio=src_ratio, flavor='tau'
            )
            for z in redz_range
        ]
        ax.plot(redz_range, ftau, label=lbl, color=col, linestyle=ls, linewidth=1.8)

    fa, fb = flavlabel[alpha][1:-1], flavlabel[beta][1:-1]
    ax.set_title(
        fr"$\mathring{{\kappa}}^{{(6)}}_{{{fa}{fb}}}=10^{{-54}}\ \mathrm{{GeV}}^{{-2}}$",
        fontsize=11
    )
    ax.tick_params(top=True, bottom=True, left=True, right=True,
                   which='both', direction='in')
    ax.tick_params(axis='both', which='major', labelsize=11)
    if idx >= 3:
        ax.set_xlabel("redshift $z$", fontsize=13)
    if idx % 3 == 0:
        ax.set_ylabel(r"$f_{\tau,\oplus}$", fontsize=13)

ax.set_ylim(-0.02, 0.52)
ax.set_xlim(redz_range[0], redz_range[-1])

handles, labs = axes[0].get_legend_handles_labels()
axes[-1].legend(handles, labs, frameon=False, fontsize=11, loc='lower right')

plt.tight_layout()
plt.savefig(f'figures/tau_frac_redshift_dim6_src{srctext}_6panel.pdf', dpi=200)
plt.show()
"""


### plot all fraction vs E for fixed d, fixed src varying alpha beta
"""
d=6

dm21, dm31 = np.array([7.49e-5, 2.534e-3]) # eV^2, as E is put in eV unit *1e-18
theta12, theta13, theta23 =  [np.arcsin(np.sqrt(0.303)), 
                              np.arcsin(np.sqrt(0.02225)), 
                              np.arcsin(np.sqrt(0.572))]

delta = 197*np.pi/180

pairs = [(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]
flavlabel = [r'$e$', r'$\mu$', r'$\tau$']

for src_ratio in [[0,1,0],[1,0,0]]:

    srctext = ''.join(f"{x}" for x in src_ratio)

    Erange = np.concatenate([
        np.logspace(4,np.log10(2e7),40,endpoint=False), 
        np.logspace(np.log10(2e7),np.log10(9e8),80,endpoint=False), 
        np.logspace(np.log10(9e8),11,40)
    ])

    fig, axes = plt.subplots(2,3, figsize=(14,8), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, (alpha,beta) in enumerate(pairs):

        LIVmatrix = np.zeros((3,3),dtype=complex)
        if d==4:
            LIVmatrix[alpha,beta] = 1e-36
        elif d==6:
            LIVmatrix[alpha,beta] = 1e-54

        frac_e, frac_mu, frac_tau = [], [], []

        for E in Erange:
            frac_e.append(flavor_fraction(E*1e9, dm21, dm31,
                                theta12, theta23, theta13, delta,
                                d=d, a_eff=LIVmatrix, c_eff=LIVmatrix,
                                src_ratio=src_ratio, flavor='e'))

            frac_mu.append(flavor_fraction(E*1e9, dm21, dm31,
                                theta12, theta23, theta13, delta,
                                d=d, a_eff=LIVmatrix, c_eff=LIVmatrix,
                                src_ratio=src_ratio, flavor='mu'))

            frac_tau.append(flavor_fraction(E*1e9, dm21, dm31,
                                theta12, theta23, theta13, delta,
                                d=d, a_eff=LIVmatrix, c_eff=LIVmatrix,
                                src_ratio=src_ratio, flavor='tau'))

        ax = axes[i]
        ax.plot(Erange, frac_e, label=r'$e$')
        ax.plot(Erange, frac_mu, label=r'$\mu$')
        ax.plot(Erange, frac_tau, label=r'$\tau$')

        ax.set_xscale('log')
        ax.set_title(f"{flavlabel[alpha]}{flavlabel[beta]}")
        ax.axvspan(1e4, 1e7, color='gray', alpha=0.1)   # first shade
        ax.axvspan(1e7, 1e11, color='gray', alpha=0.35)  # second shade

        ax.tick_params(top=True, bottom=True, left=True, right=True,
                        which='both', direction='in')
        ax.tick_params(axis='both', which='major', labelsize=12)  # major ticks
        ax.tick_params(axis='both', which='minor', labelsize=8)  # minor ticks
        ax.minorticks_on()

        if i==0:
            ax.legend()

    if d==4:
        plt.suptitle(fr"$f_S={tuple(src_ratio)}$, $d={d}$, $|\kappa_{{\alpha\beta}}|=10^{{-36}}\ $", fontsize=16)
    elif d==6:
        plt.suptitle(fr"$f_S={tuple(src_ratio)}$, $d={d}$, $|\kappa_{{\alpha\beta}}|=10^{{-55}}\ \mathrm{{GeV}}^{{-2}}$", fontsize=16)

    fig.supxlabel("Energy (GeV)")
    fig.supylabel("Flavor fraction")

    # plt.xscale("log")   # since your c is like 1e-75 ...
    # plt.xlabel("neutrino energy, E [GeV]", fontsize=16)
    # plt.ylabel(r"tau neutrino fraction, $f_{\tau,\oplus}$", fontsize=16)
    plt.legend(frameon=False, ncol=2, loc="lower left", fontsize=16)
    # plt.xlim(Erange[0],Erange[-1])
    plt.xlim(1e5,1e10)
#     plt.grid(True)
    # shaded regions

    plt.tight_layout()
    # plt.savefig(f'figures/all_frac_E_dim{d}_src{srctext}.pdf',dpi=300)
    plt.show()
"""

### plot osc prob vs E for fixed d, fix LIV, fixed src
"""
d = 6
dm21, dm31 = np.array([7.49e-5, 2.534e-3]) # eV^2, as E is put in eV unit *1e-18
theta12, theta13, theta23 =  [np.arcsin(np.sqrt(0.303)), 
                              np.arcsin(np.sqrt(0.02225)), 
                              np.arcsin(np.sqrt(0.572))]

delta = 197*np.pi/180

Erange = np.concatenate([
    np.logspace(4, np.log10(2e7), 40, endpoint=False),
    np.logspace(np.log10(2e7), np.log10(9e8), 80, endpoint=False),
    np.logspace(np.log10(9e8), 11, 40)
])

# ── LIV matrix: only κ_{μτ} = 1e-55 GeV^{-2} ─────────────────────────────────
LIVmatrix = np.zeros((3, 3), dtype=complex)
LIVmatrix[1, 2] = 1e-55   # κ_{μτ}, d=6 → GeV^{-2}

# ── compute avgP over energy range ────────────────────────────────────────────

def compute_avgP_vs_E(LIVmat=None):
    # Returns array of shape (len(Erange), 3, 3).
    P = np.zeros((len(Erange), 3, 3))
    for i, E in enumerate(Erange):
        P[i] = prob_avg(E * 1e9, dm21, dm31,
                        theta12, theta23, theta13, delta,
                        d=d,
                        a_eff=LIVmat,
                        c_eff=LIVmat)
    return P

P_std = compute_avgP_vs_E(None)           # no LIV baseline
P_liv = compute_avgP_vs_E(LIVmatrix)      # with κ_{μτ}

# ── plot ───────────────────────────────────────────────────────────────────────
dest_colors = ['steelblue', 'tomato', 'mediumseagreen']
dest_labels = [r'$\nu_e$', r'$\nu_\mu$', r'$\nu_\tau$']
src_labels  = [r'$\nu_e$', r'$\nu_\mu$', r'$\nu_\tau$']

fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)

src_greek  = [r'e', r'\mu', r'\tau']   # for use inside math mode
for src_idx, ax in enumerate(axes):
    for dest_idx in range(3):
        # no-LIV baseline — dotted
        ax.plot(Erange, P_std[:, src_idx, dest_idx],
                color=dest_colors[dest_idx], lw=1.5, ls=':',
                alpha=0.6)
        # with LIV — solid
        # ax.plot(Erange, P_liv[:, src_idx, dest_idx],
        #         color=dest_colors[dest_idx], lw=2.0, ls='-',
        #         label=dest_labels[dest_idx])
        plabel = (fr'$P(\nu_{{{src_greek[src_idx]}}}'
                  fr'\to \nu_{{{src_greek[dest_idx]}}})$')
        ax.plot(Erange, P_liv[:, src_idx, dest_idx],
                color=dest_colors[dest_idx], lw=2.0, ls='-',
                label=plabel)
    # shaded energy regions
    ax.axvspan(1e4, 1e7,  color='gray', alpha=0.08)
    ax.axvspan(1e7, 1e11, color='gray', alpha=0.25)

    ax.set_xscale('log')
    ax.set_xlim(1e5, 1e10)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r'Neutrino energy $E$ [GeV]', fontsize=13)
    ax.set_title(fr'Source: {src_labels[src_idx]}', fontsize=14)
    ax.tick_params(top=True, bottom=True, left=True, right=True,
                   which='both', direction='in')
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    ax.legend(frameon=False, fontsize=11, loc='center left')

axes[0].set_ylabel(r'Transition probability $P(\alpha \to \beta)$', fontsize=13)

# ── legend on rightmost axis ───────────────────────────────────────────────────
color_handles = [plt.Line2D([0],[0], color=c, lw=2, label=l)
                 for c, l in zip(dest_colors, dest_labels)]
style_handles = [
    plt.Line2D([-0.5],[0], color='gray', lw=1.5, ls=':', alpha=0.6, label='No LIV'),
    plt.Line2D([-0.5],[0], color='gray', lw=2.0, ls='-',             label=r'LIV: $\kappa_{\mu\tau}=10^{-55}\ \mathrm{GeV}^{-2}$'),
]

# leg1 = axes[0].legend(handles=color_handles, frameon=False, fontsize=12,
#                       loc='upper right', title='Final state', title_fontsize=11)
# axes[0].add_artist(leg1)
# axes[0].legend(handles=style_handles, frameon=False, fontsize=12,
#                loc='center right')

fig.suptitle(
    fr'$\mathring{{\kappa}}^{d}_{{\mu\tau}} = 10^{{-55}}\ \mathrm{{GeV}}^{{-2}}$',
    fontsize=14, y=0.99
)

plt.tight_layout()
# plt.savefig(f'figures/avgP_3src_dim{d}_kappa_mutau.pdf', dpi=300, bbox_inches='tight')
plt.show()
"""
