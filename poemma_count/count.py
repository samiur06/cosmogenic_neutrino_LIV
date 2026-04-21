import numpy as np
from sympy import sin, cos 
import sympy as sp
import pandas as pd
from cmath import *
import matplotlib.pyplot as plt
from scipy.integrate import odeint
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

import ternary

print("Version", ternary.__version__)

t0 = timeit.default_timer()

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

# ---- Test the interpolator ----
E_test = 1e15  # eV
print(f"flux at E = {E_test:.1e} eV : {flux_interp(E_test):.3e} GeV/(cm^2 s sr)")

poemma_area = pd.read_csv("eff_area/poemma_nutau_tau.csv", 
                      header=None, names=["log10Enu", "area"]) 

kmtocm = 1e5
E_eV = 10**poemma_area["log10Enu"].values
eff_area = poemma_area["area"].values * kmtocm**2   # cm^2 sr

# Build interpolation function: input E (eV) → 
area_interp = interp1d(
    E_eV,
    eff_area,
    kind="linear",       # or "cubic" for smoother
    bounds_error=False,
    fill_value="extrapolate"
)

# ---- Test the interpolator ----
E_test = 1e17  # eV
print(f"area at E = {E_test:.1e} eV : {area_interp(E_test):.3e} cm^2 sr")

# Integration limits in eV
E_min = 1e15      # 1 PeV = 1e15 eV
E_max = 100e18      # 100 EeV = 1e20 eV
year = 365*24*3600
# Define the integrand_flux_area
def integrand_flux_area(E):
    EnuGeV = E * 1e-9
    # flux = rate/E^2 = 1/(GeV cm^2 s sr)
    # area = cm^2 sr
    # flux * area = GeV^-1 s^-1
    unit_refiner = (1/EnuGeV**2 ) * 1e-9 
    #factor explanaton: 1/E^2, 1/GeV to 1/eV
    return flux_interp(E) * area_interp(E) * unit_refiner    # 1/(s)

# Perform the integral
result, err = quad(integrand_flux_area, E_min, E_max, limit=500, epsabs=0, epsrel=1e-4)

print(f"Integral = {result:.3e}  (same units as flux × area × eV)")
print(f"Estimated absolute error = {err:.1e}")

# tobs = 0.2*5 years, pg. 16 , https://arxiv.org/pdf/1902.11287
# full 360 degree i.e. 2pi azimuth # hence no factor 30/360
# 0.2 for duty cycle, 1/3 for tau flavor
print('tau count:', result * 5*year*0.20/3) 
