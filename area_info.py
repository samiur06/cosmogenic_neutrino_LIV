import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

year = 365.25 * 24 * 3600  # seconds

## --- POEMMA effective area data (eV, cm^2 sr) ---
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

## --- GRAND effective area data (eV, cm^2) ---
grand200k_area = pd.read_csv("eff_area/grand_200k.csv", 
                      header=None, names=["log10Enu", "area"]) 

E_eV = 1e9*grand200k_area["log10Enu"].values
eff_area = grand200k_area["area"].values    # cm^2 
grand_area_interp = interp1d(
    E_eV,
    eff_area,
    kind="linear",       # or "cubic" for smoother
    bounds_error=False,
    fill_value="extrapolate"
)


## --- IceCube Gen2 Radio effective area data (GeV, cm^2) ---
effareadata_ICgen2radio = [
    [10**7.5, 4.06024e6],
    [10**8.0, 4.06024e6],
    [10**8.5, 4.75677e7],
    [10**9.0, 2.99916e8],
    [10**9.5, 1.15092e9],
    [10**10.0, 3.11424e9],
    [10**10.5, 6.63344e9],
    [10**11.0, 1.21362e10],
    [10**11.5, 2.03881e10],
]

# Build interpolation in log-log space (E in eV, A in cm^2)
E_eV_list = 1e9 * np.array([x for x, _ in effareadata_ICgen2radio])
A_cm2_list = np.array([y for _, y in effareadata_ICgen2radio])
_area_interp = interp1d(np.log10(E_eV_list), np.log10(A_cm2_list), 
                        kind='linear', fill_value='extrapolate')

def ICgen2radio_area_interp(E_eV):
    """Effective area [cm^2] of IceCube Gen2 Radio vs energy [eV]."""

    E = np.asarray(E_eV)  # handles float or array input

    Emin = np.min(E_eV_list)
    Emax = np.max(E_eV_list)

    # Initialize output array
    out = np.zeros_like(E, dtype=float)

    # Valid energy region (boolean mask)
    mask = (E >= Emin) & (E <= Emax)

    # Interpolate only where valid
    out[mask] = 10 ** _area_interp(np.log10(E[mask]))

    # If original input was scalar, return scalar
    return out.item() if np.isscalar(E_eV) else out

# # Example:
# E_test = 1e19  # eV
# print(f"A_eff at {E_test:.2e} eV:  \nPOEMMA = {poemma_area_interp(E_test)} cm^2 sr",
#     f"\nGRAND = {grand_area_interp(E_test)} cm^2",
#     f"\nIC-gen2 radio = {ICgen2radio_area_interp(E_test)} cm^2")

def experimental_factor(exp):
    if exp=='poemma':
        time_factor=5*year*0.2 # 0.2 for duty cycle
    elif exp=='ICgen2radio':
        time_factor=10*year
    elif exp=='grand200k':
        time_factor=10*year
        
    else:
        raise TypeError("experiment not found")

    return time_factor

# experimental_factor('poemma')