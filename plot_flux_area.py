from LIV_fraction import * 


##################################
######## make flux plot ##########
##################################

files = {
    "opt": "neuflux_1505.04020/cosmo_nuflux_1505_04020_fig2_opt_AGN_blue_solid.csv",
    "mod": "neuflux_1505.04020/cosmo_nuflux_1505_04020_fig2_mod_SFR_green_solid.csv",
    "pess": "neuflux_1505.04020/cosmo_nuflux_1505_04020_fig2_pess_noEvol_red_solid.csv"
}

plt.figure(figsize=(7,5))

for fluxtype, filename in files.items():
    neuflux = pd.read_csv(filename, header=None, names=["log10Enu", "rate"])

    E_eV = 10**neuflux["log10Enu"].values
    rate = neuflux["rate"].values  # GeV/(cm^2 s sr)

    # interpolation function (optional)
    flux_interp = interp1d(
        E_eV, rate,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate"
    )

    plt.loglog(E_eV, rate, label=fluxtype)

plt.xlabel(r"$E_\nu \;[\mathrm{eV}]$", fontsize=13)
plt.ylabel(r"Flux  $[\mathrm{GeV^{-1}\,cm^{-2}\,s^{-1}\,sr^{-1}}]$", fontsize=13)
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.title("Cosmogenic Neutrino Flux (opt / mod / pess)")

plt.tight_layout()
plt.close()


##################################
######## load area ###############
##################################

# Define the integrand_flux_area
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


##################################
######## make area plot ##########
##################################

# Common energy grid (eV)
E_plot = np.logspace(15, 20, 300)

# Evaluate effective areas
A_poemma = poemma_area_interp(E_plot)      # cm^2 sr (if you want cm^2 only, divide by sr if needed)
A_grand  = grand_area_interp(E_plot)       # cm^2
A_IC     = ICgen2radio_area_interp(E_plot) # cm^2

plt.figure(figsize=(7,5))

plt.loglog(E_plot, A_poemma/(4*np.pi), label="POEMMA", lw=2)
plt.loglog(E_plot, A_grand,  label="GRAND200k", lw=2)
plt.loglog(E_plot, A_IC,     label="IceCube Gen2 Radio", lw=2)

plt.xlabel(r"$E_\nu\;[\mathrm{eV}]$", fontsize=13)
plt.ylabel(r"Effective Area $[\mathrm{cm}^2]$", fontsize=13)

plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.title("Effective Area vs Energy")

plt.tight_layout()
plt.close()

##################################
######## make twin plot ##########
##################################

colors_flux = {
    "opt":  "#0072B2",  # blue
    "mod":  "#009E73",  # green
    "pess": "#D55E00"   # vermillion
}

colors_area = {
    "POEMMA": "#CC79A7",  # purple
    "GRAND":  "#E69F00",  # orange
    "IC":     "#56B4E9"   # sky blue
}

fig, ax1 = plt.subplots(figsize=(8,6))

# ================= FLUX (left y-axis) =================
for fluxtype, filename in files.items():
    neuflux = pd.read_csv(filename, header=None, names=["log10Enu", "rate"])

    E_flux = 10**neuflux["log10Enu"].values
    rate = neuflux["rate"].values

    ax1.loglog(
        E_flux, rate,
        label=f"Flux ({fluxtype})",
        color=colors_flux[fluxtype],
        lw=2
    )

ax1.set_xlabel(r"$E_\nu\;[\mathrm{eV}]$", fontsize=13)
ax1.set_ylabel(r"Flux  $[\mathrm{GeV^{-1}\,cm^{-2}\,s^{-1}\,sr^{-1}}]$", fontsize=13)
ax1.tick_params(axis='y')
ax1.set_xlim(1e14,3e19)
ax1.set_ylim(2e-9,3e-7)

# ================= AREA (right y-axis) =================
ax2 = ax1.twinx()

E_plot = np.logspace(15, 20, 300)

A_poemma = poemma_area_interp(E_plot)/(4*np.pi)
A_grand  = grand_area_interp(E_plot)
A_IC     = ICgen2radio_area_interp(E_plot)

ax2.loglog(E_plot, A_poemma, label="POEMMA", color=colors_area["POEMMA"], lw=2, ls="--")
ax2.loglog(E_plot, A_grand,  label="GRAND200k", color=colors_area["GRAND"], lw=2, ls="--")
# ax2.loglog(E_plot, A_IC,     label="IceCube Gen2 Radio", color=colors_area["IC"], lw=2, ls="--")

ax2.set_ylabel(r"Effective Area $[\mathrm{cm}^2]$", fontsize=13)
ax2.tick_params(axis='y')
# ax2.set_xlim(1e14,3e19)
# ================= Legends =================
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=10)

# ================= Formatting =================
# ax1.grid(True, which="both", ls="--", alpha=0.4)
plt.title("Cosmogenic Neutrino Flux and Detector Effective Area")

plt.tight_layout()
plt.show()