## universal tau_count

from LIV_fraction import * 
from area_info import *

fluxtype='dentonGRAND'
exp='grand200k'

neuflux = pd.read_csv("neuflux_1505.04020/flux_fig3_alphaS0m3_1809.04866.csv", 
        header=None, names=["log10Enu", "rate"]) 

E_eV = 1e9 * 10**neuflux["log10Enu"].values
rate = neuflux["rate"].values   # already GeV/(cm^2 s sr)")

# Build interpolation function: input E (eV) 
flux_interp = interp1d(
    E_eV,
    rate,
    kind="linear",       # or "cubic" for smoother
    bounds_error=False,
    fill_value="extrapolate"
)

# Define the integrand_flux_area
# Integration limits in eV
E_min = 1e15      # 1 PeV = 1e15 eV
E_max = 100e18      # 100 EeV = 1e20 eV
year = 365*24*3600


# Define the integrand_flux_area
def integrand_flux_area_exp(E, exp):
    EnuGeV = E * 1e-9
    # flux = rate/E^2 = 1/(GeV cm^2 s sr)
    # area = cm^2 sr
    # flux * area = GeV^-1 s^-1
    unit_refiner = (1/EnuGeV**2 ) * 1e-9 
    #factor explanaton: 1/E^2, 1/GeV to 1/eV
    area_factor = 0
    if type(exp)!=str:
        raise TypeError("experiment is not string!")
        
    if exp=='grand200k':
        area_factor=grand_area_interp(E)
    else:
        raise TypeError(f"experiment not found, it must be on of {explist}")

    area_factor = np.maximum(area_factor, 0)

    dm21, dm31 = np.array([7.49e-5, 2.534e-3]) # eV^2, as E is put in eV unit *1e-18
    theta12, theta13, theta23 =  [np.arcsin(np.sqrt(0.303)), 
                                  np.arcsin(np.sqrt(0.02225)), 
                                  np.arcsin(np.sqrt(0.572))]

    delta = 197*np.pi/180


    # ftau_fraction = flavor_fraction(E, dm21, dm31, 
    #           theta12, theta23, theta13, delta,
    #         flavor='tau')
    ftau_fraction = 1/3
    
    return flux_interp(E) * ftau_fraction * area_factor * unit_refiner    # 1/(s)

def tau_count(exp, E_min, E_max):
    # E_min = 1e15        # 1 PeV = 1e15 eV
    # E_max = 100e18      # 100 EeV = 1e20 eV
    # Perform the integral
    def log_integrand(logE, *args):
        E = 10**logE
        # We multiply by E * ln(10) because dE = E * ln(10) * d(log10E)
        return integrand_flux_area_exp(E, *args) * E * np.log(10)

    result, err = quad(log_integrand, 
                       np.log10(E_min), np.log10(E_max),
                       args=(exp), 
                       limit=100,
                       epsabs=1e-10, epsrel=1e-3)

    return result * experimental_factor(exp)


# print(tau_count(exp, 1e18, 10**(18.25)))

### reproduce fig 3, leftmost plot with m=-3

# bin edges in eV (convert GeV to eV: multiply by 1e9)
log_edges_GeV = np.arange(8, 10.25 + 0.25, 0.25)  # log10(E/GeV)
energy_edges_eV = 10**log_edges_GeV * 1e9           # convert to eV

# compute tau count per bin
counts = []
for i in range(len(energy_edges_eV) - 1):
    E_min = energy_edges_eV[i]
    E_max = energy_edges_eV[i + 1]
    counts.append(tau_count(exp, E_min, E_max))

counts = np.array(counts)

# plot

fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
ax.step(
    10**log_edges_GeV,          # x in GeV
    np.append(counts, counts[-1]),
    where='post',
    color='#0072B2',
    linewidth=3,
    linestyle='-',
)
ax.set_xscale('log')
ax.set_xlabel(r'$E$ [GeV]')
ax.set_ylabel(r'Tau count/10 years/bin')
ax.set_xlim(1e8, 10**10.25)
ax.set_ylim(0, 20)
ax.grid(True)

ticks = np.arange(8, 10.5, 0.5)
ax.set_xticks(10**ticks)
ax.set_xticklabels([f'$10^{{{t:.1f}}}$' for t in ticks])
ax.set_yticks(np.arange(0, 25, 5))
ax.set_title(r'1809.04866 (figure 4): GRAND count for flux $\alpha_S=0,m=-3$')

plt.tight_layout()
# plt.savefig('figures/reproduce_fig4_1809.04866.png',dpi=200)
plt.show()
