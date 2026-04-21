## universal tau_count

from LIV_fraction import * 
from area_info import *
fluxtype=input("Enter fluxtype (opt/mod/pess): ")
## cosmogenic flux
if fluxtype=='opt':
    neuflux = pd.read_csv("neuflux_1505.04020/cosmo_nuflux_1505_04020_fig2_opt_AGN_blue_solid.csv", 
        header=None, names=["log10Enu", "rate"]) 
elif fluxtype=='mod':
    neuflux = pd.read_csv("neuflux_1505.04020/cosmo_nuflux_1505_04020_fig2_mod_SFR_green_solid.csv", 
        header=None, names=["log10Enu", "rate"]) 
elif fluxtype=='pess':
    neuflux = pd.read_csv("neuflux_1505.04020/cosmo_nuflux_1505_04020_fig2_pess_noEvol_red_solid.csv", 
        header=None, names=["log10Enu", "rate"]) 
elif fluxtype=='dentonGRAND':
    neuflux = pd.read_csv("neuflux_1505.04020/flux_fig3_alphaS0m3_1809.04866.csv", 
        header=None, names=["log10Enu", "rate"]) 
else:
    raise TypeError("fluxtype is incorrect,\nit has to be one of opt/mod/pess")

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

# Define the integrand_flux_area
# Integration limits in eV
E_min = 1e15      # 1 PeV = 1e15 eV
E_max = 100e18      # 100 EeV = 1e20 eV
year = 365*24*3600


# Define the integrand_flux_area
def integrand_flux_area_exp(E, exp, d, a_eff, c_eff, src_ratio):
    EnuGeV = E * 1e-9
    # flux = rate/E^2 = 1/(GeV cm^2 s sr)
    # area = cm^2 sr
    # flux * area = GeV^-1 s^-1
    unit_refiner = (1/EnuGeV**2 ) * 1e-9 
    #factor explanaton: 1/E^2, 1/GeV to 1/eV
    area_factor = 0
    if type(exp)!=str:
        raise TypeError("experiment is not string!")
        
    explist = ['poemma', 'ICgen2radio', 'grand200k']
    if exp=='poemma':
        area_factor=poemma_area_interp(E)
    elif exp=='ICgen2radio':
        area_factor=ICgen2radio_area_interp(E)
    elif exp=='grand200k':
        area_factor=grand_area_interp(E)
    else:
        raise TypeError(f"experiment not found, it must be on of {explist}")

    area_factor = np.maximum(area_factor, 0)

    dm21, dm31 = np.array([7.49e-5, 2.534e-3]) # eV^2, as E is put in eV unit *1e-18
    theta12, theta13, theta23 =  [np.arcsin(np.sqrt(0.303)), 
                                  np.arcsin(np.sqrt(0.02225)), 
                                  np.arcsin(np.sqrt(0.572))]

    delta = 197*np.pi/180

    ftau_fraction = flavor_fraction(E, dm21, dm31, 
              theta12, theta23, theta13, delta,
              d, a_eff, c_eff,
              src_ratio, flavor='tau')
    
    return flux_interp(E) * ftau_fraction * area_factor * unit_refiner    # 1/(s)

def tau_count(exp, d=0, a_eff=None, c_eff=None, src_ratio=[1,2,0]):
    E_min = 1e15        # 1 PeV = 1e15 eV
    E_max = 100e18      # 100 EeV = 1e20 eV
    # Perform the integral
    def log_integrand(logE, *args):
        E = 10**logE
        # We multiply by E * ln(10) because dE = E * ln(10) * d(log10E)
        return integrand_flux_area_exp(E, *args) * E * np.log(10)

    logE_peak = [18,18.3,18.6,18.9]
    result, err = quad(log_integrand, 
                       np.log10(E_min), np.log10(E_max),
                       args=(exp,d, a_eff, c_eff, src_ratio), 
                       limit=100,
                       points=logE_peak, # Tells quad: "Pay attention here!"
                       epsabs=1e-10, epsrel=1e-3)

    # logE_vals = np.linspace(np.log10(E_min), np.log10(E_max), 500)
    # y_vals = [log_integrand(le, exp, d, a_eff, c_eff, src_ratio) for le in logE_vals]
    # plt.plot(logE_vals, y_vals)
    # plt.show()
    # print(result,err)
    return result * experimental_factor(exp)

"""
print(tau_count('poemma'), 
    # tau_count('ICgen2radio'), 
    tau_count('grand200k'))
print(tau_count('poemma', src_ratio=[0,1,0]), 
    # tau_count('ICgen2radio', src_ratio=[0,1,0]), 
    tau_count('grand200k', src_ratio=[0,1,0]))

print(tau_count('poemma', src_ratio=[1,0,0]), 
    # tau_count('ICgen2radio', src_ratio=[1,0,0]), 
    tau_count('grand200k', src_ratio=[1,0,0]))
"""
