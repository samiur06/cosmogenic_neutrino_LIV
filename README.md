# cosmogenic_neutrino_LIV

Code released with arXiv:xxxx:xxxxxx. 
Code for computing cosmogenic (ultra-high energy) neutrino fluxes and tau neutrino event counts in the presence of Lorentz Invariance Violation (LIV). The cosmogenic neutrino flux is produced using the ultra-high-energy cosmic rays Monte Carlo (MC) events from [SimProp-v2r4](https://github.com/SimProp/SimProp-v2r4) simulation.

We share the code to construct all-flavor and flavor-specific cosmogenic neutrino flux (including the effect of flavor transition) from the MC events. 

Feel free to contact via [GitHub](https://github.com/samiur06) for any questions.

*Please cite us if you use any of our code/results.* 

---

## Table of Contents
- [Requirements](#requirements)
- [Workflow](#workflow)
- [Scripts](#scripts)
- [Directory Structure](#directory-structure)
- [SimProp MC Events](#simprop-mc-events-optional)

---

## Requirements

- Python 3.8+
- NumPy
- SciPy
- Matplotlib
- uproot (for reading `SimProp` ROOT files)

Install dependencies with:

```bash
pip install numpy scipy matplotlib uproot
```

If LaTeX in Matplotlib is not supported, comment out the LaTeX rendering lines. 

---

## Workflow

1. Generate `SimProp` MC events. It is **optional** as derived data are provided (also see step 3).
2. To produce Fig. 2, run `totalflux_neutrino_mc.py` to generate the flux sensitivity plot using the saved files in `data/total_neutrino_flux`. Optionally, you may uncomment the snippet in that script to create flux from the function `run_flux_pipeline` using `SimProp` ROOT files. This is independent of steps 3 and 4.
3. Run `save_neutrino_mc.py` to read `SimProp` ROOT files and save per-flavor neutrino flux arrays to `data/flux_array`. This is **optional** as the derived data are provided in that directory. 
4. Run `compute_taucount.py` to compute the expected tau neutrino event counts in GRAND and POEMMA for a given LIV parameter. You can change the detected flavor to compute other neutrino species (muon and electron neutrinos) as well. Users can calculate event count for other future experiments sensitive to the similar energy range with the experimental effective area provided (`area_info.py` is to be updated).  


---

## Scripts 

### `totalflux_neutrino_mc.py`

Reads `SimProp` Monte Carlo ROOT file(s) and saves the all-flavor cosmogenic neutrino flux (stored in `data/total_neutrino_flux/`).
Produces a flux plot comparing two source redshift evolution models — Star Formation Rate (SFR) evolution and no source evolution — overlaid with experimental constraints and sensitivities from the `NeuExpSensitivity/` directory. The resulting figures are saved in `figures/` and correspond to Fig. 2 in the paper.

### `save_neutrino_mc.py`

Reads `SimProp` Monte Carlo ROOT file(s) and saves events containing neutrinos of a particular flavor. The saved outputs (stored in `data/flux_array/`) are used to compute the cosmogenic neutrino flux and the expected number of neutrino events for that flavor.

### `LIV_fraction.py`

Computes the neutrino flavor transition probability and flavor fraction in the presence of LIV. The output is used by `compute_taucount.py` to determine the tau neutrino fraction for a given LIV parameter value.

### `area_info.py`

Provides the effective area or geometric aperture (including field of view) and detector runtime for the experiments. Loads and processes the relevant files from the `eff_area/` directory.

### `compute_taucount.py`

Computes the expected number of tau neutrino events in GRAND and POEMMA for a given LIV parameter value. Requires the flux arrays saved by `save_neutrino_mc.py` (from `data/flux_array`), the flavor fraction functions from `LIV_fraction.py`, and experiment-specific area/aperture from `area_info.py`.

---

## Directory Structure

```
cosmogenic_neutrino_LIV/
├── NeuExpSensitivity/         # Experimental constraints and sensitivity 
├── data/
│   ├── flux_array/            # Per-flavor neutrino flux arrays (output of save_neutrino_mc.py)
│   └── total_neutrino_flux/   # All-flavor neutrino flux (output of totalflux_neutrino_mc.py)
├── eff_area/                  # Effective area / aperture files for GRAND, POEMMA, and others
├── figures/                   # Output figures (all-flavor flux with experimental constraints)
├── LIV_fraction.py
├── save_neutrino_mc.py
├── totalflux_neutrino_mc.py
├── compute_taucount.py
└── area_info.py
```


---

## `SimProp` MC Events (optional)

Event generation follows the procedure described in the [SimProp-v2r4 paper (arXiv:1705.03729)](https://arxiv.org/abs/1705.03729), 
which is the key reference for the flux computation in `totalflux_neutrino_mc.py` and `save_neutrino_mc.py`.

Generating new MC events is optional — the necessary derived data are already provided in the `data` directory. If users wish to simulate new events under different conditions, the MC event paths must be set correctly in `totalflux_neutrino_mc.py` and `save_neutrino_mc.py`.

The spectral index used to generate the events and the source emissivity applied to compute the fluxes can be found in both the scripts and the paper.

