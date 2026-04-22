# cosmogenic_neutrino_LIV

Code released with arXiv:xxxx:xxxxxx. 
Code for computing cosmogenic (ultra-high energy) neutrino fluxes and tau neutrino event counts in the presence of Lorentz Invariance Violation (LIV). The cosmogenic neutrino flux is produced using the ultra-high-energy cosmic rays Monte Carlo (MC) events from [SimProp-v2r4](https://github.com/SimProp/SimProp-v2r4) simulation.

We share the code to construct all-flavor and flavor-specific cosmogenic neutrino flux (including the effect of flavor transition) from the MC events. 

Feel free to contact via [GitHub](https://github.com/samiur06) for any questions.

*Please cite us if you use any of our code/results.* 

---

## `SimProp` MC Events (optional)

It is not required to generate MC events as we have provided the necessary derived data from `SimProp` MC events in the `data` directory. Users can create new MC events simulating [SimProp-v2r4](https://github.com/SimProp/SimProp-v2r4). The MC events location must be input correctly in `totalflux_neutrino_mc.py` and `save_neutrino_mc.py` to generate derived data. 

---

## Scripts 

### `totalflux_neutrino_mc.py`
This script does two things:
- Reads `SimProp` Monte Carlo ROOT file(s) and saves the all-flavor cosmogenic neutrino flux (stored in `data/total_neutrino_flux/`).
- Produces a flux plot comparing two source redshift evolution models — Star Formation Rate (SFR) evolution and no source evolution — overlaid with experimental constraints and sensitivities from the `NeuExpSensitivity/` directory. The resulting figures are saved in `figures/` and correspond to Fig. 2 in the paper.

### `save_neutrino_mc.py`
This is the key program to generate event counts. It reads `SimProp` Monte Carlo ROOT file(s) and saves events containing neutrinos of a particular flavor. The saved outputs (stored in `data/flux_array/`) are used to compute the cosmogenic neutrino flux and the expected number of neutrino events for that flavor.

### `LIV_fraction.py`
Computes the neutrino flavor transition probability and flavor fraction in the presence of LIV. The output is used by `compute_taucount.py` to determine the tau neutrino fraction for a given LIV parameter value.

### `area_info.py`
Provides the effective area or geometric aperture (including field of view) and detector runtime for the experiments. Loads and processes the relevant files from the `eff_area/` directory.

### `compute_taucount.py`
Dependencies: `LIV_fraction.py`, `area_info.py`, the data in the directories `eff_area` and `data/flux_array`. 
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

1. Generate `SimProp` MC events. It is *optional* as derived data are provided (see workflow 3).
2. Producing Fig. 2: Run `totalflux_neutrino_mc.py` to generate the flux sensitivity plot using the saved files in `data/total_neutrino_flux`. Optionally, you may uncomment the snippet in that script to create flux from the function `run_flux_pipeline` using `SimProp` ROOT files. This is independent from the remaining two scripts in 3 and 4.
3. Run `save_neutrino_mc.py` to read `SimProp` ROOT files and save per-flavor neutrino flux arrays to `data/flux_array`. This is *optional* as the derived data are provided in that directory. 
4. Run `compute_taucount.py` to compute the expected tau neutrino event counts in GRAND and POEMMA for a given LIV parameter. You can change the detected flavor to compute other neutrino species (muon and electron neutrinos) as well. 

