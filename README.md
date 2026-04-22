# cosmogenic_neutrino_LIV

Analysis code for computing cosmogenic (ultra-high energy) neutrino fluxes and tau neutrino event rates in the presence of Lorentz Invariance Violation (LIV), targeting the **GRAND** and **POEMMA** experiments.

---

## Scripts

### `LIV_fraction.py`
Computes the neutrino flavor transition probability and flavor fraction in the presence of LIV. The output is used by `compute_taucount.py` to determine the tau neutrino fraction for a given LIV parameter value.

### `save_neutrino_mc.py`
Reads SiMProp Monte Carlo ROOT file(s) and saves events containing neutrinos of a particular flavor. The saved outputs (stored in `data/flux_array/`) are used to compute the cosmogenic neutrino flux and the expected number of neutrino events for that flavor.

### `totalflux_neutrino_mc.py`
Does two things:
- Reads SiMProp Monte Carlo ROOT file(s) and saves the all-flavor cosmogenic neutrino flux (stored in `data/total_neutrino_flux/`).
- Produces a flux plot comparing two source redshift evolution models — Star Formation Rate (SFR) evolution and no source evolution — overlaid with experimental constraints and sensitivities from the `NeuExpSensitivity/` directory. The resulting figures are saved in `figures/` and correspond to Fig. 2 in the paper.

### `compute_taucount.py`
Computes the expected number of tau neutrino events in GRAND and POEMMA for a given LIV parameter value. Requires the flux arrays saved by `save_neutrino_mc.py` (from `data/flux_array/`) and the flavor fraction functions from `LIV_fraction.py`.

### `area_info.py`
Provides the effective area or geometric aperture (including field of view) and detector runtime for the experiments. Loads and processes the relevant files from the `eff_area/` directory.

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
- uproot (for reading SiMProp ROOT files)

Install dependencies with:

```bash
pip install numpy scipy matplotlib uproot
```

---

## Workflow

1. Run `save_neutrino_mc.py` to read SiMProp ROOT files and save per-flavor neutrino flux arrays to `data/flux_array/`.
2. Run `totalflux_neutrino_mc.py` to compute and save the all-flavor flux and generate the flux comparison plot (Fig. 2).
3. Run `compute_taucount.py` to compute the expected tau neutrino event counts in GRAND and POEMMA for a given LIV parameter.
