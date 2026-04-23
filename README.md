# cosmogenic_neutrino_LIV

Code released with [arXiv:2604.19880](https://arxiv.org/abs/2604.19880), for computing cosmogenic neutrino fluxes and tau neutrino event counts in the presence of Lorentz Invariance Violation (LIV). The cosmogenic neutrino flux is produced using the ultra-high-energy cosmic rays Monte Carlo (MC) events from [SimProp-v2r4](https://github.com/SimProp/SimProp-v2r4) simulation.

We share the code to construct all-flavor and flavor-specific cosmogenic neutrino flux (including the effect of flavor transition) from the MC events. 

*Please cite our paper [arXiv:2604.19880](https://arxiv.org/abs/2604.19880) if you use any of our code/results.* Feel free to contact via [GitHub](https://github.com/samiur06) for any questions.


---

## Table of Contents
- [Requirements](#requirements)
- [Workflow](#workflow)
- [Scripts](#scripts)
- [Directory Structure](#directory-structure)
- [SimProp MC Events](#simprop-mc-events)

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

The expected number of neutrino events for any flavor can be computed directly from step 4, skipping steps 1–3.

1. Generate `SimProp` MC events. It is **optional** as derived data are provided (also see step 3).
2. To produce Fig. 2 from the paper (shown below), run `totalflux_neutrino_mc.py` to generate the flux sensitivity plot using the saved flux files in `data/total_neutrino_flux` and experimental results and projections from `NeuExpSensitivity`. Optionally, you may uncomment the snippet in that script to create flux from the function `run_flux_pipeline` using `SimProp` ROOT files. The all-flavor flux calculation and plot is independent of steps 3 and 4.

<div align="center">
<figure>
  <img width="700" height="490" alt="fitted_flux_with_sensitivities" src="https://github.com/user-attachments/assets/00be58a2-f14c-4b4a-b39b-7a0655ceba86" />
  <figcaption> Fig. All-flavor cosmogenic neutrino flux for SFR and no-evolution models, with experimental constraints and sensitivities. </figcaption>
</figure>
</div>

3. Run `save_neutrino_mc.py` to read `SimProp` ROOT files and save per-flavor neutrino flux arrays to `data/flux_array`. This is **optional** as the derived data are provided in that directory. 
4. Run `compute_taucount.py` to compute the expected tau neutrino event counts in GRAND and POEMMA for a given LIV parameter. You can change the detected flavor (`det_flav`) to compute other neutrino species as well (`det_flav = 0, 1, 2` corresponds to electron, muon, and tau neutrinos, default being `2`). Users can calculate event count for other experiments sensitive to a similar energy range with the experimental effective area provided (`area_info.py` is to be updated). An example snippet, at the end of the script, provides the following event counts:

| Experiment | Redshift evolution | $N_\tau$ (standard) | $N_\tau$ ($\mathring{\kappa}^{(6)}_{e\mu} = 10^{-58}\ \text{GeV}^{-2}$) |
|:-----------:|:----------:|--------------------:|:-------------------------------------------------------------------------:|
| GRAND200k  | No evolution | 172.61 | 90.89 |
| POEMMA     | No evolution | 3.31   | 1.15  |
| GRAND200k  | SFR          | 580.33 | 277.81 |
| POEMMA     | SFR          | 11.89  | 3.54  | 

<div align="center">
<figure>
  <img width="700" height="490" alt="image" src="https://github.com/user-attachments/assets/1be7a0e1-5432-4387-ba4c-14feac81b9f8" />
  <figcaption> Fig. Number of tau neutrino events as a function of LIV parameters at GRAND for the neutrino flux with SFR evolution. </figcaption>
</figure>
</div>

---

## Scripts 

| Script name | Description |
|:------:|:------------|
| `totalflux_neutrino_mc.py` | Reads `SimProp` Monte Carlo ROOT file(s) and saves the all-flavor cosmogenic neutrino flux (stored in `data/total_neutrino_flux/`). Produces a flux plot comparing SFR evolution and no source evolution models, overlaid with experimental constraints from `NeuExpSensitivity/`. Figures saved in `figures/` correspond to Fig. 2 in the paper. |
| `save_neutrino_mc.py` | Reads `SimProp` Monte Carlo ROOT file(s) and saves events containing neutrinos of a particular flavor. Outputs stored in `data/flux_array/` are used to compute the cosmogenic neutrino flux and expected number of neutrino events for that flavor. |
| `LIV_fraction.py` | Computes the neutrino flavor transition probability and flavor fraction in the presence of LIV. Output is used by `compute_taucount.py` to determine the tau neutrino fraction for a given LIV parameter value. |
| `area_info.py` | Provides the effective area or geometric aperture (including field of view) and detector runtime for the experiments. Loads and processes relevant files from the `eff_area/` directory. |
| `compute_taucount.py` | Computes the expected number of tau neutrino events in GRAND and POEMMA for a given LIV parameter value. Requires flux arrays from `save_neutrino_mc.py` (`data/flux_array/`), flavor fraction functions from `LIV_fraction.py`, and experiment-specific area/aperture from `area_info.py`. |

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

Generating new MC events is optional, since the necessary derived data are provided in the `data` directory to reproduce our results. 


Events can be generated from [SimProp-v2r4](https://github.com/SimProp/SimProp-v2r4) simulation. Event generation follows the procedure described in the [SimProp-v2r4 paper (arXiv:1705.03729)](https://arxiv.org/abs/1705.03729), 
which is the key reference for the flux computation in `totalflux_neutrino_mc.py` and `save_neutrino_mc.py`.

The ROOT files directory (MC events) must be set correctly in the function `run_flux_pipeline` in `totalflux_neutrino_mc.py` and the function `save_per_neu_arrays` in `save_neutrino_mc.py`. The ROOT files directory `{data_dir}_{cosmo_evolution}` is evaluated inside these functions,  which can be changed to user-specific path containing the ROOT files. 

The spectral index used to generate the events and the source emissivity applied to compute the fluxes can be found in both the scripts and the paper. The dictionary `SCENARIOS` contains this information in the scripts. This dictionary must be updated in the scripts for different spectral indices and source emissivities. The emissivity can take any reasonable value, but the spectral index should match the value used to produce the events. 

