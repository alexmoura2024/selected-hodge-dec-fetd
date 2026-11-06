# Topological Selected-Solve Inverse Hodge Code Package

This repository contains the Python scripts used for the numerical experiments in the manuscript:

**A Topological Selected-Solve Approximation of Inverse Hodge Operators for Explicit DEC/FETD Maxwell Simulations**

## Authors

- A. S. Moura
- W. G. Facco

## Scope

The scripts reproduce the algebraic, dynamic, selected-solve, scaling, LU-step, CFL, and figure-generation experiments used in the article. The main object of study is the sparse selected inverse Hodge approximation

```text
G_{H,r} = P_{S_{H^r}}(H^{-1})
```

with emphasis on the radius-three approximation `G_{H,3}` and the selected-solve construction `G_{H,3}^{sel}`.

## Directory structure

```text
scripts/
  figures_cpc.py
  teste01_selected_hodge.py
  teste01B_pattern_enrichment.py
  teste01D_material_contrast.py
  teste02_leapfrog_selected_hodge.py
  teste02B_time_growth_cavity.py
  teste02C_modal_frequency_cavity.py
  teste03A_selected_solve_hodge.py
  teste03B_scaling_selected_solve.py
  teste04_large_cavity_selected_solve.py
  teste05_compare_selected_inverse_vs_lu_step.py
  teste06_cfl_stability_sweep.py
requirements.txt
CITATION.cff
LICENSE
MANIFEST.json
```

## Main workflow

A typical reproduction sequence is:

1. Run algebraic validation:
   ```bash
   python scripts/teste01B_pattern_enrichment.py --meshes 8 12 16 20 --radii 1 2 3 --out-prefix teste01B_pattern_enrichment
   python scripts/teste01D_material_contrast.py --contrasts 10 100 1000 --jitter-values 0.0 0.1 --out-prefix teste01D_material_contrast
   ```

2. Run dynamic validation:
   ```bash
   python scripts/teste02_leapfrog_selected_hodge.py --meshes 8 12 16 20 --contrasts 1 10 1000 --jitter-values 0.0 0.1 --steps 1000 --cfl 0.25 --out-prefix teste02_leapfrog
   python scripts/teste02B_time_growth_cavity.py --mesh 20 --contrasts 1 1000 --jitter-values 0.0 0.1 --steps 4000 --sample-stride 25 --cfl 0.25 --out-prefix teste02B_cavity
   python scripts/teste02C_modal_frequency_cavity.py --mesh 20 --contrasts 1 1000 --jitter-values 0.0 0.1 --steps 8000 --cfl 0.25 --out-prefix teste02C_modal
   ```

3. Run selected-solve and scaling tests:
   ```bash
   python scripts/teste03A_selected_solve_hodge.py --meshes 20 30 40 60 --contrasts 1 1000 --jitter-values 0.0 0.1 --radii 3 --out-prefix teste03A
   python scripts/teste03B_scaling_selected_solve.py --meshes 80 100 120 --contrasts 1 1000 --jitter-values 0.0 0.1 --chunk-size 16 --nr 5 --matvec-reps 20 --out-prefix teste03B_scaling
   ```

4. Run large-cavity, LU-step, and CFL tests:
   ```bash
   python scripts/teste04_large_cavity_selected_solve.py --meshes 80 --contrasts 1 1000 --jitter-values 0.0 0.1 --steps 4000 --sample-stride 5 --chunk-size 16 --out-prefix teste04_large
   python scripts/teste05_compare_selected_inverse_vs_lu_step.py --meshes 80 --contrasts 1 1000 --jitter-values 0.0 0.1 --steps 4000 --sample-stride 5 --chunk-size 16 --out-prefix teste05_lu_compare
   python scripts/teste06_cfl_stability_sweep.py --mesh 80 --cases 1:0.0 1000:0.1 --cfl-values 0.25 0.40 0.50 0.60 0.70 0.80 0.90 1.00 1.05 1.10 --steps 4000 --sample-stride 5 --chunk-size 16 --out-prefix teste06_cfl
   ```

5. Generate figures after the CSV files are available:
   ```bash
   python scripts/figures_cpc.py
   ```

## Dependencies

See `requirements.txt`.

## License

This code is distributed under the BSD 3-Clause License. See the `LICENSE` file for details.

## Notes for public release

This is the reproducibility code for a manuscript being prepared for submission to *Computer Physics Communications*. The DOI of the eventual journal article should be added after acceptance. The accompanying numerical dataset has been deposited on Zenodo (dataset DOI reported by the depositor: **[10.5281/zenodo.24021616](https://doi.org/10.5281/zenodo.24021616)**). A separate versioned software release can also be archived on Zenodo to obtain a software-specific DOI.

## Archived data

The original result CSV files and figure source data are distributed separately in the [Zenodo dataset](https://doi.org/10.5281/zenodo.24021616), which contains 43 CSV files and eight PDF figures. Extract the dataset CSV files into a working directory before running `python scripts/figures_cpc.py`, or invoke each test with its documented `--out-prefix`. Numerical runtimes depend on hardware, compiler/library versions and threading environment.
