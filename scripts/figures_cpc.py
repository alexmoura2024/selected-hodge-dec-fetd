# -*- coding: utf-8 -*-
"""
================================================================================
Topological Selected-Solve Approximation of Inverse Hodge Operators
for Explicit DEC/FETD Maxwell Simulations
================================================================================

This script is part of the reproducibility package for the manuscript:

    "A Topological Selected-Solve Approximation of Inverse Hodge Operators
     for Explicit DEC/FETD Maxwell Simulations"

Authors
-------
    A. S. Moura
    W. G. Facco

Corresponding author
--------------------
    A. S. Moura

Purpose of this script
----------------------
    Generate publication-ready figures for the selected inverse Hodge manuscript.

Research context
----------------
The manuscript studies sparse, Hodge-aware selected approximations of the
inverse electric Whitney--Hodge matrix in explicit DEC/FETD Maxwell simulations.
The central approximation is

    G_{H,r} = P_{S_{H^r}}(H^{-1}),

with emphasis on the radius-three operator G_{H,3} and its selected-solve
construction without forming the dense inverse H^{-1}.

Reproducibility note
--------------------
The code was developed as research code for the numerical experiments reported
in the manuscript. Default parameters reproduce lightweight or representative
runs; larger runs may require substantial memory and runtime.

License
-------
BSD 3-Clause License. See the LICENSE file for details.

Citation
--------
If this code is used, please cite the associated manuscript and the GitHub
repository/DOI once available.
================================================================================
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


                                                              
       
                                                              

DATA_DIR = Path(".")
OUT_DIR = Path("figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


                                                              
                          
                                                              

mpl.rcParams.update({
    "figure.figsize": (6.8, 4.4),
    "figure.dpi": 120,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def savefig(fig: plt.Figure, name: str) -> None:
    """Helper routine used in the numerical experiments."""
    pdf_path = OUT_DIR / f"{name}.pdf"
    png_path = OUT_DIR / f"{name}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


def read_csv(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def clean_method_label(method: str) -> str:
    if method == "G_diag":
        return r"$G_{\mathrm{diag}}$"
    if method == "G_full":
        return r"$G_{\mathrm{full}}$"
    if method == "LU_step_reference":
        return "LU-step"
    if method == "G_H_radius_1":
        return r"$G_{H,1}$"
    if method == "G_H_radius_2":
        return r"$G_{H,2}$"
    if method == "G_H_radius_3":
        return r"$G_{H,3}$"
    if method == "G_H_radius_3_selected_solve":
        return r"$G_{H,3}$"
    return method.replace("_", r"\_")


def case_label(contrast: float, jitter: float) -> str:
    if float(contrast).is_integer():
        c = int(contrast)
    else:
        c = contrast
    return rf"$\varepsilon_r={c}$, jitter={jitter:g}"


                                                              
                                      
                                                              

def fig01_conceptual_pattern() -> None:
    """Helper routine used in the numerical experiments."""
    n = 36

                                      
    H = np.zeros((n, n), dtype=int)
    for i in range(n):
        H[i, i] = 1
        for d in (1, 2):
            if i + d < n:
                H[i, i + d] = 1
                H[i + d, i] = 1

                                                                           
    for i in range(0, n - 6, 6):
        H[i, i + 6] = 1
        H[i + 6, i] = 1
        if i + 7 < n:
            H[i + 1, i + 7] = 1
            H[i + 7, i + 1] = 1

    P = (H > 0).astype(int)
    P3 = ((P @ P @ P) > 0).astype(int)

                                                                   
    rng = np.random.default_rng(1234)
    G = P3 * (rng.random((n, n)) > 0.55)
    G = np.maximum(G, G.T)
    np.fill_diagonal(G, 1)

    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.8), constrained_layout=True)

    mats = [P, P3, G]
    labels = [
        r"$\operatorname{pattern}(H)$",
        r"$\mathcal{S}_{H^3}$",
        r"$\operatorname{pattern}(G_{H,3})$",
    ]

    for ax, mat, lab in zip(axes, mats, labels):
        ax.spy(mat, markersize=2.2)
        ax.set_xlabel(lab)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    savefig(fig, "fig01_topological_pattern_concept")


                                                              
                                     
                                                              

def fig02_action_error_vs_radius() -> None:
    df = read_csv("teste01B_pattern_enrichment_comparison.csv")

    df = df[df["method"].isin([
        "G_diag",
        "G_H_radius_1",
        "G_H_radius_2",
        "G_H_radius_3",
    ])].copy()

                                                        
    finest_nx = int(df["nx"].max())
    g = df[df["nx"] == finest_nx].copy()

    order = ["G_diag", "G_H_radius_1", "G_H_radius_2", "G_H_radius_3"]
    x = np.arange(len(order))
    y = [float(g[g["method"] == m]["E_action"].iloc[0]) for m in order]

    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    ax.plot(x, y, marker="o")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([
        r"$G_{\mathrm{diag}}$",
        r"$G_{H,1}$",
        r"$G_{H,2}$",
        r"$G_{H,3}$",
    ])
    ax.set_ylabel(r"Action error $E_{\mathrm{act}}$")
    ax.set_xlabel("Inverse Hodge approximation")
    ax.grid(True, which="both", alpha=0.3)

    savefig(fig, "fig02_action_error_vs_radius")


                                                              
                                                  
                                                              

def fig03_material_jitter_robustness() -> None:
    df = read_csv("teste01D_material_contrast_summary.csv")

                                   
    finest_nx = int(df["nx"].max())
    df = df[df["nx"] == finest_nx].copy()

    fig, ax = plt.subplots(figsize=(6.4, 4.2))

    for jitter, g in df.groupby("jitter"):
        g = g.sort_values("contrast")
        ax.plot(
            g["contrast"],
            g["r3_E_action"],
            marker="o",
            label=rf"jitter={jitter:g}",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Material contrast $\varepsilon_r$")
    ax.set_ylabel(r"$E_{\mathrm{act}}(G_{H,3})$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=True)

    savefig(fig, "fig03_material_jitter_robustness")


                                                              
                                    
                                                              

def fig04_large_cavity_energy_drift() -> None:
    df = read_csv("teste04_large_energy_series.csv")

    methods = ["G_diag", "G_H_radius_3_selected_solve"]
    df = df[df["method"].isin(methods)].copy()

    fig, ax = plt.subplots(figsize=(6.8, 4.4))

                                                                       
    selected_cases = [
        (1.0, 0.0),
        (1000.0, 0.1),
    ]

    for contrast, jitter in selected_cases:
        for method in methods:
            g = df[
                (df["contrast"] == contrast)
                & (df["jitter"] == jitter)
                & (df["method"] == method)
            ].copy()

            if g.empty:
                continue

            g = g.sort_values("time")
            ax.plot(
                g["time"],
                g["energy_drift"],
                label=f"{clean_method_label(method)}; {case_label(contrast, jitter)}",
            )

    ax.set_yscale("log")
    ax.set_xlabel(r"Time")
    ax.set_ylabel(r"Energy drift $D_E(t)$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=True, ncol=1)

    savefig(fig, "fig04_large_cavity_energy_drift")


                                                              
                                          
                                                              

def fig05_speedup_vs_lu() -> None:
    df = read_csv("teste05_lu_compare_summary.csv")

    rows = []

    group_cols = ["contrast", "jitter"]
    for (contrast, jitter), g in df.groupby(group_cols):
        lu = g[g["method"] == "LU_step_reference"]
        if lu.empty:
            continue

        t_lu = float(lu["runtime_update_per_step_s"].iloc[0])

        for _, row in g.iterrows():
            method = row["method"]
            if method == "LU_step_reference":
                continue

            t_method = float(row["runtime_update_per_step_s"])
            rows.append({
                "contrast": contrast,
                "jitter": jitter,
                "method": method,
                "speedup": t_lu / t_method,
            })

    speed = pd.DataFrame(rows)
    speed = speed[speed["method"].isin([
        "G_diag",
        "G_H_radius_3_selected_solve",
    ])].copy()

    labels = []
    x = np.arange(len(speed))
    y = speed["speedup"].to_numpy()

    for _, row in speed.iterrows():
        labels.append(
            clean_method_label(row["method"])
            + "\n"
            + rf"$\varepsilon_r={int(row['contrast'])}$, jit={row['jitter']:g}"
        )

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.bar(x, y)
    ax.axhline(1.0, linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Per-step speedup versus LU-step")
    ax.grid(True, axis="y", alpha=0.3)

    savefig(fig, "fig05_speedup_vs_lu_step")


                                                              
                              
                                                              

def fig06_cfl_stability() -> None:
    df = read_csv("teste06_cfl_summary.csv")

    methods = ["G_H_radius_3_selected_solve", "LU_step_reference"]
    df = df[df["method"].isin(methods)].copy()

    fig, ax = plt.subplots(figsize=(6.8, 4.4))

    for (contrast, jitter, method), g in df.groupby(["contrast", "jitter", "method"]):
        g = g.sort_values("cfl")
        ax.plot(
            g["cfl"],
            g["max_energy_drift"],
            marker="o",
            label=f"{clean_method_label(method)}; {case_label(contrast, jitter)}",
        )

    ax.axhline(0.10, linestyle="--", linewidth=1.0, label="practical threshold")
    ax.set_yscale("log")
    ax.set_xlabel("CFL")
    ax.set_ylabel(r"Maximum energy drift")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=True)

    savefig(fig, "fig06_cfl_energy_stability")


                                                              
                                         
                                                              

def fig07_critical_cfl() -> None:
    df = read_csv("teste06_cfl_critical_cfl.csv")

    df = df[df["method"].isin([
        "G_diag",
        "G_H_radius_3_selected_solve",
        "LU_step_reference",
    ])].copy()

    df["practical"] = df["max_cfl_practically_stable"].fillna(0.0)
    df["numerical"] = df["max_cfl_numerically_stable"].fillna(0.0)

    labels = []
    for _, row in df.iterrows():
        labels.append(
            clean_method_label(row["method"])
            + "\n"
            + rf"$\varepsilon_r={int(row['contrast'])}$, jit={row['jitter']:g}"
        )

    x = np.arange(len(df))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.bar(x - width / 2, df["numerical"], width, label="Numerically stable")
    ax.bar(x + width / 2, df["practical"], width, label="Practically stable")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Maximum CFL")
    ax.set_ylim(0.0, 1.2)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=True)

    savefig(fig, "fig07_critical_cfl")


                                                              
                                 
                                                              

def fig08_field_error_vs_cfl() -> None:
    df = read_csv("teste06_cfl_summary.csv")

    df = df[df["method"].isin([
        "G_diag",
        "G_H_radius_3_selected_solve",
    ])].copy()

    df = df[df["stable"] == True].copy()

    fig, ax = plt.subplots(figsize=(6.8, 4.4))

    for (contrast, jitter, method), g in df.groupby(["contrast", "jitter", "method"]):
        g = g.sort_values("cfl")
        ax.plot(
            g["cfl"],
            g["max_rel_e_error_vs_LU"],
            marker="o",
            label=f"{clean_method_label(method)}; {case_label(contrast, jitter)}",
        )

    ax.set_yscale("log")
    ax.set_xlabel("CFL")
    ax.set_ylabel(r"$\max_t E_e(t)$ versus LU-step")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=True)

    savefig(fig, "fig08_field_error_vs_cfl")


                                                              
      
                                                              

def main() -> None:
    print("Generating publication-ready figures...")
    print(f"Input directory : {DATA_DIR.resolve()}")
    print(f"Output directory: {OUT_DIR.resolve()}")

    tasks = [
        fig01_conceptual_pattern,
        fig02_action_error_vs_radius,
        fig03_material_jitter_robustness,
        fig04_large_cavity_energy_drift,
        fig05_speedup_vs_lu,
        fig06_cfl_stability,
        fig07_critical_cfl,
        fig08_field_error_vs_cfl,
    ]

    for task in tasks:
        try:
            task()
        except FileNotFoundError as exc:
            print(f"Skipped {task.__name__}: {exc}")
        except Exception as exc:
            print(f"Error in {task.__name__}: {type(exc).__name__}: {exc}")

    print("Done.")


if __name__ == "__main__":
    main()
