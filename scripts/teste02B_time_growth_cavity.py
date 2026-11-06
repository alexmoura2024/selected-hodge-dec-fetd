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
    Long-time error-growth analysis for selected inverse Hodge approximations in a two-dimensional PEC cavity.

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

import argparse
import time
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla

try:
    from teste01D_material_contrast import (
        TriMesh2D,
        build_structured_tri_mesh,
        assemble_hodge_h1_material,
        prune_sparse,
        radius_patterns_from_H,
        project_dense_inverse_to_pattern,
        diag_inverse,
        density,
    )
except ImportError as exc:
    raise ImportError(
        "Could not import teste01D_material_contrast.py. "
        "Place this script in the same folder do teste01D_material_contrast.py "
        "and run with --wdir pointing to that folder."
    ) from exc


ArrayOrSparse = Union[np.ndarray, sp.spmatrix]


                                                              
                           
                                                              

def triangle_area(coords: np.ndarray) -> float:
    """Helper routine used in the numerical experiments."""
    x0, y0 = coords[0]
    x1, y1 = coords[1]
    x2, y2 = coords[2]
    return 0.5 * ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))


def boundary_edge_mask(mesh: TriMesh2D, tol: float = 1.0e-12) -> np.ndarray:
    """Helper routine used in the numerical experiments."""
    p = mesh.points
    e = mesh.edges

    a = p[e[:, 0]]
    b = p[e[:, 1]]

    left = (np.abs(a[:, 0]) < tol) & (np.abs(b[:, 0]) < tol)
    right = (np.abs(a[:, 0] - 1.0) < tol) & (np.abs(b[:, 0] - 1.0) < tol)
    bottom = (np.abs(a[:, 1]) < tol) & (np.abs(b[:, 1]) < tol)
    top = (np.abs(a[:, 1] - 1.0) < tol) & (np.abs(b[:, 1] - 1.0) < tol)

    return left | right | bottom | top


def assemble_curl_and_m2(
    mesh: TriMesh2D,
    free_edges: np.ndarray,
    nu: float = 1.0,
) -> Tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray]:
    """Helper routine used in the numerical experiments."""
    nf = mesh.triangles.shape[0]

    g2l = -np.ones(mesh.edges.shape[0], dtype=np.int64)
    g2l[free_edges] = np.arange(len(free_edges), dtype=np.int64)

    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    areas = np.empty(nf, dtype=float)

    for t, tri in enumerate(mesh.triangles):
        coords = mesh.points[tri]
        area = triangle_area(coords)

        if area <= 0.0:
            raise ValueError(
                "Triangle with non-positive area. "
                "Reduce the jitter or check the mesh orientation."
            )

        areas[t] = area

        for k, eid in enumerate(mesh.tri_edges[t]):
            lid = g2l[int(eid)]
            if lid >= 0:
                rows.append(t)
                cols.append(int(lid))
                vals.append(float(mesh.tri_edge_signs[t, k]))

    C = sp.coo_matrix((vals, (rows, cols)), shape=(nf, len(free_edges))).tocsr()
    M2 = sp.diags(nu / areas, format="csr")

    return C, M2, areas


def gaussian_initial_b(
    mesh: TriMesh2D,
    areas: np.ndarray,
    center: Tuple[float, float] = (0.35, 0.50),
    sigma: float = 0.08,
) -> np.ndarray:
    """Helper routine used in the numerical experiments."""
    centroids = mesh.points[mesh.triangles].mean(axis=1)
    dx = centroids[:, 0] - center[0]
    dy = centroids[:, 1] - center[1]
    B = np.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))
    return B * areas


                                                              
                  
                                                              

def h_norm(x: np.ndarray, H: sp.csr_matrix) -> float:
    """Helper routine used in the numerical experiments."""
    val = float(x @ (H @ x))
    return float(np.sqrt(max(val, 0.0)))


def m2_norm(x: np.ndarray, M2: sp.csr_matrix) -> float:
    """Helper routine used in the numerical experiments."""
    val = float(x @ (M2 @ x))
    return float(np.sqrt(max(val, 0.0)))


def electromagnetic_energy(
    e: np.ndarray,
    b: np.ndarray,
    H: sp.csr_matrix,
    M2: sp.csr_matrix,
) -> float:
    """Helper routine used in the numerical experiments."""
    return 0.5 * float(e @ (H @ e)) + 0.5 * float(b @ (M2 @ b))


def apply_G(G: ArrayOrSparse, rhs: np.ndarray) -> np.ndarray:
    """Helper routine used in the numerical experiments."""
    return G @ rhs


def estimate_lambda_max_full(K: sp.csr_matrix, H: sp.csr_matrix) -> float:
    """Helper routine used in the numerical experiments."""
    val = spla.eigsh(
        K,
        M=H,
        k=1,
        which="LA",
        return_eigenvectors=False,
        tol=1.0e-8,
        maxiter=10000,
    )[0]
    return float(val)


                                                              
                      
                                                              

def build_inverse_methods(
    H: sp.csr_matrix,
    radii: Sequence[int],
    tol: float,
) -> Dict[str, ArrayOrSparse]:
    """Helper routine used in the numerical experiments."""
    H_dense = H.toarray()
    Hinv = np.linalg.inv(H_dense)

    methods: Dict[str, ArrayOrSparse] = {
        "G_full": Hinv,
        "G_diag": diag_inverse(H),
    }

    patterns = radius_patterns_from_H(H, radii=radii, tol=tol)

    for r in radii:
        methods[f"G_H_radius_{r}"] = project_dense_inverse_to_pattern(
            Hinv,
            patterns[r],
        )

    return methods


def method_nnz(G: ArrayOrSparse) -> int:
    """Helper routine used in the numerical experiments."""
    if sp.issparse(G):
        return int(G.nnz)
    return int(G.shape[0] * G.shape[1])


def method_density(G: ArrayOrSparse) -> float:
    """Helper routine used in the numerical experiments."""
    if sp.issparse(G):
        return float(density(G))
    return 1.0


def method_radius(name: str) -> int:
    """Helper routine used in the numerical experiments."""
    if name in ("G_full", "G_diag"):
        return 0
    return int(name.split("_")[-1])


                                                              
                      
                                                              

def linear_growth_slope(times: np.ndarray, values: np.ndarray) -> float:
    """Helper routine used in the numerical experiments."""
    mask = np.isfinite(times) & np.isfinite(values)
    if np.count_nonzero(mask) < 3:
        return np.nan

    t = times[mask]
    y = values[mask]

    if np.max(t) <= np.min(t):
        return np.nan

    coeff = np.polyfit(t, y, deg=1)
    return float(coeff[0])


def classify_growth(values: np.ndarray, tol_plateau: float = 0.10) -> str:
    """Helper routine used in the numerical experiments."""
    y = values[np.isfinite(values)]
    if y.size < 3:
        return "insufficient"

    ymax = float(np.max(y))
    yfinal = float(y[-1])

    if ymax <= 0.0:
        return "zero"

    ratio = yfinal / ymax
    imax = int(np.argmax(y))

    if ratio < 0.75:
        return "oscillatory_or_saturated"

    if imax >= int(0.8 * (y.size - 1)):
        return "increasing"

    if abs(1.0 - ratio) <= tol_plateau:
        return "plateau_like"

    return "mixed"


                                                              
                      
                                                              

def run_time_growth_case(
    nx: int,
    ny: int,
    contrast: float,
    jitter: float,
    radii: Sequence[int],
    steps: int,
    cfl: float,
    seed: int,
    tol: float,
    sample_stride: int,
    pulse_sigma: float,
    pulse_x: float,
    pulse_y: float,
    max_dense_n: int,
    blowup_tol: float,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """Helper routine used in the numerical experiments."""
    t_case0 = time.perf_counter()

    mesh = build_structured_tri_mesh(nx, ny, jitter=jitter, seed=seed)

    boundary = boundary_edge_mask(mesh)
    free_edges = np.where(~boundary)[0]

    H_full_edges = assemble_hodge_h1_material(
        mesh,
        eps_left=1.0,
        eps_right=float(contrast),
        interface_x=0.5,
    )

    H = prune_sparse(H_full_edges[free_edges, :][:, free_edges], tol=tol)
    n = H.shape[0]

    if n > max_dense_n:
        raise MemoryError(
            f"n={n} excede --max-dense-n={max_dense_n}. "
            "Reduza a malha ou aumente o limite conscientemente."
        )

    C, M2, areas = assemble_curl_and_m2(mesh, free_edges, nu=1.0)
    K = (C.T @ M2 @ C).tocsr()

    lambda_max = estimate_lambda_max_full(K, H)
    dt_cfl = 2.0 / np.sqrt(lambda_max)
    dt = cfl * dt_cfl

    methods = build_inverse_methods(H, radii=radii, tol=tol)

    b0 = gaussian_initial_b(
        mesh,
        areas,
        center=(pulse_x, pulse_y),
        sigma=pulse_sigma,
    )
    e0 = np.zeros(n, dtype=float)

    CTM2 = C.T @ M2
    ref_name = "G_full"

    states: Dict[str, dict] = {}

    for name in methods:
        E0 = electromagnetic_energy(e0, b0, H, M2)
        states[name] = {
            "e": e0.copy(),
            "b": b0.copy(),
            "E0": float(E0),
            "stable": True,
            "first_unstable_step": -1,
        }

    sample_steps = set(range(0, steps + 1, sample_stride))
    sample_steps.add(steps)

    time_rows: List[dict] = []

    def sample(step: int) -> None:
        time_value = step * dt
        e_ref = states[ref_name]["e"]
        b_ref = states[ref_name]["b"]

        e_ref_norm = max(h_norm(e_ref, H), 1.0e-30)
        b_ref_norm = max(m2_norm(b_ref, M2), 1.0e-30)

        for name, G in methods.items():
            st = states[name]
            e = st["e"]
            b = st["b"]

            En = electromagnetic_energy(e, b, H, M2)
            Erel = En / st["E0"] if st["E0"] > 0.0 else np.nan
            drift = abs(Erel - 1.0) if np.isfinite(Erel) else np.nan

            if name == ref_name:
                rel_e = 0.0
                rel_b = 0.0
            else:
                rel_e = h_norm(e - e_ref, H) / e_ref_norm
                rel_b = m2_norm(b - b_ref, M2) / b_ref_norm

            nnz_g = method_nnz(G)

            time_rows.append({
                "mesh": f"{nx}x{ny}",
                "nx": int(nx),
                "ny": int(ny),
                "contrast": float(contrast),
                "jitter": float(jitter),
                "method": name,
                "radius": int(method_radius(name)),
                "step": int(step),
                "time": float(time_value),
                "dt": float(dt),
                "cfl": float(cfl),
                "NE_free": int(n),
                "NF": int(mesh.triangles.shape[0]),
                "nnz_H": int(H.nnz),
                "rho_H": float(density(H)),
                "nnz_G": int(nnz_g),
                "rho_G": float(method_density(G)),
                "nnzG_over_nnzH": float(nnz_g / H.nnz),
                "rel_e_error_vs_full": float(rel_e),
                "rel_b_error_vs_full": float(rel_b),
                "energy": float(En),
                "energy_relative": float(Erel),
                "energy_drift": float(drift),
                "stable": bool(st["stable"]),
                "first_unstable_step": int(st["first_unstable_step"]),
                "lambda_max_full": float(lambda_max),
                "dt_cfl": float(dt_cfl),
            })

    sample(0)

    t_dyn0 = time.perf_counter()

    for it in range(1, steps + 1):
        for name, G in methods.items():
            st = states[name]

            if not st["stable"]:
                continue

            e = st["e"]
            b = st["b"]

            b_new = b - dt * (C @ e)
            rhs = CTM2 @ b_new
            e_new = e + dt * apply_G(G, rhs)

            st["e"] = e_new
            st["b"] = b_new

            En = electromagnetic_energy(e_new, b_new, H, M2)
            Erel = En / st["E0"] if st["E0"] > 0.0 else np.nan
            drift = abs(Erel - 1.0) if np.isfinite(Erel) else np.inf

            if (
                not np.isfinite(drift)
                or drift > blowup_tol
                or not np.all(np.isfinite(e_new))
                or not np.all(np.isfinite(b_new))
            ):
                st["stable"] = False
                st["first_unstable_step"] = int(it)

        if it in sample_steps:
            sample(it)

    runtime_dyn = time.perf_counter() - t_dyn0
    runtime_total = time.perf_counter() - t_case0

    df_ts = pd.DataFrame(time_rows)

    summary_rows: List[dict] = []
    growth_rows: List[dict] = []

    for name in methods.keys():
        sub = df_ts[df_ts["method"] == name].copy()

        times = sub["time"].to_numpy(dtype=float)
        ee = sub["rel_e_error_vs_full"].to_numpy(dtype=float)
        eb = sub["rel_b_error_vs_full"].to_numpy(dtype=float)
        drift = sub["energy_drift"].to_numpy(dtype=float)

        slope_e = linear_growth_slope(times, ee)
        slope_b = linear_growth_slope(times, eb)
        slope_drift = linear_growth_slope(times, drift)

        half = max(0, len(times) // 2)
        slope_e_second_half = linear_growth_slope(times[half:], ee[half:])
        slope_b_second_half = linear_growth_slope(times[half:], eb[half:])

        growth_class_e = classify_growth(ee)
        growth_class_b = classify_growth(eb)

        max_rel_e = float(np.nanmax(ee))
        final_rel_e = float(ee[-1])
        max_rel_b = float(np.nanmax(eb))
        final_rel_b = float(eb[-1])
        max_drift = float(np.nanmax(drift))
        final_drift = float(drift[-1])

        stable_all = bool(sub["stable"].all())
        first_unstable = int(sub["first_unstable_step"].max())

        nnzG_over_nnzH = float(sub["nnzG_over_nnzH"].iloc[0])

        summary_rows.append({
            "mesh": f"{nx}x{ny}",
            "nx": int(nx),
            "ny": int(ny),
            "contrast": float(contrast),
            "jitter": float(jitter),
            "method": name,
            "radius": int(method_radius(name)),
            "NE_free": int(n),
            "NF": int(mesh.triangles.shape[0]),
            "nnz_H": int(H.nnz),
            "nnzG_over_nnzH": float(nnzG_over_nnzH),
            "steps": int(steps),
            "sample_stride": int(sample_stride),
            "dt": float(dt),
            "dt_cfl": float(dt_cfl),
            "cfl": float(cfl),
            "t_final": float(steps * dt),
            "lambda_max_full": float(lambda_max),
            "max_rel_e_error_vs_full": max_rel_e,
            "final_rel_e_error_vs_full": final_rel_e,
            "max_rel_b_error_vs_full": max_rel_b,
            "final_rel_b_error_vs_full": final_rel_b,
            "max_energy_drift": max_drift,
            "final_energy_drift": final_drift,
            "growth_slope_e": float(slope_e),
            "growth_slope_b": float(slope_b),
            "growth_slope_energy_drift": float(slope_drift),
            "growth_slope_e_second_half": float(slope_e_second_half),
            "growth_slope_b_second_half": float(slope_b_second_half),
            "growth_class_e": growth_class_e,
            "growth_class_b": growth_class_b,
            "stable": stable_all,
            "first_unstable_step": first_unstable,
            "runtime_dynamics_s": float(runtime_dyn),
            "runtime_total_case_s": float(runtime_total),
        })

        growth_rows.append({
            "mesh": f"{nx}x{ny}",
            "contrast": float(contrast),
            "jitter": float(jitter),
            "method": name,
            "radius": int(method_radius(name)),
            "growth_slope_e": float(slope_e),
            "growth_slope_e_second_half": float(slope_e_second_half),
            "growth_slope_b": float(slope_b),
            "growth_slope_b_second_half": float(slope_b_second_half),
            "growth_slope_energy_drift": float(slope_drift),
            "growth_class_e": growth_class_e,
            "growth_class_b": growth_class_b,
            "max_rel_e_error_vs_full": max_rel_e,
            "final_rel_e_error_vs_full": final_rel_e,
            "max_rel_b_error_vs_full": max_rel_b,
            "final_rel_b_error_vs_full": final_rel_b,
            "max_energy_drift": max_drift,
            "final_energy_drift": final_drift,
            "stable": stable_all,
        })

    return time_rows, summary_rows, growth_rows


                                                              
     
                                                              

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Teste 02B — evolução temporal do erro na cavidade PEC 2D."
    )

    p.add_argument(
        "--mesh",
        type=int,
        default=20,
        help="Malha quadrada nx=ny. Default: 20",
    )

    p.add_argument(
        "--radii",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="Raios topológicos. Default: 1 2 3",
    )

    p.add_argument(
        "--contrasts",
        nargs="+",
        type=float,
        default=[1.0, 1000.0],
        help="Contrasts eps_right. Default: 1 1000",
    )

    p.add_argument(
        "--jitter-values",
        nargs="+",
        type=float,
        default=[0.0, 0.1],
        help="Jitter values. Default: 0.0 0.1",
    )

    p.add_argument(
        "--steps",
        type=int,
        default=4000,
        help="Número de passos de tempo. Default: 4000",
    )

    p.add_argument(
        "--cfl",
        type=float,
        default=0.25,
        help="Fração do limite dt_cfl=2/sqrt(lambda_max). Default: 0.25",
    )

    p.add_argument(
        "--sample-stride",
        type=int,
        default=25,
        help="Intervalo de amostragem temporal. Default: 25",
    )

    p.add_argument(
        "--pulse-sigma",
        type=float,
        default=0.08,
        help="Largura da gaussiana inicial magnética. Default: 0.08",
    )

    p.add_argument(
        "--pulse-x",
        type=float,
        default=0.35,
        help="Centro x da gaussiana inicial. Default: 0.35",
    )

    p.add_argument(
        "--pulse-y",
        type=float,
        default=0.50,
        help="Centro y da gaussiana inicial. Default: 0.50",
    )

    p.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed para jitter. Default: 1234",
    )

    p.add_argument(
        "--tol",
        type=float,
        default=1.0e-14,
        help="Structural tolerance. Default: 1e-14",
    )

    p.add_argument(
        "--max-dense-n",
        type=int,
        default=2500,
        help="Dimensão máxima para inversão densa. Default: 2500",
    )

    p.add_argument(
        "--blowup-tol",
        type=float,
        default=1.0e6,
        help="Limite de drift para declarar blow-up. Default: 1e6",
    )

    p.add_argument(
        "--out-prefix",
        type=str,
        default="teste02B_cavity",
        help="Output CSV prefix.",
    )

    return p.parse_args()


                                                              
      
                                                              

def main() -> None:
    args = parse_args()

    radii = sorted(set(args.radii))
    nx = int(args.mesh)
    ny = int(args.mesh)

    print("\nTeste 02B — evolução temporal do erro na cavidade PEC 2D")
    print("=" * 100)
    print(f"mesh          : {nx}x{ny}")
    print(f"radii         : {radii}")
    print(f"contrasts     : {args.contrasts}")
    print(f"jitter_values : {args.jitter_values}")
    print(f"steps         : {args.steps}")
    print(f"cfl           : {args.cfl}")
    print(f"sample_stride : {args.sample_stride}")
    print(f"pulse_sigma   : {args.pulse_sigma}")
    print(f"tol           : {args.tol:.1e}")
    print(f"max_dense_n   : {args.max_dense_n}")
    print(f"out_prefix    : {args.out_prefix}")
    print("=" * 100)

    all_ts: List[dict] = []
    all_summary: List[dict] = []
    all_growth: List[dict] = []

    for contrast in args.contrasts:
        for jitter in args.jitter_values:
            print(f"\nRodando caso: contrast={contrast:g}, jitter={jitter:g}")
            t0 = time.perf_counter()

            try:
                ts_rows, summary_rows, growth_rows = run_time_growth_case(
                    nx=nx,
                    ny=ny,
                    contrast=float(contrast),
                    jitter=float(jitter),
                    radii=radii,
                    steps=int(args.steps),
                    cfl=float(args.cfl),
                    seed=int(args.seed),
                    tol=float(args.tol),
                    sample_stride=int(args.sample_stride),
                    pulse_sigma=float(args.pulse_sigma),
                    pulse_x=float(args.pulse_x),
                    pulse_y=float(args.pulse_y),
                    max_dense_n=int(args.max_dense_n),
                    blowup_tol=float(args.blowup_tol),
                )

            except Exception as exc:
                print(f"  ERRO: {exc}")
                continue

            all_ts.extend(ts_rows)
            all_summary.extend(summary_rows)
            all_growth.extend(growth_rows)

            print(f"Concluído em {time.perf_counter() - t0:.3f} s")

            df_s_case = pd.DataFrame(summary_rows)
            cols_print = [
                "method",
                "max_rel_e_error_vs_full",
                "final_rel_e_error_vs_full",
                "max_rel_b_error_vs_full",
                "max_energy_drift",
                "growth_slope_e",
                "growth_class_e",
                "stable",
            ]
            print(df_s_case[cols_print].to_string(index=False))

            r3 = df_s_case[df_s_case["method"] == "G_H_radius_3"].iloc[0]
            diag = df_s_case[df_s_case["method"] == "G_diag"].iloc[0]

            ratio = (
                diag["max_rel_e_error_vs_full"] /
                max(r3["max_rel_e_error_vs_full"], 1.0e-30)
            )

            print(
                f"  ganho diag/r3 em erro máximo e: {ratio:.3f} | "
                f"r3 class={r3['growth_class_e']} | "
                f"r3 stable={bool(r3['stable'])}"
            )

    if not all_ts:
        print("\nNenhum resultado foi gerado.")
        return

    df_ts = pd.DataFrame(all_ts)
    df_summary = pd.DataFrame(all_summary)
    df_growth = pd.DataFrame(all_growth)

    ts_path = f"{args.out_prefix}_time_series.csv"
    summary_path = f"{args.out_prefix}_summary.csv"
    growth_path = f"{args.out_prefix}_growth_rates.csv"

    df_ts.to_csv(ts_path, index=False)
    df_summary.to_csv(summary_path, index=False)
    df_growth.to_csv(growth_path, index=False)

    print("\nArquivos gerados:")
    print(f"  {ts_path}")
    print(f"  {summary_path}")
    print(f"  {growth_path}")

    print("\nSíntese rápida de G_H_radius_3:")
    r3_all = df_summary[df_summary["method"] == "G_H_radius_3"].copy()
    cols = [
        "mesh",
        "contrast",
        "jitter",
        "max_rel_e_error_vs_full",
        "final_rel_e_error_vs_full",
        "max_rel_b_error_vs_full",
        "max_energy_drift",
        "growth_slope_e",
        "growth_class_e",
        "nnzG_over_nnzH",
        "stable",
    ]
    print(r3_all[cols].to_string(index=False))


if __name__ == "__main__":
    main()
