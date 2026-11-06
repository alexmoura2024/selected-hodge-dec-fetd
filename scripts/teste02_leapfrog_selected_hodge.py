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
    Dynamic leapfrog validation of selected inverse Hodge approximations in a two-dimensional PEC cavity.

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

    C = sp.coo_matrix(
        (vals, (rows, cols)),
        shape=(nf, len(free_edges))
    ).tocsr()

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
        return density(G)

    return 1.0


def method_radius(name: str) -> int:
    """Helper routine used in the numerical experiments."""
    if name in ("G_full", "G_diag"):
        return 0

    return int(name.split("_")[-1])


                                                              
                    
                                                              

def run_leapfrog_case(
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
) -> Tuple[List[dict], dict]:
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
            "max_energy_drift": 0.0,
            "final_energy_drift": 0.0,
            "max_rel_e_error_vs_full": 0.0,
            "final_rel_e_error_vs_full": 0.0,
            "max_rel_b_error_vs_full": 0.0,
            "final_rel_b_error_vs_full": 0.0,
            "stable": True,
        }

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

            if st["E0"] > 0.0 and np.isfinite(En):
                drift = abs(En / st["E0"] - 1.0)

                st["final_energy_drift"] = float(drift)
                st["max_energy_drift"] = max(
                    st["max_energy_drift"],
                    float(drift),
                )
            else:
                st["stable"] = False

            if (
                not np.all(np.isfinite(e_new))
                or not np.all(np.isfinite(b_new))
                or st["max_energy_drift"] > blowup_tol
            ):
                st["stable"] = False

        if it % sample_stride == 0 or it == steps:
            e_ref = states[ref_name]["e"]
            b_ref = states[ref_name]["b"]

            e_ref_norm = max(h_norm(e_ref, H), 1.0e-30)
            b_ref_norm = max(m2_norm(b_ref, M2), 1.0e-30)

            for name, st in states.items():
                if name == ref_name:
                    continue

                rel_e = h_norm(st["e"] - e_ref, H) / e_ref_norm
                rel_b = m2_norm(st["b"] - b_ref, M2) / b_ref_norm

                st["final_rel_e_error_vs_full"] = float(rel_e)
                st["final_rel_b_error_vs_full"] = float(rel_b)

                st["max_rel_e_error_vs_full"] = max(
                    st["max_rel_e_error_vs_full"],
                    float(rel_e),
                )

                st["max_rel_b_error_vs_full"] = max(
                    st["max_rel_b_error_vs_full"],
                    float(rel_b),
                )

    t_dyn = time.perf_counter() - t_dyn0
    total_time = time.perf_counter() - t_case0

    rows: List[dict] = []

    full_drift = states[ref_name]["max_energy_drift"]

    for name, G in methods.items():
        st = states[name]

        nnz_g = method_nnz(G)
        rho_g = method_density(G)

        En_final = electromagnetic_energy(st["e"], st["b"], H, M2)
        Erel_final = En_final / st["E0"] if st["E0"] > 0.0 else np.nan

        rows.append({
            "mesh": f"{nx}x{ny}",
            "nx": int(nx),
            "ny": int(ny),
            "jitter": float(jitter),
            "contrast": float(contrast),
            "method": name,
            "radius": int(method_radius(name)),
            "NV": int(mesh.points.shape[0]),
            "NE_total": int(mesh.edges.shape[0]),
            "NE_free": int(n),
            "NF": int(mesh.triangles.shape[0]),
            "nnz_H": int(H.nnz),
            "rho_H": float(density(H)),
            "nnz_G": int(nnz_g),
            "rho_G": float(rho_g),
            "nnzG_over_nnzH": float(nnz_g / H.nnz),
            "lambda_max_full": float(lambda_max),
            "dt_cfl": float(dt_cfl),
            "dt": float(dt),
            "cfl": float(cfl),
            "steps": int(steps),
            "t_final": float(steps * dt),
            "sample_stride": int(sample_stride),
            "pulse_sigma": float(pulse_sigma),
            "energy_initial": float(st["E0"]),
            "energy_final": float(En_final),
            "energy_final_relative": float(Erel_final),
            "final_energy_drift": float(st["final_energy_drift"]),
            "max_energy_drift": float(st["max_energy_drift"]),
            "excess_max_energy_drift_vs_full": float(
                st["max_energy_drift"] - full_drift
            ),
            "final_rel_e_error_vs_full": float(
                st["final_rel_e_error_vs_full"]
            ),
            "max_rel_e_error_vs_full": float(
                st["max_rel_e_error_vs_full"]
            ),
            "final_rel_b_error_vs_full": float(
                st["final_rel_b_error_vs_full"]
            ),
            "max_rel_b_error_vs_full": float(
                st["max_rel_b_error_vs_full"]
            ),
            "stable": bool(st["stable"]),
            "runtime_dynamics_s": float(t_dyn),
            "runtime_total_case_s": float(total_time),
        })

    non_full = [r for r in rows if r["method"] != "G_full"]

    best = min(non_full, key=lambda z: z["max_rel_e_error_vs_full"])

    r3 = next((r for r in rows if r["method"] == "G_H_radius_3"), None)
    diag = next((r for r in rows if r["method"] == "G_diag"), None)

    summary = {
        "mesh": f"{nx}x{ny}",
        "nx": int(nx),
        "ny": int(ny),
        "jitter": float(jitter),
        "contrast": float(contrast),
        "NE_free": int(n),
        "NF": int(mesh.triangles.shape[0]),
        "nnz_H": int(H.nnz),
        "lambda_max_full": float(lambda_max),
        "dt_cfl": float(dt_cfl),
        "dt": float(dt),
        "cfl": float(cfl),
        "steps": int(steps),
        "t_final": float(steps * dt),
        "full_max_energy_drift": float(full_drift),
        "best_method": best["method"],
        "best_radius": int(best["radius"]),
        "best_max_rel_e_error_vs_full": float(
            best["max_rel_e_error_vs_full"]
        ),
        "best_final_rel_e_error_vs_full": float(
            best["final_rel_e_error_vs_full"]
        ),
        "best_max_energy_drift": float(best["max_energy_drift"]),
        "best_stable": bool(best["stable"]),
        "diag_max_rel_e_error_vs_full": float(
            diag["max_rel_e_error_vs_full"]
        ) if diag else np.nan,
        "diag_max_energy_drift": float(
            diag["max_energy_drift"]
        ) if diag else np.nan,
        "r3_max_rel_e_error_vs_full": float(
            r3["max_rel_e_error_vs_full"]
        ) if r3 else np.nan,
        "r3_final_rel_e_error_vs_full": float(
            r3["final_rel_e_error_vs_full"]
        ) if r3 else np.nan,
        "r3_max_rel_b_error_vs_full": float(
            r3["max_rel_b_error_vs_full"]
        ) if r3 else np.nan,
        "r3_final_rel_b_error_vs_full": float(
            r3["final_rel_b_error_vs_full"]
        ) if r3 else np.nan,
        "r3_max_energy_drift": float(
            r3["max_energy_drift"]
        ) if r3 else np.nan,
        "r3_excess_max_energy_drift_vs_full": float(
            r3["excess_max_energy_drift_vs_full"]
        ) if r3 else np.nan,
        "r3_nnzG_over_nnzH": float(
            r3["nnzG_over_nnzH"]
        ) if r3 else np.nan,
        "r3_stable": bool(r3["stable"]) if r3 else False,
        "runtime_total_case_s": float(total_time),
    }

    return rows, summary


                                                              
     
                                                              

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Teste 02 — Leapfrog explícito com inversa seletiva "
            "topológica da Hodge."
        )
    )

    p.add_argument(
        "--meshes",
        nargs="+",
        type=int,
        default=[8, 12, 16, 20],
        help="Square meshes nx=ny. Default: 8 12 16 20",
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
        default=[1.0, 10.0, 1000.0],
        help="Contrasts eps_right. Default: 1 10 1000",
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
        default=1000,
        help="Número de passos de tempo. Default: 1000",
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
        help="Intervalo de amostragem para erro contra G_full. Default: 25",
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
        help="Random seed para malha perturbada. Default: 1234",
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
        help="Limite de drift de energia para declarar blow-up. Default: 1e6",
    )

    p.add_argument(
        "--out-prefix",
        type=str,
        default="teste02_leapfrog",
        help="Output CSV prefix.",
    )

    return p.parse_args()


                                                              
      
                                                              

def main() -> None:
    args = parse_args()

    radii = sorted(set(args.radii))

    print("\nTeste 02 — Leapfrog explícito com inversa seletiva topológica da Hodge")
    print("=" * 100)
    print(f"meshes        : {args.meshes}")
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

    all_rows: List[dict] = []
    summaries: List[dict] = []

    for contrast in args.contrasts:
        for jitter in args.jitter_values:
            print(f"\n### Contraste eps_right={contrast:g} | jitter={jitter:g} ###")

            for nx in args.meshes:
                print(f"\nRodando malha {nx}x{nx} ...")

                t0 = time.perf_counter()

                try:
                    rows, summary = run_leapfrog_case(
                        nx=nx,
                        ny=nx,
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

                all_rows.extend(rows)
                summaries.append(summary)

                print(f"Concluído em {time.perf_counter() - t0:.3f} s")

                print(
                    f"  dt={summary['dt']:.4e} | "
                    f"t_final={summary['t_final']:.4e} | "
                    f"full drift={summary['full_max_energy_drift']:.3e}"
                )

                for r in rows:
                    print(
                        f"  {r['method']:<14s} "
                        f"nnzG/nnzH={r['nnzG_over_nnzH']:.2f} "
                        f"maxEe={r['max_rel_e_error_vs_full']:.3e} "
                        f"finalEe={r['final_rel_e_error_vs_full']:.3e} "
                        f"maxEb={r['max_rel_b_error_vs_full']:.3e} "
                        f"drift={r['max_energy_drift']:.3e} "
                        f"stable={r['stable']}"
                    )

                print(
                    f"  melhor não-full: {summary['best_method']} | "
                    f"maxEe={summary['best_max_rel_e_error_vs_full']:.3e} | "
                    f"stable={summary['best_stable']}"
                )

                print(
                    f"  r=3: maxEe={summary['r3_max_rel_e_error_vs_full']:.3e} | "
                    f"finalEe={summary['r3_final_rel_e_error_vs_full']:.3e} | "
                    f"drift={summary['r3_max_energy_drift']:.3e} | "
                    f"stable={summary['r3_stable']}"
                )

    if not all_rows:
        print("\nNenhum resultado foi gerado.")
        return

    df = pd.DataFrame(all_rows)
    dfs = pd.DataFrame(summaries)

    comp_path = f"{args.out_prefix}_comparison.csv"
    summ_path = f"{args.out_prefix}_summary.csv"

    df.to_csv(comp_path, index=False)
    dfs.to_csv(summ_path, index=False)

    print("\nArquivos gerados:")
    print(f"  {comp_path}")
    print(f"  {summ_path}")

    print("\nSíntese rápida r=3:")

    cols = [
        "contrast",
        "jitter",
        "mesh",
        "r3_max_rel_e_error_vs_full",
        "r3_final_rel_e_error_vs_full",
        "r3_max_energy_drift",
        "r3_nnzG_over_nnzH",
        "r3_stable",
    ]

    print(dfs[cols].to_string(index=False))


if __name__ == "__main__":
    main()

