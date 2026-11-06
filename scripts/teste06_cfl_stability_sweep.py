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
    CFL stability sweep comparing the diagonal approximation, selected-solve inverse, and LU-step reference.

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
import os
import time
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp

try:
    import psutil
except Exception:
    psutil = None

try:
    from teste01D_material_contrast import (
        build_structured_tri_mesh,
        assemble_hodge_h1_material,
        prune_sparse,
        density,
    )
except ImportError as exc:
    raise ImportError(
        "Could not import teste01D_material_contrast.py. "
        "Place this script in the same folder and run with the correct --wdir."
    ) from exc

try:
    from teste05_compare_selected_inverse_vs_lu_step import (
        boundary_edge_mask,
        assemble_curl_and_m2,
        estimate_lambda_max_full,
        build_methods,
        gaussian_initial_b,
        nearest_face_probe,
        electromagnetic_energy,
        apply_method,
        relative_h_error,
        relative_m2_error,
        dominant_frequency_fft,
        time_correlation,
        relative_l2_signal_error,
        dense_memory_bytes,
        sparse_or_dense_nnz,
        sparse_or_dense_density,
        sparse_or_dense_memory_bytes,
    )
except ImportError as exc:
    raise ImportError(
        "Could not import teste05_compare_selected_inverse_vs_lu_step.py. "
        "Place this script in the same folder do Teste 05 and run with the correct --wdir."
    ) from exc


ArrayOrSparse = Union[np.ndarray, sp.spmatrix]


                                                              
            
                                                              

def current_rss_mb() -> float:
    if psutil is None:
        return np.nan

    try:
        proc = psutil.Process(os.getpid())
        return float(proc.memory_info().rss / (1024.0 ** 2))
    except Exception:
        return np.nan


def parse_case_token(token: str) -> Tuple[float, float]:
    """Helper routine used in the numerical experiments."""
    token = token.strip()

    if ":" in token:
        a, b = token.split(":", 1)
    elif "," in token:
        a, b = token.split(",", 1)
    else:
        raise ValueError(
            f"Invalid case: {token}. Use format contrast:jitter, por exemplo 1000:0.1"
        )

    return float(a), float(b)


def method_radius(name: str, radius: int) -> int:
    if name == "G_diag":
        return 0
    if name == "LU_step_reference":
        return -1
    return int(radius)


def is_practically_stable(max_drift: float, stable_flag: bool, energy_tol: float) -> bool:
    if not stable_flag:
        return False
    if not np.isfinite(max_drift):
        return False
    return bool(max_drift <= energy_tol)


def write_partial_outputs(
    build_rows: List[dict],
    summary_rows: List[dict],
    stability_rows: List[dict],
    critical_rows: List[dict],
    energy_rows: List[dict],
    error_rows: List[dict],
    signal_rows: List[dict],
    out_prefix: str,
) -> None:
    if build_rows:
        pd.DataFrame(build_rows).to_csv(f"{out_prefix}_build.csv", index=False)

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(f"{out_prefix}_summary.csv", index=False)

    if stability_rows:
        pd.DataFrame(stability_rows).to_csv(f"{out_prefix}_stability_table.csv", index=False)

    if critical_rows:
        pd.DataFrame(critical_rows).to_csv(f"{out_prefix}_critical_cfl.csv", index=False)

    if energy_rows:
        pd.DataFrame(energy_rows).to_csv(f"{out_prefix}_energy_series.csv", index=False)

    if error_rows:
        pd.DataFrame(error_rows).to_csv(f"{out_prefix}_error_series.csv", index=False)

    if signal_rows:
        pd.DataFrame(signal_rows).to_csv(f"{out_prefix}_signals.csv", index=False)


                                                              
                    
                                                              

def prepare_case(
    nx: int,
    ny: int,
    contrast: float,
    jitter: float,
    radius: int,
    seed: int,
    tol: float,
    chunk_size: int,
    pulse_sigma: float,
    pulse_x: float,
    pulse_y: float,
    probe_x: float,
    probe_y: float,
    verbose_solve: bool,
) -> dict:
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

    H = H_full_edges[free_edges, :][:, free_edges].tocsr()
    H = prune_sparse(H, tol=tol)

    n = H.shape[0]

    C, M2, areas = assemble_curl_and_m2(mesh, free_edges, nu=1.0)
    K = (C.T @ M2 @ C).tocsr()

    t0 = time.perf_counter()
    lambda_max = estimate_lambda_max_full(K, H)
    runtime_lambda_max_s = time.perf_counter() - t0

    dt_cfl = 2.0 / np.sqrt(lambda_max)

    methods, lu, build_rows = build_methods(
        H=H,
        radius=radius,
        chunk_size=chunk_size,
        tol=tol,
        verbose_solve=verbose_solve,
    )

    methods["LU_step_reference"] = None

    b0 = gaussian_initial_b(
        mesh,
        areas,
        center=(pulse_x, pulse_y),
        sigma=pulse_sigma,
    )

    e0 = np.zeros(n, dtype=float)

    CTM2 = C.T @ M2

    probe_face, probe_cx, probe_cy = nearest_face_probe(
        mesh,
        probe_x,
        probe_y,
    )

    runtime_prepare_s = time.perf_counter() - t_case0

    for row in build_rows:
        row["mesh"] = f"{nx}x{ny}"
        row["nx"] = int(nx)
        row["ny"] = int(ny)
        row["contrast"] = float(contrast)
        row["jitter"] = float(jitter)
        row["radius_global"] = int(radius)
        row["NE_free"] = int(n)
        row["NF"] = int(mesh.triangles.shape[0])
        row["NV"] = int(mesh.points.shape[0])
        row["nnz_H"] = int(H.nnz)
        row["rho_H"] = float(density(H))
        row["lambda_max"] = float(lambda_max)
        row["dt_cfl"] = float(dt_cfl)
        row["runtime_lambda_max_s"] = float(runtime_lambda_max_s)
        row["runtime_prepare_case_s"] = float(runtime_prepare_s)

    return {
        "mesh": mesh,
        "H": H,
        "C": C,
        "M2": M2,
        "areas": areas,
        "K": K,
        "lambda_max": lambda_max,
        "dt_cfl": dt_cfl,
        "methods": methods,
        "lu": lu,
        "build_rows": build_rows,
        "b0": b0,
        "e0": e0,
        "CTM2": CTM2,
        "probe_face": probe_face,
        "probe_x_actual": probe_cx,
        "probe_y_actual": probe_cy,
        "runtime_prepare_s": runtime_prepare_s,
    }


                                                              
                      
                                                              

def run_one_cfl(
    prepared: dict,
    nx: int,
    ny: int,
    contrast: float,
    jitter: float,
    radius: int,
    cfl: float,
    steps: int,
    sample_stride: int,
    blowup_tol: float,
    energy_stability_tol: float,
    use_window: bool,
) -> Tuple[List[dict], List[dict], List[dict], List[dict]]:
    """Helper routine used in the numerical experiments."""
    H = prepared["H"]
    C = prepared["C"]
    M2 = prepared["M2"]
    areas = prepared["areas"]
    methods = prepared["methods"]
    lu = prepared["lu"]
    b0 = prepared["b0"]
    e0 = prepared["e0"]
    CTM2 = prepared["CTM2"]
    probe_face = prepared["probe_face"]
    probe_cx = prepared["probe_x_actual"]
    probe_cy = prepared["probe_y_actual"]
    dt_cfl = prepared["dt_cfl"]
    lambda_max = prepared["lambda_max"]

    n = H.shape[0]
    dt = float(cfl * dt_cfl)
    dt_sample = dt * sample_stride

    reference_method = "LU_step_reference"

    states: Dict[str, dict] = {}

    for name in methods:
        E0 = electromagnetic_energy(e0, b0, H, M2)

        states[name] = {
            "e": e0.copy(),
            "b": b0.copy(),
            "E0": float(E0),
            "energy_final": float(E0),
            "energy_final_relative": 1.0,
            "max_energy_drift": 0.0,
            "final_energy_drift": 0.0,
            "stable": True,
            "first_unstable_step": -1,
            "runtime_update_s": 0.0,
            "signal": [],
            "time": [],
            "step": [],
            "energy_relative": [],
            "energy_drift": [],
            "rel_e_error_vs_LU": [],
            "rel_b_error_vs_LU": [],
        }

    t_dyn0 = time.perf_counter()

    for it in range(1, steps + 1):
        for name, G in methods.items():
            st = states[name]

            if not st["stable"]:
                continue

            e = st["e"]
            b = st["b"]

            t_up0 = time.perf_counter()

            b_new = b - dt * (C @ e)
            rhs = CTM2 @ b_new

            correction = apply_method(
                method_name=name,
                G=G,
                lu=lu,
                rhs=rhs,
            )

            e_new = e + dt * correction

            st["runtime_update_s"] += time.perf_counter() - t_up0

            st["e"] = e_new
            st["b"] = b_new

            En = electromagnetic_energy(e_new, b_new, H, M2)
            st["energy_final"] = float(En)

            if st["E0"] > 0.0 and np.isfinite(En):
                Erel = En / st["E0"]
                drift = abs(Erel - 1.0)

                st["energy_final_relative"] = float(Erel)
                st["final_energy_drift"] = float(drift)
                st["max_energy_drift"] = max(st["max_energy_drift"], float(drift))
            else:
                st["stable"] = False
                st["first_unstable_step"] = int(it)

            if (
                not np.all(np.isfinite(e_new))
                or not np.all(np.isfinite(b_new))
                or not np.isfinite(st["max_energy_drift"])
                or st["max_energy_drift"] > blowup_tol
            ):
                st["stable"] = False
                st["first_unstable_step"] = int(it)

        if it % sample_stride == 0 or it == steps:
            t_now = it * dt

            ref_st = states[reference_method]
            ref_e = ref_st["e"]
            ref_b = ref_st["b"]

            for name, st in states.items():
                if st["stable"] or len(st["signal"]) == 0:
                    b_probe_flux = st["b"][probe_face]
                    b_probe_density = b_probe_flux / areas[probe_face]
                else:
                    b_probe_density = np.nan

                if ref_st["stable"]:
                    rel_e = relative_h_error(st["e"], ref_e, H)
                    rel_b = relative_m2_error(st["b"], ref_b, M2)
                else:
                    rel_e = np.nan
                    rel_b = np.nan

                st["signal"].append(float(b_probe_density))
                st["time"].append(float(t_now))
                st["step"].append(int(it))
                st["energy_relative"].append(float(st["energy_final_relative"]))
                st["energy_drift"].append(float(st["final_energy_drift"]))
                st["rel_e_error_vs_LU"].append(float(rel_e))
                st["rel_b_error_vs_LU"].append(float(rel_b))

    runtime_dynamics_all_methods_s = time.perf_counter() - t_dyn0

    ref_signal = np.asarray(states[reference_method]["signal"], dtype=float)

    ref_fft = dominant_frequency_fft(
        ref_signal,
        dt_sample=dt_sample,
        use_window=use_window,
    )

    ref_f_peak = ref_fft["f_peak"]

    summary_rows: List[dict] = []
    energy_rows: List[dict] = []
    error_rows: List[dict] = []
    signal_rows: List[dict] = []

    dense_bytes = dense_memory_bytes(n)

    for name, G in methods.items():
        st = states[name]

        signal = np.asarray(st["signal"], dtype=float)
        times = np.asarray(st["time"], dtype=float)
        sampled_steps = np.asarray(st["step"], dtype=int)

        fft_info = dominant_frequency_fft(
            signal,
            dt_sample=dt_sample,
            use_window=use_window,
        )

        if np.isfinite(ref_f_peak) and abs(ref_f_peak) > 1.0e-30:
            freq_rel_error = abs(fft_info["f_peak"] - ref_f_peak) / abs(ref_f_peak)
        else:
            freq_rel_error = np.nan

        corr = time_correlation(signal, ref_signal)
        sig_l2 = relative_l2_signal_error(signal, ref_signal)

        rel_e_series = np.asarray(st["rel_e_error_vs_LU"], dtype=float)
        rel_b_series = np.asarray(st["rel_b_error_vs_LU"], dtype=float)

        if name == "LU_step_reference":
            nnz_G = np.nan
            rho_G = np.nan
            mem_G = np.nan
            mem_ratio = np.nan
            nnz_ratio = np.nan
        else:
            nnz_G = sparse_or_dense_nnz(G)
            rho_G = sparse_or_dense_density(G)
            mem_G = sparse_or_dense_memory_bytes(G)
            mem_ratio = mem_G / dense_bytes
            nnz_ratio = nnz_G / H.nnz

        practical = is_practically_stable(
            max_drift=st["max_energy_drift"],
            stable_flag=st["stable"],
            energy_tol=energy_stability_tol,
        )

        summary_rows.append({
            "mesh": f"{nx}x{ny}",
            "nx": int(nx),
            "ny": int(ny),
            "contrast": float(contrast),
            "jitter": float(jitter),
            "method": name,
            "reference_method": reference_method,
            "radius": int(method_radius(name, radius)),
            "NE_free": int(n),
            "NF": int(prepared["mesh"].triangles.shape[0]),
            "NV": int(prepared["mesh"].points.shape[0]),
            "nnz_H": int(H.nnz),
            "rho_H": float(density(H)),
            "nnz_G": nnz_G,
            "rho_G": rho_G,
            "nnzG_over_nnzH": nnz_ratio,
            "memory_bytes_G": mem_G,
            "memory_ratio_vs_dense": mem_ratio,
            "lambda_max": float(lambda_max),
            "dt_cfl": float(dt_cfl),
            "cfl": float(cfl),
            "dt": float(dt),
            "steps": int(steps),
            "sample_stride": int(sample_stride),
            "dt_sample": float(dt_sample),
            "t_final": float(steps * dt),
            "n_samples": int(signal.size),
            "probe_face": int(probe_face),
            "probe_x_actual": float(probe_cx),
            "probe_y_actual": float(probe_cy),
            "f_peak": float(fft_info["f_peak"]),
            "f_reference_LU": float(ref_f_peak),
            "freq_rel_error_vs_LU": float(freq_rel_error),
            "signal_corr_vs_LU": float(corr),
            "signal_l2_error_vs_LU": float(sig_l2),
            "max_rel_e_error_vs_LU": (
                float(np.nanmax(rel_e_series)) if rel_e_series.size else np.nan
            ),
            "final_rel_e_error_vs_LU": (
                float(rel_e_series[-1]) if rel_e_series.size else np.nan
            ),
            "max_rel_b_error_vs_LU": (
                float(np.nanmax(rel_b_series)) if rel_b_series.size else np.nan
            ),
            "final_rel_b_error_vs_LU": (
                float(rel_b_series[-1]) if rel_b_series.size else np.nan
            ),
            "energy_initial": float(st["E0"]),
            "energy_final": float(st["energy_final"]),
            "energy_final_relative": float(st["energy_final_relative"]),
            "final_energy_drift": float(st["final_energy_drift"]),
            "max_energy_drift": float(st["max_energy_drift"]),
            "energy_stability_tol": float(energy_stability_tol),
            "stable": bool(st["stable"]),
            "practically_stable": bool(practical),
            "first_unstable_step": int(st["first_unstable_step"]),
            "runtime_update_s": float(st["runtime_update_s"]),
            "runtime_update_per_step_s": float(st["runtime_update_s"] / steps),
            "runtime_dynamics_all_methods_s": float(runtime_dynamics_all_methods_s),
        })

        for k in range(signal.size):
            signal_rows.append({
                "mesh": f"{nx}x{ny}",
                "contrast": float(contrast),
                "jitter": float(jitter),
                "cfl": float(cfl),
                "method": name,
                "step": int(sampled_steps[k]),
                "time": float(times[k]),
                "probe_bz": float(signal[k]),
                "probe_face": int(probe_face),
                "probe_x": float(probe_cx),
                "probe_y": float(probe_cy),
            })

            energy_rows.append({
                "mesh": f"{nx}x{ny}",
                "contrast": float(contrast),
                "jitter": float(jitter),
                "cfl": float(cfl),
                "method": name,
                "step": int(sampled_steps[k]),
                "time": float(times[k]),
                "energy_relative": float(st["energy_relative"][k]),
                "energy_drift": float(st["energy_drift"][k]),
                "stable": bool(st["stable"]),
                "practically_stable": bool(practical),
            })

            error_rows.append({
                "mesh": f"{nx}x{ny}",
                "contrast": float(contrast),
                "jitter": float(jitter),
                "cfl": float(cfl),
                "method": name,
                "reference_method": reference_method,
                "step": int(sampled_steps[k]),
                "time": float(times[k]),
                "rel_e_error_vs_LU": float(st["rel_e_error_vs_LU"][k]),
                "rel_b_error_vs_LU": float(st["rel_b_error_vs_LU"][k]),
            })

    return summary_rows, energy_rows, error_rows, signal_rows


                                                              
                    
                                                              

def build_stability_table(summary_rows: List[dict]) -> List[dict]:
    rows: List[dict] = []

    df = pd.DataFrame(summary_rows)

    if df.empty:
        return rows

    for _, row in df.iterrows():
        rows.append({
            "mesh": row["mesh"],
            "contrast": row["contrast"],
            "jitter": row["jitter"],
            "cfl": row["cfl"],
            "method": row["method"],
            "stable": row["stable"],
            "practically_stable": row["practically_stable"],
            "first_unstable_step": row["first_unstable_step"],
            "max_energy_drift": row["max_energy_drift"],
            "final_energy_drift": row["final_energy_drift"],
            "max_rel_e_error_vs_LU": row["max_rel_e_error_vs_LU"],
            "final_rel_e_error_vs_LU": row["final_rel_e_error_vs_LU"],
            "freq_rel_error_vs_LU": row["freq_rel_error_vs_LU"],
            "signal_corr_vs_LU": row["signal_corr_vs_LU"],
            "runtime_update_per_step_s": row["runtime_update_per_step_s"],
        })

    return rows


def build_critical_cfl_table(summary_rows: List[dict]) -> List[dict]:
    df = pd.DataFrame(summary_rows)

    if df.empty:
        return []

    rows: List[dict] = []

    group_cols = ["mesh", "contrast", "jitter", "method"]

    for keys, g in df.groupby(group_cols):
        mesh, contrast, jitter, method = keys

        g = g.sort_values("cfl")

        stable_g = g[g["stable"] == True]
        practical_g = g[g["practically_stable"] == True]

        max_cfl_stable = float(stable_g["cfl"].max()) if not stable_g.empty else np.nan
        max_cfl_practical = float(practical_g["cfl"].max()) if not practical_g.empty else np.nan

        first_fail = g[g["stable"] == False]
        first_practical_fail = g[g["practically_stable"] == False]

        rows.append({
            "mesh": mesh,
            "contrast": float(contrast),
            "jitter": float(jitter),
            "method": method,
            "max_cfl_numerically_stable": max_cfl_stable,
            "max_cfl_practically_stable": max_cfl_practical,
            "first_cfl_numerical_failure": (
                float(first_fail["cfl"].min()) if not first_fail.empty else np.nan
            ),
            "first_cfl_practical_failure": (
                float(first_practical_fail["cfl"].min())
                if not first_practical_fail.empty else np.nan
            ),
            "energy_tol_for_practical_stability": (
                float(g["energy_stability_tol"].iloc[0])
                if "energy_stability_tol" in g.columns else np.nan
            ),
        })

    return rows


                                                              
     
                                                              

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Teste 06 — varredura de CFL para G_H3, diagonal e LU-step."
    )

    p.add_argument(
        "--mesh",
        type=int,
        default=80,
        help="Malha quadrada nx=ny. Default: 80",
    )

    p.add_argument(
        "--cases",
        nargs="+",
        default=["1:0.0", "1000:0.1"],
        help="Casos no formato contraste:jitter. Default: 1:0.0 1000:0.1",
    )

    p.add_argument(
        "--cfl-values",
        nargs="+",
        type=float,
        default=[0.25, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.05, 1.10],
        help="Valores de CFL a testar.",
    )

    p.add_argument(
        "--radius",
        type=int,
        default=3,
        help="Raio topológico. Default: 3",
    )

    p.add_argument(
        "--steps",
        type=int,
        default=4000,
        help="Número de passos por CFL. Default: 4000",
    )

    p.add_argument(
        "--sample-stride",
        type=int,
        default=5,
        help="Intervalo de amostragem. Default: 5",
    )

    p.add_argument(
        "--chunk-size",
        type=int,
        default=16,
        help="Número de colunas por bloco no selected-solve. Default: 16",
    )

    p.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed. Default: 1234",
    )

    p.add_argument(
        "--tol",
        type=float,
        default=1.0e-14,
        help="Structural tolerance. Default: 1e-14",
    )

    p.add_argument(
        "--pulse-sigma",
        type=float,
        default=0.08,
        help="Largura da gaussiana inicial. Default: 0.08",
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
        "--probe-x",
        type=float,
        default=0.65,
        help="Coordenada x da sonda. Default: 0.65",
    )

    p.add_argument(
        "--probe-y",
        type=float,
        default=0.50,
        help="Coordenada y da sonda. Default: 0.50",
    )

    p.add_argument(
        "--blowup-tol",
        type=float,
        default=1.0e3,
        help="Drift de energia para declarar blow-up numérico. Default: 1e3",
    )

    p.add_argument(
        "--energy-stability-tol",
        type=float,
        default=0.10,
        help="Tolerância de drift para estabilidade prática. Default: 0.10",
    )

    p.add_argument(
        "--no-window",
        action="store_true",
        help="Desativa janela Hann na FFT.",
    )

    p.add_argument(
        "--verbose-solve",
        action="store_true",
        help="Imprime progresso interno do selected-solve.",
    )

    p.add_argument(
        "--out-prefix",
        type=str,
        default="teste06_cfl",
        help="Output CSV prefix.",
    )

    return p.parse_args()


                                                              
      
                                                              

def main() -> None:
    args = parse_args()

    cases = [parse_case_token(tok) for tok in args.cases]

    print("\nTeste 06 — Varredura de CFL e estabilidade temporal")
    print("=" * 100)
    print(f"mesh                  : {args.mesh}x{args.mesh}")
    print(f"cases                 : {cases}")
    print(f"cfl_values            : {args.cfl_values}")
    print(f"radius                : {args.radius}")
    print(f"steps                 : {args.steps}")
    print(f"sample_stride         : {args.sample_stride}")
    print(f"chunk_size            : {args.chunk_size}")
    print(f"blowup_tol            : {args.blowup_tol}")
    print(f"energy_stability_tol  : {args.energy_stability_tol}")
    print(f"psutil_available      : {psutil is not None}")
    print(f"out_prefix            : {args.out_prefix}")
    print("=" * 100)

    all_build: List[dict] = []
    all_summary: List[dict] = []
    all_energy: List[dict] = []
    all_errors: List[dict] = []
    all_signals: List[dict] = []
    all_stability: List[dict] = []
    all_critical: List[dict] = []
    failures: List[dict] = []

    global_t0 = time.perf_counter()

    for contrast, jitter in cases:
        print("\n" + "#" * 100)
        print(f"Preparando caso contrast={contrast:g}, jitter={jitter:g}")
        print("#" * 100)

        rss_before = current_rss_mb()
        t_prepare0 = time.perf_counter()

        try:
            prepared = prepare_case(
                nx=int(args.mesh),
                ny=int(args.mesh),
                contrast=float(contrast),
                jitter=float(jitter),
                radius=int(args.radius),
                seed=int(args.seed),
                tol=float(args.tol),
                chunk_size=int(args.chunk_size),
                pulse_sigma=float(args.pulse_sigma),
                pulse_x=float(args.pulse_x),
                pulse_y=float(args.pulse_y),
                probe_x=float(args.probe_x),
                probe_y=float(args.probe_y),
                verbose_solve=bool(args.verbose_solve),
            )

            rss_after_prepare = current_rss_mb()
            prepare_elapsed = time.perf_counter() - t_prepare0

            for row in prepared["build_rows"]:
                row["rss_before_prepare_mb"] = float(rss_before)
                row["rss_after_prepare_mb"] = float(rss_after_prepare)
                row["rss_prepare_delta_mb"] = (
                    float(rss_after_prepare - rss_before)
                    if np.isfinite(rss_after_prepare) and np.isfinite(rss_before)
                    else np.nan
                )
                row["wall_prepare_s"] = float(prepare_elapsed)

            all_build.extend(prepared["build_rows"])

            print(f"Preparação concluída em {prepare_elapsed:.3f} s")
            print(
                f"n={prepared['H'].shape[0]} | nnzH={prepared['H'].nnz} | "
                f"dt_cfl={prepared['dt_cfl']:.6e}"
            )

        except Exception as exc:
            failure = {
                "stage": "prepare_case",
                "mesh": f"{args.mesh}x{args.mesh}",
                "contrast": float(contrast),
                "jitter": float(jitter),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            failures.append(failure)
            pd.DataFrame(failures).to_csv(f"{args.out_prefix}_failures.csv", index=False)

            print(f"ERRO na preparação: {type(exc).__name__}: {exc}")
            continue

        case_summary_rows: List[dict] = []

        for cfl in args.cfl_values:
            print("\n" + "-" * 100)
            print(f"Rodando CFL={cfl:g} | contrast={contrast:g} | jitter={jitter:g}")
            print("-" * 100)

            rss_before_cfl = current_rss_mb()
            t0 = time.perf_counter()

            try:
                summary_rows, energy_rows, error_rows, signal_rows = run_one_cfl(
                    prepared=prepared,
                    nx=int(args.mesh),
                    ny=int(args.mesh),
                    contrast=float(contrast),
                    jitter=float(jitter),
                    radius=int(args.radius),
                    cfl=float(cfl),
                    steps=int(args.steps),
                    sample_stride=int(args.sample_stride),
                    blowup_tol=float(args.blowup_tol),
                    energy_stability_tol=float(args.energy_stability_tol),
                    use_window=not args.no_window,
                )

                elapsed = time.perf_counter() - t0
                rss_after_cfl = current_rss_mb()

                for row in summary_rows:
                    row["rss_before_cfl_mb"] = float(rss_before_cfl)
                    row["rss_after_cfl_mb"] = float(rss_after_cfl)
                    row["rss_cfl_delta_mb"] = (
                        float(rss_after_cfl - rss_before_cfl)
                        if np.isfinite(rss_after_cfl) and np.isfinite(rss_before_cfl)
                        else np.nan
                    )
                    row["wall_cfl_s"] = float(elapsed)

                stability_rows = build_stability_table(summary_rows)

                all_summary.extend(summary_rows)
                all_energy.extend(energy_rows)
                all_errors.extend(error_rows)
                all_signals.extend(signal_rows)
                all_stability.extend(stability_rows)
                case_summary_rows.extend(summary_rows)

                all_critical = build_critical_cfl_table(all_summary)

                write_partial_outputs(
                    build_rows=all_build,
                    summary_rows=all_summary,
                    stability_rows=all_stability,
                    critical_rows=all_critical,
                    energy_rows=all_energy,
                    error_rows=all_errors,
                    signal_rows=all_signals,
                    out_prefix=args.out_prefix,
                )

                df_cfl = pd.DataFrame(summary_rows)

                print(f"Concluído CFL={cfl:g} em {elapsed:.3f} s")

                for _, row in df_cfl.iterrows():
                    print(
                        f"  {row['method']:<30s} "
                        f"stable={row['stable']} "
                        f"practical={row['practically_stable']} "
                        f"drift={row['max_energy_drift']:.3e} "
                        f"Ee_max={row['max_rel_e_error_vs_LU']:.3e} "
                        f"ferr={row['freq_rel_error_vs_LU']:.3e} "
                        f"corr={row['signal_corr_vs_LU']:.4f} "
                        f"upd/step={row['runtime_update_per_step_s']:.3e}s"
                    )

            except Exception as exc:
                failure = {
                    "stage": "run_one_cfl",
                    "mesh": f"{args.mesh}x{args.mesh}",
                    "contrast": float(contrast),
                    "jitter": float(jitter),
                    "cfl": float(cfl),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }

                failures.append(failure)
                pd.DataFrame(failures).to_csv(f"{args.out_prefix}_failures.csv", index=False)

                print(f"ERRO no CFL={cfl:g}: {type(exc).__name__}: {exc}")
                continue

                                                  
        all_critical = build_critical_cfl_table(all_summary)

        write_partial_outputs(
            build_rows=all_build,
            summary_rows=all_summary,
            stability_rows=all_stability,
            critical_rows=all_critical,
            energy_rows=all_energy,
            error_rows=all_errors,
            signal_rows=all_signals,
            out_prefix=args.out_prefix,
        )

    total_time = time.perf_counter() - global_t0

    print("\n" + "=" * 100)
    print("Arquivos gerados:")
    print(f"  {args.out_prefix}_build.csv")
    print(f"  {args.out_prefix}_summary.csv")
    print(f"  {args.out_prefix}_stability_table.csv")
    print(f"  {args.out_prefix}_critical_cfl.csv")
    print(f"  {args.out_prefix}_energy_series.csv")
    print(f"  {args.out_prefix}_error_series.csv")
    print(f"  {args.out_prefix}_signals.csv")

    if failures:
        print(f"  {args.out_prefix}_failures.csv")

    print("=" * 100)
    print(f"Tempo total: {total_time:.3f} s")

    if all_summary:
        df = pd.DataFrame(all_summary)

        cols = [
            "contrast",
            "jitter",
            "cfl",
            "method",
            "stable",
            "practically_stable",
            "first_unstable_step",
            "max_energy_drift",
            "max_rel_e_error_vs_LU",
            "freq_rel_error_vs_LU",
            "signal_corr_vs_LU",
            "runtime_update_per_step_s",
        ]

        print("\nSíntese final:")
        print(df[cols].to_string(index=False))

    if all_critical:
        dfc = pd.DataFrame(all_critical)

        print("\nCFL crítico estimado:")
        print(dfc.to_string(index=False))


if __name__ == "__main__":
    main()

