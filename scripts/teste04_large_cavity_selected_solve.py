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
    Large-cavity DEC/FETD simulation using the radius-three selected-solve inverse Hodge approximation.

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
import scipy.sparse.linalg as spla

try:
    import psutil
except Exception:
    psutil = None

try:
    from teste01D_material_contrast import (
        build_structured_tri_mesh,
        assemble_hodge_h1_material,
        prune_sparse,
        radius_patterns_from_H,
        diag_inverse,
        density,
    )
except ImportError as exc:
    raise ImportError(
        "Could not import teste01D_material_contrast.py. "
        "Place this script in the same folder and run with the correct --wdir."
    ) from exc

try:
    from teste03A_selected_solve_hodge import (
        selected_inverse_from_solve,
        ensure_symmetric_pattern,
        sparse_memory_bytes,
        dense_memory_bytes,
    )
except ImportError as exc:
    raise ImportError(
        "Could not import teste03A_selected_solve_hodge.py. "
        "Place this script in the same folder do Teste 03A and run with the correct --wdir."
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


def method_nnz(G: ArrayOrSparse) -> int:
    if sp.issparse(G):
        return int(G.nnz)

    return int(G.shape[0] * G.shape[1])


def method_density(G: ArrayOrSparse) -> float:
    if sp.issparse(G):
        return float(density(G))

    return 1.0


def method_memory_bytes(G: ArrayOrSparse) -> int:
    if sp.issparse(G):
        return int(sparse_memory_bytes(G))

    return int(G.nbytes)


def apply_G(G: ArrayOrSparse, rhs: np.ndarray) -> np.ndarray:
    return G @ rhs


                                                              
                
                                                              

def triangle_area(coords: np.ndarray) -> float:
    x0, y0 = coords[0]
    x1, y1 = coords[1]
    x2, y2 = coords[2]

    return 0.5 * ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))


def boundary_edge_mask(mesh, tol: float = 1.0e-12) -> np.ndarray:
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
    mesh,
    free_edges: np.ndarray,
    nu: float = 1.0,
) -> Tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray]:
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
        shape=(nf, len(free_edges)),
    ).tocsr()

    M2 = sp.diags(nu / areas, format="csr")

    return C, M2, areas


def estimate_lambda_max_full(K: sp.csr_matrix, H: sp.csr_matrix) -> float:
    val = spla.eigsh(
        K,
        M=H,
        k=1,
        which="LA",
        return_eigenvectors=False,
        tol=1.0e-8,
        maxiter=20000,
    )[0]

    return float(val)


                                                              
                                   
                                                              

def gaussian_initial_b(
    mesh,
    areas: np.ndarray,
    center: Tuple[float, float] = (0.35, 0.50),
    sigma: float = 0.08,
) -> np.ndarray:
    centroids = mesh.points[mesh.triangles].mean(axis=1)

    dx = centroids[:, 0] - center[0]
    dy = centroids[:, 1] - center[1]

    B = np.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))

    return B * areas


def nearest_face_probe(
    mesh,
    x: float,
    y: float,
) -> Tuple[int, float, float]:
    centroids = mesh.points[mesh.triangles].mean(axis=1)

    dx = centroids[:, 0] - x
    dy = centroids[:, 1] - y

    idx = int(np.argmin(dx * dx + dy * dy))

    return idx, float(centroids[idx, 0]), float(centroids[idx, 1])


def electromagnetic_energy(
    e: np.ndarray,
    b: np.ndarray,
    H: sp.csr_matrix,
    M2: sp.csr_matrix,
) -> float:
    return 0.5 * float(e @ (H @ e)) + 0.5 * float(b @ (M2 @ b))


                                                              
                            
                                                              

def dominant_frequency_fft(
    signal: np.ndarray,
    dt_sample: float,
    use_window: bool = True,
) -> Dict[str, float]:
    y = np.asarray(signal, dtype=float)

    if y.size < 8:
        return {
            "f_peak": np.nan,
            "peak_amplitude": np.nan,
            "df": np.nan,
            "peak_index": -1,
        }

    y = y - np.mean(y)

    if np.linalg.norm(y) < 1.0e-30:
        return {
            "f_peak": np.nan,
            "peak_amplitude": np.nan,
            "df": np.nan,
            "peak_index": -1,
        }

    if use_window:
        y = y * np.hanning(y.size)

    spec = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(y.size, d=dt_sample)

    if spec.size <= 1:
        return {
            "f_peak": np.nan,
            "peak_amplitude": np.nan,
            "df": np.nan,
            "peak_index": -1,
        }

    spec[0] = 0.0

    k = int(np.argmax(spec))
    f_peak = float(freqs[k])

                                             
    if 1 <= k < spec.size - 1:
        alpha = float(spec[k - 1])
        beta = float(spec[k])
        gamma = float(spec[k + 1])

        denom = alpha - 2.0 * beta + gamma

        if abs(denom) > 1.0e-30:
            delta = 0.5 * (alpha - gamma) / denom
            delta = float(np.clip(delta, -0.5, 0.5))
            f_peak = float((k + delta) / (y.size * dt_sample))

    df = float(freqs[1] - freqs[0]) if freqs.size > 1 else np.nan

    return {
        "f_peak": f_peak,
        "peak_amplitude": float(spec[k]),
        "df": df,
        "peak_index": int(k),
    }


def time_correlation(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=float) - np.mean(a)
    bb = np.asarray(b, dtype=float) - np.mean(b)

    den = np.linalg.norm(aa) * np.linalg.norm(bb)

    if den < 1.0e-30:
        return np.nan

    return float(np.dot(aa, bb) / den)


def relative_l2_signal_error(a: np.ndarray, ref: np.ndarray) -> float:
    den = np.linalg.norm(ref)

    if den < 1.0e-30:
        return np.nan

    return float(np.linalg.norm(a - ref) / den)


                                                              
                 
                                                              

def build_inverse_methods(
    H: sp.csr_matrix,
    radius: int,
    chunk_size: int,
    tol: float,
    seed: int,
    include_full_if_possible: bool,
    full_max_n: int,
    verbose_solve: bool,
) -> Tuple[Dict[str, ArrayOrSparse], List[dict]]:
    n = H.shape[0]

    rows: List[dict] = []

    methods: Dict[str, ArrayOrSparse] = {}

    dense_bytes = dense_memory_bytes(n)

                                                              
               
                                                              
    t0 = time.perf_counter()
    lu = spla.splu(H.tocsc())
    t_factor = time.perf_counter() - t0

                                                              
              
                                                              
    t0 = time.perf_counter()
    G_diag = diag_inverse(H).tocsr()
    t_build_diag = time.perf_counter() - t0

    methods["G_diag"] = G_diag

    rows.append({
        "method": "G_diag",
        "radius": 0,
        "runtime_factor_s": float(t_factor),
        "runtime_pattern_s": 0.0,
        "runtime_build_s": float(t_build_diag),
        "nnz_G": int(method_nnz(G_diag)),
        "rho_G": float(method_density(G_diag)),
        "memory_bytes_G": int(method_memory_bytes(G_diag)),
        "dense_inverse_memory_bytes": int(dense_bytes),
        "memory_ratio_vs_dense": float(method_memory_bytes(G_diag) / dense_bytes),
    })

                                                              
                          
                                                              
    t0 = time.perf_counter()
    patterns = radius_patterns_from_H(H, radii=[radius], tol=tol)
    P = ensure_symmetric_pattern(patterns[radius])
    t_pattern = time.perf_counter() - t0

    t0 = time.perf_counter()
    G_sel = selected_inverse_from_solve(
        lu,
        P,
        chunk_size=chunk_size,
        tol=tol,
        symmetrize=True,
        verbose=verbose_solve,
    )
    t_build_sel = time.perf_counter() - t0

    method_sel = f"G_H_radius_{radius}_selected_solve"
    methods[method_sel] = G_sel

    rows.append({
        "method": method_sel,
        "radius": int(radius),
        "runtime_factor_s": float(t_factor),
        "runtime_pattern_s": float(t_pattern),
        "runtime_build_s": float(t_build_sel),
        "nnz_G": int(method_nnz(G_sel)),
        "rho_G": float(method_density(G_sel)),
        "memory_bytes_G": int(method_memory_bytes(G_sel)),
        "dense_inverse_memory_bytes": int(dense_bytes),
        "memory_ratio_vs_dense": float(method_memory_bytes(G_sel) / dense_bytes),
    })

                                                              
                                                       
                                                              
    if include_full_if_possible and n <= full_max_n:
        t0 = time.perf_counter()
        G_full = np.linalg.inv(H.toarray())
        t_build_full = time.perf_counter() - t0

        methods["G_full"] = G_full

        rows.append({
            "method": "G_full",
            "radius": -1,
            "runtime_factor_s": float(t_factor),
            "runtime_pattern_s": 0.0,
            "runtime_build_s": float(t_build_full),
            "nnz_G": int(method_nnz(G_full)),
            "rho_G": float(method_density(G_full)),
            "memory_bytes_G": int(method_memory_bytes(G_full)),
            "dense_inverse_memory_bytes": int(dense_bytes),
            "memory_ratio_vs_dense": float(method_memory_bytes(G_full) / dense_bytes),
        })

    return methods, rows


                                                              
          
                                                              

def run_dynamic_case(
    nx: int,
    ny: int,
    contrast: float,
    jitter: float,
    radius: int,
    steps: int,
    cfl: float,
    sample_stride: int,
    seed: int,
    tol: float,
    chunk_size: int,
    pulse_sigma: float,
    pulse_x: float,
    pulse_y: float,
    probe_x: float,
    probe_y: float,
    blowup_tol: float,
    include_full_if_possible: bool,
    full_max_n: int,
    use_window: bool,
    verbose_solve: bool,
) -> Tuple[List[dict], List[dict], List[dict], List[dict]]:
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
    t_lmax = time.perf_counter() - t0

    dt_cfl = 2.0 / np.sqrt(lambda_max)
    dt = cfl * dt_cfl

    methods, build_rows = build_inverse_methods(
        H=H,
        radius=radius,
        chunk_size=chunk_size,
        tol=tol,
        seed=seed,
        include_full_if_possible=include_full_if_possible,
        full_max_n=full_max_n,
        verbose_solve=verbose_solve,
    )

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

    dense_bytes = dense_memory_bytes(n)

                                     
                                                                           
    method_sel = f"G_H_radius_{radius}_selected_solve"

    if "G_full" in methods:
        reference_method = "G_full"
    else:
        reference_method = method_sel

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
            "runtime_update_s": 0.0,
            "signal": [],
            "time": [],
            "step": [],
            "energy_relative": [],
            "energy_drift": [],
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
            e_new = e + dt * apply_G(G, rhs)

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
            t_now = it * dt

            for name, st in states.items():
                b_probe_flux = st["b"][probe_face]
                b_probe_density = b_probe_flux / areas[probe_face]

                st["signal"].append(float(b_probe_density))
                st["time"].append(float(t_now))
                st["step"].append(int(it))
                st["energy_relative"].append(float(st["energy_final_relative"]))
                st["energy_drift"].append(float(st["final_energy_drift"]))

    runtime_dynamics_s = time.perf_counter() - t_dyn0
    runtime_total_case_s = time.perf_counter() - t_case0

                                                              
                       
                                                              
    ref_signal = np.asarray(states[reference_method]["signal"], dtype=float)
    dt_sample = dt * sample_stride

    ref_fft = dominant_frequency_fft(
        ref_signal,
        dt_sample=dt_sample,
        use_window=use_window,
    )

    ref_f_peak = ref_fft["f_peak"]

    summary_rows: List[dict] = []
    signal_rows: List[dict] = []
    energy_rows: List[dict] = []

    build_by_method = {row["method"]: row for row in build_rows}

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

        mem_bytes = method_memory_bytes(G)

        summary_rows.append({
            "mesh": f"{nx}x{ny}",
            "nx": int(nx),
            "ny": int(ny),
            "contrast": float(contrast),
            "jitter": float(jitter),
            "method": name,
            "reference_method": reference_method,
            "radius": int(radius if "radius" in name else 0),
            "NV": int(mesh.points.shape[0]),
            "NE_total": int(mesh.edges.shape[0]),
            "NE_free": int(n),
            "NF": int(mesh.triangles.shape[0]),
            "nnz_H": int(H.nnz),
            "rho_H": float(density(H)),
            "nnz_G": int(method_nnz(G)),
            "rho_G": float(method_density(G)),
            "nnzG_over_nnzH": float(method_nnz(G) / H.nnz),
            "memory_bytes_G": int(mem_bytes),
            "dense_inverse_memory_bytes": int(dense_bytes),
            "memory_ratio_vs_dense": float(mem_bytes / dense_bytes),
            "lambda_max": float(lambda_max),
            "runtime_lambda_max_s": float(t_lmax),
            "dt_cfl": float(dt_cfl),
            "dt": float(dt),
            "cfl": float(cfl),
            "steps": int(steps),
            "sample_stride": int(sample_stride),
            "dt_sample": float(dt_sample),
            "t_final": float(steps * dt),
            "n_samples": int(signal.size),
            "probe_face": int(probe_face),
            "probe_x_requested": float(probe_x),
            "probe_y_requested": float(probe_y),
            "probe_x_actual": float(probe_cx),
            "probe_y_actual": float(probe_cy),
            "f_peak": float(fft_info["f_peak"]),
            "f_reference": float(ref_f_peak),
            "freq_abs_error_vs_reference": (
                float(abs(fft_info["f_peak"] - ref_f_peak))
                if np.isfinite(fft_info["f_peak"]) and np.isfinite(ref_f_peak)
                else np.nan
            ),
            "freq_rel_error_vs_reference": float(freq_rel_error),
            "signal_corr_vs_reference": float(corr),
            "signal_l2_error_vs_reference": float(sig_l2),
            "peak_amplitude": float(fft_info["peak_amplitude"]),
            "df": float(fft_info["df"]),
            "energy_initial": float(st["E0"]),
            "energy_final": float(st["energy_final"]),
            "energy_final_relative": float(st["energy_final_relative"]),
            "final_energy_drift": float(st["final_energy_drift"]),
            "max_energy_drift": float(st["max_energy_drift"]),
            "stable": bool(st["stable"]),
            "runtime_factor_s": float(build_by_method[name]["runtime_factor_s"]),
            "runtime_pattern_s": float(build_by_method[name]["runtime_pattern_s"]),
            "runtime_build_s": float(build_by_method[name]["runtime_build_s"]),
            "runtime_update_s": float(st["runtime_update_s"]),
            "runtime_update_per_step_s": float(st["runtime_update_s"] / steps),
            "runtime_dynamics_all_methods_s": float(runtime_dynamics_s),
            "runtime_total_case_s": float(runtime_total_case_s),
        })

        for k in range(signal.size):
            signal_rows.append({
                "mesh": f"{nx}x{ny}",
                "nx": int(nx),
                "ny": int(ny),
                "contrast": float(contrast),
                "jitter": float(jitter),
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
                "nx": int(nx),
                "ny": int(ny),
                "contrast": float(contrast),
                "jitter": float(jitter),
                "method": name,
                "step": int(sampled_steps[k]),
                "time": float(times[k]),
                "energy_relative": float(st["energy_relative"][k]),
                "energy_drift": float(st["energy_drift"][k]),
                "stable": bool(st["stable"]),
            })

                                                   
    for row in build_rows:
        row["mesh"] = f"{nx}x{ny}"
        row["nx"] = int(nx)
        row["ny"] = int(ny)
        row["contrast"] = float(contrast)
        row["jitter"] = float(jitter)
        row["NE_free"] = int(n)
        row["nnz_H"] = int(H.nnz)
        row["rho_H"] = float(density(H))
        row["runtime_lambda_max_s"] = float(t_lmax)
        row["runtime_total_case_s"] = float(runtime_total_case_s)

    return build_rows, summary_rows, signal_rows, energy_rows


                                                              
     
                                                              

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Teste 04 — dinâmica grande em cavidade PEC usando "
            "G_H,3 selected-solve without dense inverse."
        )
    )

    p.add_argument(
        "--meshes",
        nargs="+",
        type=int,
        default=[80],
        help="Square meshes nx=ny. Default: 80",
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
        "--radius",
        type=int,
        default=3,
        help="Raio topológico. Default: 3",
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
        default=5,
        help="Intervalo de amostragem. Default: 5",
    )

    p.add_argument(
        "--chunk-size",
        type=int,
        default=16,
        help="Número de colunas resolvidas por bloco. Default: 16",
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
        default=1.0e6,
        help="Limite de drift para declarar blow-up. Default: 1e6",
    )

    p.add_argument(
        "--include-full-if-possible",
        action="store_true",
        help="Inclui G_full apenas se NE_free <= full_max_n. Útil para smoke pequeno.",
    )

    p.add_argument(
        "--full-max-n",
        type=int,
        default=2500,
        help="n máximo para formar G_full opcional. Default: 2500",
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
        default="teste04_large",
        help="Output CSV prefix.",
    )

    return p.parse_args()


                                                              
      
                                                              

def write_partial_outputs(
    build_rows: List[dict],
    summary_rows: List[dict],
    signal_rows: List[dict],
    energy_rows: List[dict],
    out_prefix: str,
) -> None:
    if build_rows:
        pd.DataFrame(build_rows).to_csv(
            f"{out_prefix}_build.csv",
            index=False,
        )

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            f"{out_prefix}_summary.csv",
            index=False,
        )

    if signal_rows:
        pd.DataFrame(signal_rows).to_csv(
            f"{out_prefix}_signals.csv",
            index=False,
        )

    if energy_rows:
        pd.DataFrame(energy_rows).to_csv(
            f"{out_prefix}_energy_series.csv",
            index=False,
        )


def main() -> None:
    args = parse_args()

    print("\nTeste 04 — Cavidade grande com G_H,3 selected-solve")
    print("=" * 100)
    print(f"meshes                  : {args.meshes}")
    print(f"contrasts               : {args.contrasts}")
    print(f"jitter_values           : {args.jitter_values}")
    print(f"radius                  : {args.radius}")
    print(f"steps                   : {args.steps}")
    print(f"cfl                     : {args.cfl}")
    print(f"sample_stride           : {args.sample_stride}")
    print(f"chunk_size              : {args.chunk_size}")
    print(f"include_full_if_possible: {args.include_full_if_possible}")
    print(f"full_max_n              : {args.full_max_n}")
    print(f"psutil_available        : {psutil is not None}")
    print(f"out_prefix              : {args.out_prefix}")
    print("=" * 100)

    all_build: List[dict] = []
    all_summary: List[dict] = []
    all_signals: List[dict] = []
    all_energy: List[dict] = []
    failures: List[dict] = []

    global_t0 = time.perf_counter()

    for contrast in args.contrasts:
        for jitter in args.jitter_values:
            for nx in args.meshes:
                print("\n" + "-" * 100)
                print(
                    f"Rodando mesh={nx}x{nx} | "
                    f"contrast={contrast:g} | jitter={jitter:g}"
                )
                print("-" * 100)

                rss_before = current_rss_mb()
                t0 = time.perf_counter()

                try:
                    build_rows, summary_rows, signal_rows, energy_rows = run_dynamic_case(
                        nx=int(nx),
                        ny=int(nx),
                        contrast=float(contrast),
                        jitter=float(jitter),
                        radius=int(args.radius),
                        steps=int(args.steps),
                        cfl=float(args.cfl),
                        sample_stride=int(args.sample_stride),
                        seed=int(args.seed),
                        tol=float(args.tol),
                        chunk_size=int(args.chunk_size),
                        pulse_sigma=float(args.pulse_sigma),
                        pulse_x=float(args.pulse_x),
                        pulse_y=float(args.pulse_y),
                        probe_x=float(args.probe_x),
                        probe_y=float(args.probe_y),
                        blowup_tol=float(args.blowup_tol),
                        include_full_if_possible=bool(args.include_full_if_possible),
                        full_max_n=int(args.full_max_n),
                        use_window=not args.no_window,
                        verbose_solve=bool(args.verbose_solve),
                    )

                    elapsed = time.perf_counter() - t0
                    rss_after = current_rss_mb()

                    for row in build_rows:
                        row["rss_before_mb"] = float(rss_before)
                        row["rss_after_mb"] = float(rss_after)
                        row["rss_delta_mb"] = (
                            float(rss_after - rss_before)
                            if np.isfinite(rss_after) and np.isfinite(rss_before)
                            else np.nan
                        )
                        row["wall_time_outer_s"] = float(elapsed)

                    for row in summary_rows:
                        row["rss_before_mb"] = float(rss_before)
                        row["rss_after_mb"] = float(rss_after)
                        row["rss_delta_mb"] = (
                            float(rss_after - rss_before)
                            if np.isfinite(rss_after) and np.isfinite(rss_before)
                            else np.nan
                        )
                        row["wall_time_outer_s"] = float(elapsed)

                    all_build.extend(build_rows)
                    all_summary.extend(summary_rows)
                    all_signals.extend(signal_rows)
                    all_energy.extend(energy_rows)

                    write_partial_outputs(
                        all_build,
                        all_summary,
                        all_signals,
                        all_energy,
                        out_prefix=args.out_prefix,
                    )

                    df_case = pd.DataFrame(summary_rows)

                    print(f"\nConcluído em {elapsed:.3f} s")

                    for _, row in df_case.iterrows():
                        print(
                            f"  {row['method']:<30s} "
                            f"stable={row['stable']} "
                            f"drift={row['max_energy_drift']:.3e} "
                            f"f={row['f_peak']:.6e} "
                            f"corr_ref={row['signal_corr_vs_reference']:.4f} "
                            f"L2sig_ref={row['signal_l2_error_vs_reference']:.3e} "
                            f"update/step={row['runtime_update_per_step_s']:.3e}s "
                            f"nnzG/nnzH={row['nnzG_over_nnzH']:.2f} "
                            f"mem/dense={row['memory_ratio_vs_dense']:.3e}"
                        )

                    if psutil is not None:
                        print(
                            f"RSS antes={rss_before:.1f} MB | "
                            f"RSS depois={rss_after:.1f} MB | "
                            f"delta={rss_after - rss_before:.1f} MB"
                        )

                except Exception as exc:
                    elapsed = time.perf_counter() - t0

                    failure = {
                        "mesh": f"{nx}x{nx}",
                        "nx": int(nx),
                        "contrast": float(contrast),
                        "jitter": float(jitter),
                        "radius": int(args.radius),
                        "steps": int(args.steps),
                        "elapsed_before_fail_s": float(elapsed),
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "rss_before_mb": float(rss_before),
                        "rss_after_mb": float(current_rss_mb()),
                    }

                    failures.append(failure)

                    pd.DataFrame(failures).to_csv(
                        f"{args.out_prefix}_failures.csv",
                        index=False,
                    )

                    print(f"\nERRO: {type(exc).__name__}: {exc}")
                    print("Falha salva em arquivo.")

                    continue

    if not all_summary:
        print("\nNenhum resultado foi gerado.")
        return

    write_partial_outputs(
        all_build,
        all_summary,
        all_signals,
        all_energy,
        out_prefix=args.out_prefix,
    )

    total_time = time.perf_counter() - global_t0

    print("\n" + "=" * 100)
    print("Arquivos gerados:")
    print(f"  {args.out_prefix}_build.csv")
    print(f"  {args.out_prefix}_summary.csv")
    print(f"  {args.out_prefix}_signals.csv")
    print(f"  {args.out_prefix}_energy_series.csv")

    if failures:
        print(f"  {args.out_prefix}_failures.csv")

    print("=" * 100)
    print(f"Tempo total: {total_time:.3f} s")

    df = pd.DataFrame(all_summary)

    cols = [
        "contrast",
        "jitter",
        "mesh",
        "method",
        "reference_method",
        "NE_free",
        "nnzG_over_nnzH",
        "memory_ratio_vs_dense",
        "runtime_build_s",
        "runtime_update_per_step_s",
        "max_energy_drift",
        "energy_final_relative",
        "f_peak",
        "freq_rel_error_vs_reference",
        "signal_corr_vs_reference",
        "signal_l2_error_vs_reference",
        "stable",
    ]

    cols = [c for c in cols if c in df.columns]

    print("\nSíntese final:")
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()

