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
    Modal and frequency-domain analysis of selected inverse Hodge approximations in a two-dimensional PEC cavity.

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
    x0, y0 = coords[0]
    x1, y1 = coords[1]
    x2, y2 = coords[2]

    return 0.5 * ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))


def boundary_edge_mask(mesh: TriMesh2D, tol: float = 1.0e-12) -> np.ndarray:
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
    centroids = mesh.points[mesh.triangles].mean(axis=1)

    dx = centroids[:, 0] - center[0]
    dy = centroids[:, 1] - center[1]

    B = np.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))

    return B * areas


def nearest_face_probe(
    mesh: TriMesh2D,
    x: float,
    y: float,
) -> Tuple[int, float, float]:
    centroids = mesh.points[mesh.triangles].mean(axis=1)

    dx = centroids[:, 0] - x
    dy = centroids[:, 1] - y

    idx = int(np.argmin(dx * dx + dy * dy))

    return idx, float(centroids[idx, 0]), float(centroids[idx, 1])


                                                              
                          
                                                              

def h_norm(x: np.ndarray, H: sp.csr_matrix) -> float:
    val = float(x @ (H @ x))
    return float(np.sqrt(max(val, 0.0)))


def m2_norm(x: np.ndarray, M2: sp.csr_matrix) -> float:
    val = float(x @ (M2 @ x))
    return float(np.sqrt(max(val, 0.0)))


def electromagnetic_energy(
    e: np.ndarray,
    b: np.ndarray,
    H: sp.csr_matrix,
    M2: sp.csr_matrix,
) -> float:
    return 0.5 * float(e @ (H @ e)) + 0.5 * float(b @ (M2 @ b))


def apply_G(G: ArrayOrSparse, rhs: np.ndarray) -> np.ndarray:
    return G @ rhs


def estimate_lambda_max_full(K: sp.csr_matrix, H: sp.csr_matrix) -> float:
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
    if sp.issparse(G):
        return int(G.nnz)

    return int(G.shape[0] * G.shape[1])


def method_density(G: ArrayOrSparse) -> float:
    if sp.issparse(G):
        return density(G)

    return 1.0


def method_radius(name: str) -> int:
    if name in ("G_full", "G_diag"):
        return 0

    return int(name.split("_")[-1])


                                                              
                     
                                                              

def dominant_frequency_fft(
    signal: np.ndarray,
    dt_sample: float,
    use_window: bool = True,
) -> Dict[str, float]:
    """Helper routine used in the numerical experiments."""
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
        window = np.hanning(y.size)
        y = y * window

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
        "peak_index": k,
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


def estimate_time_lag_steps(a: np.ndarray, ref: np.ndarray) -> int:
    """Helper routine used in the numerical experiments."""
    aa = np.asarray(a, dtype=float) - np.mean(a)
    rr = np.asarray(ref, dtype=float) - np.mean(ref)

    if np.linalg.norm(aa) < 1.0e-30 or np.linalg.norm(rr) < 1.0e-30:
        return 0

    corr = np.correlate(aa, rr, mode="full")

    lag = int(np.argmax(corr) - (len(ref) - 1))

    return lag


                                                              
                      
                                                              

def run_modal_case(
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
    probe_x: float,
    probe_y: float,
    max_dense_n: int,
    blowup_tol: float,
    use_window: bool,
) -> Tuple[List[dict], List[dict], List[dict]]:
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

    probe_face, probe_cx, probe_cy = nearest_face_probe(
        mesh,
        probe_x,
        probe_y,
    )

    states: Dict[str, dict] = {}

    for name in methods:
        E0 = electromagnetic_energy(e0, b0, H, M2)

        states[name] = {
            "e": e0.copy(),
            "b": b0.copy(),
            "E0": float(E0),
            "max_energy_drift": 0.0,
            "final_energy_drift": 0.0,
            "stable": True,
            "signal": [],
            "time": [],
            "step": [],
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
            t_now = it * dt

            for name, st in states.items():
                b_probe_flux = st["b"][probe_face]
                b_probe_density = b_probe_flux / areas[probe_face]

                st["signal"].append(float(b_probe_density))
                st["time"].append(float(t_now))
                st["step"].append(int(it))

    runtime_dynamics_s = time.perf_counter() - t_dyn0
    runtime_total_s = time.perf_counter() - t_case0

    signal_rows: List[dict] = []
    peak_rows: List[dict] = []
    summary_rows: List[dict] = []

    ref_signal = np.asarray(states["G_full"]["signal"], dtype=float)

    dt_sample = dt * sample_stride

    ref_fft = dominant_frequency_fft(
        ref_signal,
        dt_sample=dt_sample,
        use_window=use_window,
    )

    ref_f_peak = ref_fft["f_peak"]

    for name, G in methods.items():
        st = states[name]

        signal = np.asarray(st["signal"], dtype=float)
        times = np.asarray(st["time"], dtype=float)
        steps_arr = np.asarray(st["step"], dtype=int)

        fft_info = dominant_frequency_fft(
            signal,
            dt_sample=dt_sample,
            use_window=use_window,
        )

        if np.isfinite(ref_f_peak) and abs(ref_f_peak) > 1.0e-30:
            freq_rel_error = abs(fft_info["f_peak"] - ref_f_peak) / abs(ref_f_peak)
        else:
            freq_rel_error = np.nan

        sig_corr = time_correlation(signal, ref_signal)
        sig_l2_error = relative_l2_signal_error(signal, ref_signal)
        lag_steps_sampled = estimate_time_lag_steps(signal, ref_signal)

        lag_physical_time = lag_steps_sampled * dt_sample

        nnz_g = method_nnz(G)
        rho_g = method_density(G)

        En_final = electromagnetic_energy(st["e"], st["b"], H, M2)
        Erel_final = En_final / st["E0"] if st["E0"] > 0.0 else np.nan

        summary_rows.append({
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
            "dt_sample": float(dt_sample),
            "cfl": float(cfl),
            "steps": int(steps),
            "sample_stride": int(sample_stride),
            "n_samples": int(signal.size),
            "t_final": float(steps * dt),
            "probe_face": int(probe_face),
            "probe_x_requested": float(probe_x),
            "probe_y_requested": float(probe_y),
            "probe_x_actual": float(probe_cx),
            "probe_y_actual": float(probe_cy),
            "f_peak": float(fft_info["f_peak"]),
            "f_full": float(ref_f_peak),
            "freq_abs_error_vs_full": float(abs(fft_info["f_peak"] - ref_f_peak))
                if np.isfinite(fft_info["f_peak"]) and np.isfinite(ref_f_peak)
                else np.nan,
            "freq_rel_error_vs_full": float(freq_rel_error),
            "peak_amplitude": float(fft_info["peak_amplitude"]),
            "peak_index": int(fft_info["peak_index"]),
            "df": float(fft_info["df"]),
            "signal_corr_vs_full": float(sig_corr),
            "signal_l2_error_vs_full": float(sig_l2_error),
            "lag_steps_sampled_vs_full": int(lag_steps_sampled),
            "lag_time_vs_full": float(lag_physical_time),
            "energy_initial": float(st["E0"]),
            "energy_final": float(En_final),
            "energy_final_relative": float(Erel_final),
            "final_energy_drift": float(st["final_energy_drift"]),
            "max_energy_drift": float(st["max_energy_drift"]),
            "stable": bool(st["stable"]),
            "runtime_dynamics_s": float(runtime_dynamics_s),
            "runtime_total_case_s": float(runtime_total_s),
        })

        peak_rows.append({
            "mesh": f"{nx}x{ny}",
            "jitter": float(jitter),
            "contrast": float(contrast),
            "method": name,
            "f_peak": float(fft_info["f_peak"]),
            "f_full": float(ref_f_peak),
            "freq_rel_error_vs_full": float(freq_rel_error),
            "peak_amplitude": float(fft_info["peak_amplitude"]),
            "df": float(fft_info["df"]),
            "signal_corr_vs_full": float(sig_corr),
            "signal_l2_error_vs_full": float(sig_l2_error),
            "lag_time_vs_full": float(lag_physical_time),
            "stable": bool(st["stable"]),
        })

        for k in range(signal.size):
            signal_rows.append({
                "mesh": f"{nx}x{ny}",
                "nx": int(nx),
                "ny": int(ny),
                "jitter": float(jitter),
                "contrast": float(contrast),
                "method": name,
                "step": int(steps_arr[k]),
                "time": float(times[k]),
                "probe_bz": float(signal[k]),
                "probe_face": int(probe_face),
                "probe_x": float(probe_cx),
                "probe_y": float(probe_cy),
            })

    return signal_rows, summary_rows, peak_rows


                                                              
     
                                                              

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Teste 02C — Análise modal/frequencial da cavidade PEC "
            "com inversa seletiva topológica da Hodge."
        )
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
        default=8000,
        help="Número de passos de tempo. Default: 8000",
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
        default=1,
        help="Intervalo de amostragem do sinal. Default: 1",
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
        "--probe-x",
        type=float,
        default=0.65,
        help="Coordenada x da sonda Bz. Default: 0.65",
    )

    p.add_argument(
        "--probe-y",
        type=float,
        default=0.50,
        help="Coordenada y da sonda Bz. Default: 0.50",
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
        help="Limite de drift para declarar blow-up. Default: 1e6",
    )

    p.add_argument(
        "--no-window",
        action="store_true",
        help="Desativa janela Hann antes da FFT.",
    )

    p.add_argument(
        "--out-prefix",
        type=str,
        default="teste02C_modal",
        help="Output CSV prefix.",
    )

    return p.parse_args()


                                                              
      
                                                              

def main() -> None:
    args = parse_args()

    radii = sorted(set(args.radii))

    print("\nTeste 02C — Análise modal/frequencial da cavidade PEC 2D")
    print("=" * 100)
    print(f"mesh          : {args.mesh}x{args.mesh}")
    print(f"radii         : {radii}")
    print(f"contrasts     : {args.contrasts}")
    print(f"jitter_values : {args.jitter_values}")
    print(f"steps         : {args.steps}")
    print(f"cfl           : {args.cfl}")
    print(f"sample_stride : {args.sample_stride}")
    print(f"probe         : ({args.probe_x}, {args.probe_y})")
    print(f"pulse         : center=({args.pulse_x}, {args.pulse_y}), sigma={args.pulse_sigma}")
    print(f"tol           : {args.tol:.1e}")
    print(f"max_dense_n   : {args.max_dense_n}")
    print(f"use_window    : {not args.no_window}")
    print(f"out_prefix    : {args.out_prefix}")
    print("=" * 100)

    all_signals: List[dict] = []
    all_summary: List[dict] = []
    all_peaks: List[dict] = []

    for contrast in args.contrasts:
        for jitter in args.jitter_values:
            print(f"\n### Contraste eps_right={contrast:g} | jitter={jitter:g} ###")

            t0 = time.perf_counter()

            try:
                sig_rows, sum_rows, peak_rows = run_modal_case(
                    nx=int(args.mesh),
                    ny=int(args.mesh),
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
                    probe_x=float(args.probe_x),
                    probe_y=float(args.probe_y),
                    max_dense_n=int(args.max_dense_n),
                    blowup_tol=float(args.blowup_tol),
                    use_window=not args.no_window,
                )

            except Exception as exc:
                print(f"  ERRO: {exc}")
                continue

            all_signals.extend(sig_rows)
            all_summary.extend(sum_rows)
            all_peaks.extend(peak_rows)

            print(f"Concluído em {time.perf_counter() - t0:.3f} s")

            df_case = pd.DataFrame(sum_rows)

            f_full = float(
                df_case.loc[df_case["method"] == "G_full", "f_peak"].iloc[0]
            )

            print(f"  f_full = {f_full:.8e}")

            for _, row in df_case.iterrows():
                print(
                    f"  {row['method']:<14s} "
                    f"f={row['f_peak']:.8e} "
                    f"relerr_f={row['freq_rel_error_vs_full']:.3e} "
                    f"corr={row['signal_corr_vs_full']:.4f} "
                    f"L2sig={row['signal_l2_error_vs_full']:.3e} "
                    f"drift={row['max_energy_drift']:.3e} "
                    f"stable={row['stable']}"
                )

    if not all_summary:
        print("\nNenhum resultado foi gerado.")
        return

    df_sig = pd.DataFrame(all_signals)
    df_sum = pd.DataFrame(all_summary)
    df_peak = pd.DataFrame(all_peaks)

    sig_path = f"{args.out_prefix}_signals.csv"
    sum_path = f"{args.out_prefix}_summary.csv"
    peak_path = f"{args.out_prefix}_peaks.csv"

    df_sig.to_csv(sig_path, index=False)
    df_sum.to_csv(sum_path, index=False)
    df_peak.to_csv(peak_path, index=False)

    print("\nArquivos gerados:")
    print(f"  {sig_path}")
    print(f"  {sum_path}")
    print(f"  {peak_path}")

    print("\nSíntese rápida:")
    cols = [
        "contrast",
        "jitter",
        "method",
        "f_peak",
        "freq_rel_error_vs_full",
        "signal_corr_vs_full",
        "signal_l2_error_vs_full",
        "max_energy_drift",
        "nnzG_over_nnzH",
        "stable",
    ]

    print(df_sum[cols].to_string(index=False))


if __name__ == "__main__":
    main()
