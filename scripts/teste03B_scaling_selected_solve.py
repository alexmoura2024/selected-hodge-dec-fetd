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
    Scaling study of the selected-solve construction for the radius-three inverse Hodge approximation.

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
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import psutil
except Exception:
    psutil = None

try:
    from teste03A_selected_solve_hodge import run_case
except ImportError as exc:
    raise ImportError(
        "Could not import teste03A_selected_solve_hodge.py. "
        "Place this script in the same folder do Teste 03A e rode com o --wdir correto."
    ) from exc


                                                              
            
                                                              

def current_rss_mb() -> float:
    """Helper routine used in the numerical experiments."""
    if psutil is None:
        return np.nan

    try:
        proc = psutil.Process(os.getpid())
        return float(proc.memory_info().rss / (1024.0 ** 2))
    except Exception:
        return np.nan


def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return np.nan


def safe_power_law_fit(
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[float, float, float, int]:
    """Helper routine used in the numerical experiments."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)

    x = x[mask]
    y = y[mask]

    if x.size < 2:
        return np.nan, np.nan, np.nan, int(x.size)

    lx = np.log(x)
    ly = np.log(y)

    p, log_c = np.polyfit(lx, ly, 1)

    pred = p * lx + log_c

    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - np.mean(ly)) ** 2))

    if ss_tot <= 1.0e-30:
        r2 = np.nan
    else:
        r2 = 1.0 - ss_res / ss_tot

    c = float(np.exp(log_c))

    return float(p), c, float(r2), int(x.size)


def write_outputs(
    all_rows: List[dict],
    all_summary: List[dict],
    out_prefix: str,
) -> None:
    """Helper routine used in the numerical experiments."""
    if all_rows:
        pd.DataFrame(all_rows).to_csv(
            f"{out_prefix}_comparison.csv",
            index=False,
        )

    if all_summary:
        pd.DataFrame(all_summary).to_csv(
            f"{out_prefix}_summary.csv",
            index=False,
        )


def build_scaling_laws(
    comparison: pd.DataFrame,
    summary: pd.DataFrame,
    radius: int,
) -> pd.DataFrame:
    """Helper routine used in the numerical experiments."""
    method_name = f"G_H_radius_{radius}_selected_solve"

    r3 = comparison[comparison["method"] == method_name].copy()

    if r3.empty:
        return pd.DataFrame()

    metrics = [
        ("nnz_H", "nnz_H"),
        ("nnz_G", "nnz_G"),
        ("nnzG_over_nnzH", "nnzG_over_nnzH"),
        ("sparse_memory_bytes", "sparse_memory_bytes"),
        ("memory_ratio_vs_dense", "memory_ratio_vs_dense"),
        ("runtime_factor_s", "runtime_factor_s"),
        ("runtime_build_s", "runtime_build_s"),
        ("matvec_time_s", "matvec_time_s"),
        ("Eact_mean_vs_solve", "Eact_mean_vs_solve"),
        ("Eact_max_vs_solve", "Eact_max_vs_solve"),
    ]

    rows: List[dict] = []

    group_cols = ["contrast", "jitter"]

    for (contrast, jitter), df in r3.groupby(group_cols):
        df = df.sort_values("NE_free")

        n = df["NE_free"].to_numpy(dtype=float)

        for metric_name, col in metrics:
            if col not in df.columns:
                continue

            y = df[col].to_numpy(dtype=float)

            p, c, r2, npts = safe_power_law_fit(n, y)

            rows.append({
                "contrast": float(contrast),
                "jitter": float(jitter),
                "radius": int(radius),
                "metric": metric_name,
                "law": f"{metric_name} ~ C n^p",
                "p_exponent": float(p),
                "C_prefactor": float(c),
                "R2_loglog": float(r2),
                "n_points": int(npts),
                "n_min": float(np.nanmin(n)) if n.size else np.nan,
                "n_max": float(np.nanmax(n)) if n.size else np.nan,
                "y_min": float(np.nanmin(y)) if y.size else np.nan,
                "y_max": float(np.nanmax(y)) if y.size else np.nan,
            })

                                                  
    df_all = r3.sort_values("NE_free")
    n_all = df_all["NE_free"].to_numpy(dtype=float)

    for metric_name, col in metrics:
        if col not in df_all.columns:
            continue

        y = df_all[col].to_numpy(dtype=float)

        p, c, r2, npts = safe_power_law_fit(n_all, y)

        rows.append({
            "contrast": np.nan,
            "jitter": np.nan,
            "radius": int(radius),
            "metric": metric_name,
            "law": f"{metric_name} ~ C n^p",
            "p_exponent": float(p),
            "C_prefactor": float(c),
            "R2_loglog": float(r2),
            "n_points": int(npts),
            "n_min": float(np.nanmin(n_all)) if n_all.size else np.nan,
            "n_max": float(np.nanmax(n_all)) if n_all.size else np.nan,
            "y_min": float(np.nanmin(y)) if y.size else np.nan,
            "y_max": float(np.nanmax(y)) if y.size else np.nan,
        })

    return pd.DataFrame(rows)


                                                              
     
                                                              

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Teste 03B — escalabilidade do selected-solve para G_H,3 "
            "without dense inverse."
        )
    )

    p.add_argument(
        "--meshes",
        nargs="+",
        type=int,
        default=[80, 100, 120],
        help="Square meshes nx=ny. Default: 80 100 120",
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
        help="Raio topológico usado no scaling. Default: 3",
    )

    p.add_argument(
        "--chunk-size",
        type=int,
        default=16,
        help="Número de colunas resolvidas por bloco. Default: 16",
    )

    p.add_argument(
        "--nr",
        type=int,
        default=5,
        help="Número de vetores aleatórios para erro de ação. Default: 5",
    )

    p.add_argument(
        "--matvec-reps",
        type=int,
        default=20,
        help="Repetições para benchmark G @ v. Default: 20",
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
        "--max-n",
        type=int,
        default=80000,
        help="Dimensão máxima permitida para H. Default: 80000",
    )

    p.add_argument(
        "--skip-jitter-after-fail",
        action="store_true",
        help=(
            "Se um caso com jitter falhar por memória/tempo, pula os próximos "
            "casos com mesmo contraste e jitter."
        ),
    )

    p.add_argument(
        "--verbose-solve",
        action="store_true",
        help="Print internal progress for block solves.",
    )

    p.add_argument(
        "--out-prefix",
        type=str,
        default="teste03B_scaling",
        help="Output CSV prefix.",
    )

    return p.parse_args()


                                                              
      
                                                              

def main() -> None:
    args = parse_args()

    radius = int(args.radius)

    print("\nTeste 03B — Escalabilidade do selected-solve para G_H,3")
    print("=" * 100)
    print(f"meshes               : {args.meshes}")
    print(f"contrasts            : {args.contrasts}")
    print(f"jitter_values        : {args.jitter_values}")
    print(f"radius               : {radius}")
    print(f"chunk_size           : {args.chunk_size}")
    print(f"nr                   : {args.nr}")
    print(f"matvec_reps          : {args.matvec_reps}")
    print(f"tol                  : {args.tol:.1e}")
    print(f"max_n                : {args.max_n}")
    print(f"psutil_available     : {psutil is not None}")
    print(f"out_prefix           : {args.out_prefix}")
    print("=" * 100)

    all_rows: List[dict] = []
    all_summary: List[dict] = []
    failures: List[dict] = []

    global_t0 = time.perf_counter()

    for contrast in args.contrasts:
        for jitter in args.jitter_values:
            failed_this_group = False

            print(f"\n### Grupo: contraste={contrast:g} | jitter={jitter:g} ###")

            for nx in args.meshes:
                if failed_this_group and args.skip_jitter_after_fail:
                    print(
                        f"Skipping {nx}x{nx} because group contraste={contrast:g}, "
                        f"jitter={jitter:g} já failed."
                    )
                    continue

                print("\n" + "-" * 100)
                print(f"Rodando malha {nx}x{nx} | contraste={contrast:g} | jitter={jitter:g}")
                print("-" * 100)

                rss_before = current_rss_mb()
                t0 = time.perf_counter()

                try:
                    rows, summary = run_case(
                        nx=int(nx),
                        ny=int(nx),
                        contrast=float(contrast),
                        jitter=float(jitter),
                        radii=[radius],
                        seed=int(args.seed),
                        tol=float(args.tol),
                        chunk_size=int(args.chunk_size),
                        nr=int(args.nr),
                        dense_check_max=0,
                        eig_max_n=0,
                        dense_eig_max_n=0,
                        residual_max_n=0,
                        matvec_reps=int(args.matvec_reps),
                        max_n=int(args.max_n),
                        verbose_solve=bool(args.verbose_solve),
                    )

                    elapsed = time.perf_counter() - t0
                    rss_after = current_rss_mb()

                    for row in rows:
                        row["rss_before_mb"] = float(rss_before)
                        row["rss_after_mb"] = float(rss_after)
                        row["rss_delta_mb"] = (
                            float(rss_after - rss_before)
                            if np.isfinite(rss_after) and np.isfinite(rss_before)
                            else np.nan
                        )
                        row["wall_time_outer_s"] = float(elapsed)

                    summary["rss_before_mb"] = float(rss_before)
                    summary["rss_after_mb"] = float(rss_after)
                    summary["rss_delta_mb"] = (
                        float(rss_after - rss_before)
                        if np.isfinite(rss_after) and np.isfinite(rss_before)
                        else np.nan
                    )
                    summary["wall_time_outer_s"] = float(elapsed)

                    all_rows.extend(rows)
                    all_summary.append(summary)

                    write_outputs(
                        all_rows,
                        all_summary,
                        out_prefix=args.out_prefix,
                    )

                    print(f"\nConcluído em {elapsed:.3f} s")
                    print(
                        f"n={summary['NE_free']} | nnzH={summary['nnz_H']} | "
                        f"tfactor={summary['runtime_factor_s']:.3e}s | "
                        f"build_r3={summary['r3_build_time_s']:.3e}s | "
                        f"Eact_r3={summary['r3_Eact_mean_vs_solve']:.3e} | "
                        f"nnzG/nnzH={summary['r3_nnzG_over_nnzH']:.2f} | "
                        f"mem/dense={summary['r3_memory_ratio_vs_dense']:.3e}"
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
                        "radius": int(radius),
                        "elapsed_before_fail_s": float(elapsed),
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "rss_before_mb": float(rss_before),
                        "rss_after_mb": float(current_rss_mb()),
                    }

                    failures.append(failure)
                    failed_this_group = True

                    pd.DataFrame(failures).to_csv(
                        f"{args.out_prefix}_failures.csv",
                        index=False,
                    )

                    print(f"\nERRO no caso {nx}x{nx}: {type(exc).__name__}: {exc}")
                    print("O erro foi salvo em arquivo de falhas.")

                    continue

    if not all_rows:
        print("\nNenhum resultado foi gerado.")
        return

    comparison = pd.DataFrame(all_rows)
    summary = pd.DataFrame(all_summary)

    comparison_path = f"{args.out_prefix}_comparison.csv"
    summary_path = f"{args.out_prefix}_summary.csv"
    laws_path = f"{args.out_prefix}_scaling_laws.csv"

    comparison.to_csv(comparison_path, index=False)
    summary.to_csv(summary_path, index=False)

    laws = build_scaling_laws(
        comparison=comparison,
        summary=summary,
        radius=radius,
    )

    laws.to_csv(laws_path, index=False)

    total_time = time.perf_counter() - global_t0

    print("\n" + "=" * 100)
    print("Arquivos gerados:")
    print(f"  {comparison_path}")
    print(f"  {summary_path}")
    print(f"  {laws_path}")

    if failures:
        print(f"  {args.out_prefix}_failures.csv")

    print("=" * 100)
    print(f"Tempo total: {total_time:.3f} s")

    print("\nSíntese dos casos:")
    cols = [
        "contrast",
        "jitter",
        "mesh",
        "NE_free",
        "nnz_H",
        "runtime_factor_s",
        "r3_build_time_s",
        "r3_matvec_time_s",
        "r3_Eact_mean_vs_solve",
        "r3_Eact_max_vs_solve",
        "r3_nnzG_over_nnzH",
        "r3_memory_ratio_vs_dense",
    ]

    cols = [c for c in cols if c in summary.columns]

    print(summary[cols].to_string(index=False))

    print("\nLeis de escala log-log:")
    if not laws.empty:
        law_cols = [
            "contrast",
            "jitter",
            "metric",
            "p_exponent",
            "R2_loglog",
            "n_points",
            "n_min",
            "n_max",
            "y_min",
            "y_max",
        ]

        print(laws[law_cols].to_string(index=False))
    else:
        print("Não foi possível ajustar leis de escala.")


if __name__ == "__main__":
    main()

