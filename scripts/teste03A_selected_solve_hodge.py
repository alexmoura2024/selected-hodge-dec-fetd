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
    Selected-solve construction of sparse inverse Hodge approximations without forming the dense inverse.

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
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla

try:
    from teste01D_material_contrast import (
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


                                                              
            
                                                              

def boundary_edge_mask(mesh, tol: float = 1.0e-12) -> np.ndarray:
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


def ensure_symmetric_pattern(P: sp.spmatrix) -> sp.csr_matrix:
    """Helper routine used in the numerical experiments."""
    P = P.tocsr().astype(bool)
    P = P.maximum(P.T).astype(bool).tocsr()
    P.setdiag(True)
    P.eliminate_zeros()
    return P


def sparse_memory_bytes(A: sp.spmatrix) -> int:
    """Helper routine used in the numerical experiments."""
    A = A.tocsr()
    return int(A.data.nbytes + A.indices.nbytes + A.indptr.nbytes)


def dense_memory_bytes(n: int) -> int:
    """Helper routine used in the numerical experiments."""
    return int(n * n * 8)


def method_radius(name: str) -> int:
    if name == "G_diag":
        return 0
    return int(name.split("_")[-1])


def safe_sparse_fro_norm(A: sp.spmatrix) -> float:
    """Helper routine used in the numerical experiments."""
    return float(np.sqrt(np.sum(A.data * A.data)))


def residual_identity_error(
    H: sp.csr_matrix,
    G: sp.csr_matrix,
    max_n: int,
) -> float:
    """Helper routine used in the numerical experiments."""
    n = H.shape[0]

    if max_n <= 0 or n > max_n:
        return np.nan

    I = sp.eye(n, format="csr")
    R = I - (H @ G)

    return safe_sparse_fro_norm(R.tocsr()) / np.sqrt(n)


def lambda_min_symmetric_part(
    G: sp.csr_matrix,
    eig_max_n: int,
    dense_eig_max_n: int,
) -> float:
    """Helper routine used in the numerical experiments."""
    n = G.shape[0]

    if eig_max_n <= 0 or n > eig_max_n:
        return np.nan

    S = ((G + G.T) * 0.5).tocsr()

    try:
        if n <= dense_eig_max_n:
            vals = np.linalg.eigvalsh(S.toarray())
            return float(vals[0])

        val = spla.eigsh(
            S,
            k=1,
            which="SA",
            return_eigenvectors=False,
            tol=1.0e-8,
            maxiter=20000,
        )[0]

        return float(val)

    except Exception:
        return np.nan


                                                              
                
                                                              

def selected_inverse_from_solve(
    lu: spla.SuperLU,
    pattern: sp.spmatrix,
    chunk_size: int = 32,
    tol: float = 1.0e-14,
    symmetrize: bool = True,
    verbose: bool = False,
) -> sp.csr_matrix:
    """Helper routine used in the numerical experiments."""
    P = ensure_symmetric_pattern(pattern)
    P_csc = P.tocsc()

    n = P.shape[0]

    rows_all: List[np.ndarray] = []
    cols_all: List[np.ndarray] = []
    vals_all: List[np.ndarray] = []

    eye_cols = np.arange(n, dtype=np.int64)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        cols = eye_cols[start:end]
        k = cols.size

        if verbose:
            print(f"    selected-solve columns {start}:{end} / {n}")

        B = np.zeros((n, k), dtype=float)
        B[cols, np.arange(k)] = 1.0

        X = lu.solve(B)

        for loc, j in enumerate(cols):
            p0 = P_csc.indptr[j]
            p1 = P_csc.indptr[j + 1]

            rows = P_csc.indices[p0:p1]
            vals = X[rows, loc]

            keep = np.abs(vals) > tol

            if not np.any(keep):
                continue

            rows_keep = rows[keep].astype(np.int64)
            vals_keep = vals[keep].astype(float)
            cols_keep = np.full(rows_keep.size, j, dtype=np.int64)

            rows_all.append(rows_keep)
            cols_all.append(cols_keep)
            vals_all.append(vals_keep)

    if not rows_all:
        return sp.csr_matrix((n, n))

    rows_cat = np.concatenate(rows_all)
    cols_cat = np.concatenate(cols_all)
    vals_cat = np.concatenate(vals_all)

    G = sp.coo_matrix(
        (vals_cat, (rows_cat, cols_cat)),
        shape=(n, n),
    ).tocsr()

    if symmetrize:
        G = ((G + G.T) * 0.5).tocsr()
        G = G.multiply(P.tocsr()).tocsr()

    G = prune_sparse(G, tol=tol)

    return G.tocsr()


                                                              
                                     
                                                              

def action_error_against_solve(
    lu: spla.SuperLU,
    G: sp.csr_matrix,
    nr: int,
    seed: int,
) -> Tuple[float, float]:
    """Helper routine used in the numerical experiments."""
    n = G.shape[0]
    rng = np.random.default_rng(seed)

    V = rng.normal(size=(n, nr))
    Y_exact = lu.solve(V)
    Y_approx = G @ V

    errors = []

    for k in range(nr):
        den = np.linalg.norm(Y_exact[:, k])

        if den < 1.0e-30:
            continue

        err = np.linalg.norm(Y_exact[:, k] - Y_approx[:, k]) / den
        errors.append(float(err))

    if not errors:
        return np.nan, np.nan

    return float(np.mean(errors)), float(np.max(errors))


def dense_validation_error(
    H: sp.csr_matrix,
    G_solve: sp.csr_matrix,
    pattern: sp.spmatrix,
    dense_check_max: int,
) -> float:
    """Helper routine used in the numerical experiments."""
    n = H.shape[0]

    if dense_check_max <= 0 or n > dense_check_max:
        return np.nan

    Hinv = np.linalg.inv(H.toarray())
    G_dense = project_dense_inverse_to_pattern(Hinv, ensure_symmetric_pattern(pattern))

    if not sp.issparse(G_dense):
        G_dense = sp.csr_matrix(G_dense)

    G_dense = G_dense.tocsr()

    num = safe_sparse_fro_norm((G_solve - G_dense).tocsr())
    den = safe_sparse_fro_norm(G_dense)

    if den < 1.0e-30:
        return np.nan

    return float(num / den)


def benchmark_matvec(
    G: sp.csr_matrix,
    reps: int,
    seed: int,
) -> float:
    """Helper routine used in the numerical experiments."""
    if reps <= 0:
        return np.nan

    n = G.shape[0]
    rng = np.random.default_rng(seed)
    v = rng.normal(size=n)

                 
    _ = G @ v

    t0 = time.perf_counter()

    for _ in range(reps):
        _ = G @ v

    return float((time.perf_counter() - t0) / reps)


                                                              
                       
                                                              

def run_case(
    nx: int,
    ny: int,
    contrast: float,
    jitter: float,
    radii: Sequence[int],
    seed: int,
    tol: float,
    chunk_size: int,
    nr: int,
    dense_check_max: int,
    eig_max_n: int,
    dense_eig_max_n: int,
    residual_max_n: int,
    matvec_reps: int,
    max_n: int,
    verbose_solve: bool,
) -> Tuple[List[dict], dict]:

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

    if n > max_n:
        raise MemoryError(
            f"n={n} excede --max-n={max_n}. "
            "Reduza a malha ou aumente --max-n conscientemente."
        )

    t_factor0 = time.perf_counter()
    lu = spla.splu(H.tocsc())
    t_factor = time.perf_counter() - t_factor0

    patterns = radius_patterns_from_H(H, radii=radii, tol=tol)
    patterns = {r: ensure_symmetric_pattern(patterns[r]) for r in radii}

    rows: List[dict] = []

    dense_bytes = dense_memory_bytes(n)

                                                              
                       
                                                              
    t_build0 = time.perf_counter()
    G_diag = diag_inverse(H).tocsr()
    t_build_diag = time.perf_counter() - t_build0

    eact_mean, eact_max = action_error_against_solve(
        lu,
        G_diag,
        nr=nr,
        seed=seed + 17,
    )

    ef = residual_identity_error(H, G_diag, max_n=residual_max_n)
    lmin = lambda_min_symmetric_part(
        G_diag,
        eig_max_n=eig_max_n,
        dense_eig_max_n=dense_eig_max_n,
    )

    t_mv = benchmark_matvec(G_diag, reps=matvec_reps, seed=seed + 101)

    rows.append({
        "mesh": f"{nx}x{ny}",
        "nx": int(nx),
        "ny": int(ny),
        "jitter": float(jitter),
        "contrast": float(contrast),
        "method": "G_diag",
        "radius": 0,
        "NV": int(mesh.points.shape[0]),
        "NE_total": int(mesh.edges.shape[0]),
        "NE_free": int(n),
        "nnz_H": int(H.nnz),
        "rho_H": float(density(H)),
        "nnz_G": int(G_diag.nnz),
        "rho_G": float(density(G_diag)),
        "nnzG_over_nnzH": float(G_diag.nnz / H.nnz),
        "sparse_memory_bytes": int(sparse_memory_bytes(G_diag)),
        "dense_inverse_memory_bytes": int(dense_bytes),
        "memory_ratio_vs_dense": float(sparse_memory_bytes(G_diag) / dense_bytes),
        "runtime_factor_s": float(t_factor),
        "runtime_build_s": float(t_build_diag),
        "runtime_total_case_s": np.nan,
        "matvec_time_s": float(t_mv),
        "Eact_mean_vs_solve": float(eact_mean),
        "Eact_max_vs_solve": float(eact_max),
        "EF_identity": float(ef),
        "lambda_min_symG": float(lmin),
        "dense_validation_relF": np.nan,
    })

                                                              
                                   
                                                              
    for r in radii:
        print(f"    building G_H_radius_{r} por selected-solve ...")

        t_build0 = time.perf_counter()

        G_r = selected_inverse_from_solve(
            lu,
            patterns[r],
            chunk_size=chunk_size,
            tol=tol,
            symmetrize=True,
            verbose=verbose_solve,
        )

        t_build = time.perf_counter() - t_build0

        eact_mean, eact_max = action_error_against_solve(
            lu,
            G_r,
            nr=nr,
            seed=seed + 31 + r,
        )

        ef = residual_identity_error(H, G_r, max_n=residual_max_n)

        lmin = lambda_min_symmetric_part(
            G_r,
            eig_max_n=eig_max_n,
            dense_eig_max_n=dense_eig_max_n,
        )

        dense_val = dense_validation_error(
            H,
            G_r,
            patterns[r],
            dense_check_max=dense_check_max,
        )

        t_mv = benchmark_matvec(G_r, reps=matvec_reps, seed=seed + 151 + r)

        rows.append({
            "mesh": f"{nx}x{ny}",
            "nx": int(nx),
            "ny": int(ny),
            "jitter": float(jitter),
            "contrast": float(contrast),
            "method": f"G_H_radius_{r}_selected_solve",
            "radius": int(r),
            "NV": int(mesh.points.shape[0]),
            "NE_total": int(mesh.edges.shape[0]),
            "NE_free": int(n),
            "nnz_H": int(H.nnz),
            "rho_H": float(density(H)),
            "nnz_G": int(G_r.nnz),
            "rho_G": float(density(G_r)),
            "nnzG_over_nnzH": float(G_r.nnz / H.nnz),
            "sparse_memory_bytes": int(sparse_memory_bytes(G_r)),
            "dense_inverse_memory_bytes": int(dense_bytes),
            "memory_ratio_vs_dense": float(sparse_memory_bytes(G_r) / dense_bytes),
            "runtime_factor_s": float(t_factor),
            "runtime_build_s": float(t_build),
            "runtime_total_case_s": np.nan,
            "matvec_time_s": float(t_mv),
            "Eact_mean_vs_solve": float(eact_mean),
            "Eact_max_vs_solve": float(eact_max),
            "EF_identity": float(ef),
            "lambda_min_symG": float(lmin),
            "dense_validation_relF": float(dense_val),
        })

    total_case_time = time.perf_counter() - t_case0

    for row in rows:
        row["runtime_total_case_s"] = float(total_case_time)

    selected = [row for row in rows if row["radius"] > 0]
    best = min(selected, key=lambda z: z["Eact_mean_vs_solve"]) if selected else None
    r3 = next((row for row in rows if row["radius"] == 3), None)
    diag = next((row for row in rows if row["method"] == "G_diag"), None)

    summary = {
        "mesh": f"{nx}x{ny}",
        "nx": int(nx),
        "ny": int(ny),
        "jitter": float(jitter),
        "contrast": float(contrast),
        "NE_free": int(n),
        "nnz_H": int(H.nnz),
        "rho_H": float(density(H)),
        "runtime_factor_s": float(t_factor),
        "runtime_total_case_s": float(total_case_time),
        "best_method": best["method"] if best else "",
        "best_radius": int(best["radius"]) if best else -1,
        "best_Eact_mean_vs_solve": float(best["Eact_mean_vs_solve"]) if best else np.nan,
        "best_Eact_max_vs_solve": float(best["Eact_max_vs_solve"]) if best else np.nan,
        "best_nnzG_over_nnzH": float(best["nnzG_over_nnzH"]) if best else np.nan,
        "best_build_time_s": float(best["runtime_build_s"]) if best else np.nan,
        "best_memory_ratio_vs_dense": float(best["memory_ratio_vs_dense"]) if best else np.nan,
        "diag_Eact_mean_vs_solve": float(diag["Eact_mean_vs_solve"]) if diag else np.nan,
        "diag_Eact_max_vs_solve": float(diag["Eact_max_vs_solve"]) if diag else np.nan,
        "r3_Eact_mean_vs_solve": float(r3["Eact_mean_vs_solve"]) if r3 else np.nan,
        "r3_Eact_max_vs_solve": float(r3["Eact_max_vs_solve"]) if r3 else np.nan,
        "r3_EF_identity": float(r3["EF_identity"]) if r3 else np.nan,
        "r3_lambda_min_symG": float(r3["lambda_min_symG"]) if r3 else np.nan,
        "r3_dense_validation_relF": float(r3["dense_validation_relF"]) if r3 else np.nan,
        "r3_nnzG_over_nnzH": float(r3["nnzG_over_nnzH"]) if r3 else np.nan,
        "r3_build_time_s": float(r3["runtime_build_s"]) if r3 else np.nan,
        "r3_matvec_time_s": float(r3["matvec_time_s"]) if r3 else np.nan,
        "r3_memory_ratio_vs_dense": float(r3["memory_ratio_vs_dense"]) if r3 else np.nan,
    }

    return rows, summary


                                                              
     
                                                              

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Teste 03A — selected-solve para construir G_H,r "
            "sem formar H^{-1} densa."
        )
    )

    p.add_argument(
        "--meshes",
        nargs="+",
        type=int,
        default=[20, 30, 40, 60],
        help="Square meshes nx=ny. Default: 20 30 40 60",
    )

    p.add_argument(
        "--radii",
        nargs="+",
        type=int,
        default=[3],
        help="Raios topológicos. Default: 3",
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
        "--chunk-size",
        type=int,
        default=32,
        help="Número de colunas resolvidas por bloco. Default: 32",
    )

    p.add_argument(
        "--nr",
        type=int,
        default=10,
        help="Número de vetores aleatórios para erro de ação. Default: 10",
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
        default=20000,
        help="Dimensão máxima permitida para H. Default: 20000",
    )

    p.add_argument(
        "--dense-check-max",
        type=int,
        default=2500,
        help="Até este n, compara com projeção da inversa densa. Default: 2500",
    )

    p.add_argument(
        "--eig-max-n",
        type=int,
        default=2500,
        help="Até este n, calcula lambda_min. Use 0 para pular. Default: 2500",
    )

    p.add_argument(
        "--dense-eig-max-n",
        type=int,
        default=2500,
        help="Até este n, usa autovalor denso. Default: 2500",
    )

    p.add_argument(
        "--residual-max-n",
        type=int,
        default=6000,
        help="Até este n, calcula ||I-HG||_F. Use 0 para pular. Default: 6000",
    )

    p.add_argument(
        "--matvec-reps",
        type=int,
        default=50,
        help="Repetições para benchmark G @ v. Default: 50",
    )

    p.add_argument(
        "--verbose-solve",
        action="store_true",
        help="Print internal progress for block solves.",
    )

    p.add_argument(
        "--out-prefix",
        type=str,
        default="teste03A_selected_solve",
        help="Output CSV prefix.",
    )

    return p.parse_args()


                                                              
      
                                                              

def main() -> None:
    args = parse_args()

    radii = sorted(set(args.radii))

    print("\nTeste 03A — selected-solve da inversa seletiva topológica da Hodge")
    print("=" * 100)
    print(f"meshes          : {args.meshes}")
    print(f"radii           : {radii}")
    print(f"contrasts       : {args.contrasts}")
    print(f"jitter_values   : {args.jitter_values}")
    print(f"chunk_size      : {args.chunk_size}")
    print(f"nr              : {args.nr}")
    print(f"tol             : {args.tol:.1e}")
    print(f"max_n           : {args.max_n}")
    print(f"dense_check_max : {args.dense_check_max}")
    print(f"eig_max_n       : {args.eig_max_n}")
    print(f"residual_max_n  : {args.residual_max_n}")
    print(f"out_prefix      : {args.out_prefix}")
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
                    rows, summary = run_case(
                        nx=int(nx),
                        ny=int(nx),
                        contrast=float(contrast),
                        jitter=float(jitter),
                        radii=radii,
                        seed=int(args.seed),
                        tol=float(args.tol),
                        chunk_size=int(args.chunk_size),
                        nr=int(args.nr),
                        dense_check_max=int(args.dense_check_max),
                        eig_max_n=int(args.eig_max_n),
                        dense_eig_max_n=int(args.dense_eig_max_n),
                        residual_max_n=int(args.residual_max_n),
                        matvec_reps=int(args.matvec_reps),
                        max_n=int(args.max_n),
                        verbose_solve=bool(args.verbose_solve),
                    )

                except Exception as exc:
                    print(f"  ERRO: {exc}")
                    continue

                all_rows.extend(rows)
                summaries.append(summary)

                print(f"Concluído em {time.perf_counter() - t0:.3f} s")
                print(
                    f"  n={summary['NE_free']} | nnzH={summary['nnz_H']} | "
                    f"tfactor={summary['runtime_factor_s']:.3e}s | "
                    f"total={summary['runtime_total_case_s']:.3e}s"
                )

                for row in rows:
                    print(
                        f"  {row['method']:<30s} "
                        f"nnzG/nnzH={row['nnzG_over_nnzH']:.2f} "
                        f"Eact={row['Eact_mean_vs_solve']:.3e} "
                        f"Emax={row['Eact_max_vs_solve']:.3e} "
                        f"EF={row['EF_identity']:.3e} "
                        f"lmin={row['lambda_min_symG']:.3e} "
                        f"build={row['runtime_build_s']:.3e}s "
                        f"mem/dense={row['memory_ratio_vs_dense']:.3e} "
                        f"densecheck={row['dense_validation_relF']:.3e}"
                    )

                print(
                    f"  melhor: {summary['best_method']} | "
                    f"Eact={summary['best_Eact_mean_vs_solve']:.3e} | "
                    f"nnzG/nnzH={summary['best_nnzG_over_nnzH']:.2f}"
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

    print("\nSíntese rápida:")
    cols = [
        "contrast",
        "jitter",
        "mesh",
        "NE_free",
        "best_method",
        "best_Eact_mean_vs_solve",
        "best_nnzG_over_nnzH",
        "best_memory_ratio_vs_dense",
        "r3_Eact_mean_vs_solve",
        "r3_nnzG_over_nnzH",
        "r3_memory_ratio_vs_dense",
        "r3_build_time_s",
    ]

    print(dfs[cols].to_string(index=False))


if __name__ == "__main__":
    main()

