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
    Algebraic smoke test for selected inverse Hodge approximations on Whitney--Hodge matrices.

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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import reverse_cuthill_mckee


                                                              
                              
                                                              

@dataclass(frozen=True)
class TriMesh2D:
    points: np.ndarray                         
    triangles: np.ndarray                           
    edges: np.ndarray                                                          
    tri_edges: np.ndarray                                         
    tri_edge_signs: np.ndarray                                                       


def build_structured_tri_mesh(nx: int, ny: int, jitter: float = 0.0, seed: int = 1234) -> TriMesh2D:
    """Helper routine used in the numerical experiments."""
    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be positive.")
    if jitter < 0:
        raise ValueError("jitter must be nonnegative.")

    xs = np.linspace(0.0, 1.0, nx + 1)
    ys = np.linspace(0.0, 1.0, ny + 1)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    points = np.column_stack([X.ravel(), Y.ravel()])

    if jitter > 0.0:
        rng = np.random.default_rng(seed)
        hx = 1.0 / nx
        hy = 1.0 / ny
        for j in range(1, ny):
            for i in range(1, nx):
                idx = j * (nx + 1) + i
                points[idx, 0] += jitter * hx * (rng.random() - 0.5)
                points[idx, 1] += jitter * hy * (rng.random() - 0.5)

    def vid(i: int, j: int) -> int:
        return j * (nx + 1) + i

    triangles: List[Tuple[int, int, int]] = []
    for j in range(ny):
        for i in range(nx):
            v00 = vid(i, j)
            v10 = vid(i + 1, j)
            v01 = vid(i, j + 1)
            v11 = vid(i + 1, j + 1)
                                
            triangles.append((v00, v10, v11))
            triangles.append((v00, v11, v01))

    triangles_arr = np.asarray(triangles, dtype=np.int64)

    edge_to_id: Dict[Tuple[int, int], int] = {}
    edges: List[Tuple[int, int]] = []
    tri_edges: List[List[int]] = []
    tri_signs: List[List[int]] = []

                                                              
    local_pairs = [(0, 1), (1, 2), (2, 0)]

    for tri in triangles_arr:
        eids: List[int] = []
        signs: List[int] = []
        for a, b in local_pairs:
            va = int(tri[a])
            vb = int(tri[b])
            key = (min(va, vb), max(va, vb))
            if key not in edge_to_id:
                edge_to_id[key] = len(edges)
                edges.append(key)
            eid = edge_to_id[key]
            sign = +1 if (va, vb) == key else -1
            eids.append(eid)
            signs.append(sign)
        tri_edges.append(eids)
        tri_signs.append(signs)

    return TriMesh2D(
        points=points,
        triangles=triangles_arr,
        edges=np.asarray(edges, dtype=np.int64),
        tri_edges=np.asarray(tri_edges, dtype=np.int64),
        tri_edge_signs=np.asarray(tri_signs, dtype=np.int8),
    )


                                                              
                  
                                                              

def triangle_area_and_grads(coords: np.ndarray) -> Tuple[float, np.ndarray]:
    """Helper routine used in the numerical experiments."""
    x0, y0 = coords[0]
    x1, y1 = coords[1]
    x2, y2 = coords[2]
    det = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    area = 0.5 * det
    if area <= 0:
        raise ValueError("Triangle is not positively oriented or has zero area.")

                                                         
    grads = np.empty((3, 2), dtype=float)
    grads[0] = np.array([y1 - y2, x2 - x1]) / (2.0 * area)
    grads[1] = np.array([y2 - y0, x0 - x2]) / (2.0 * area)
    grads[2] = np.array([y0 - y1, x1 - x0]) / (2.0 * area)
    return float(area), grads


def int_lambda_lambda(area: float, a: int, b: int) -> float:
    """Helper routine used in the numerical experiments."""
    return area / 6.0 if a == b else area / 12.0


def local_whitney_h1_mass(coords: np.ndarray, eps: float = 1.0) -> np.ndarray:
    """Helper routine used in the numerical experiments."""
    area, grads = triangle_area_and_grads(coords)
    pairs = [(0, 1), (1, 2), (2, 0)]
    M = np.zeros((3, 3), dtype=float)

    for r, (a, b) in enumerate(pairs):
        for s, (c, d) in enumerate(pairs):
            term = 0.0
            term += int_lambda_lambda(area, a, c) * float(np.dot(grads[b], grads[d]))
            term -= int_lambda_lambda(area, a, d) * float(np.dot(grads[b], grads[c]))
            term -= int_lambda_lambda(area, b, c) * float(np.dot(grads[a], grads[d]))
            term += int_lambda_lambda(area, b, d) * float(np.dot(grads[a], grads[c]))
            M[r, s] = eps * term
    return M


def assemble_hodge_h1(mesh: TriMesh2D, eps: float = 1.0) -> sp.csr_matrix:
    """Helper routine used in the numerical experiments."""
    ne = mesh.edges.shape[0]
    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []

    for t, tri in enumerate(mesh.triangles):
        coords = mesh.points[tri]
        Mloc = local_whitney_h1_mass(coords, eps=eps)
        eids = mesh.tri_edges[t]
        signs = mesh.tri_edge_signs[t].astype(float)
        Mloc = (signs[:, None] * Mloc) * signs[None, :]
        for a in range(3):
            for b in range(3):
                rows.append(int(eids[a]))
                cols.append(int(eids[b]))
                vals.append(float(Mloc[a, b]))

    H = sp.coo_matrix((vals, (rows, cols)), shape=(ne, ne)).tocsr()
    H.sum_duplicates()
    H.eliminate_zeros()
    return H


                                                              
                                  
                                                              

def density(A: sp.spmatrix | np.ndarray) -> float:
    n, m = A.shape
    if sp.issparse(A):
        nnz = A.nnz
    else:
        nnz = int(np.count_nonzero(A))
    return float(nnz) / float(n * m)


def sparse_pattern(A: sp.spmatrix, include_diagonal: bool = True) -> sp.csr_matrix:
    """Helper routine used in the numerical experiments."""
    P = A.copy().tocsr()
    P.data = np.ones_like(P.data, dtype=np.float64)
    if include_diagonal:
        P = P + sp.eye(P.shape[0], format="csr")
        P.data[:] = 1.0
    P.eliminate_zeros()
    return P


def symmetrize_pattern(P: sp.spmatrix) -> sp.csr_matrix:
    Q = ((P + P.T) > 0).astype(np.float64).tocsr()
    Q.setdiag(1.0)
    Q.eliminate_zeros()
    return Q


def inverse_project_from_dense(H: sp.csr_matrix, P: sp.csr_matrix) -> sp.csr_matrix:
    """Helper routine used in the numerical experiments."""
    H_dense = H.toarray()
    G_full = np.linalg.inv(H_dense)
    Pcoo = P.tocoo()
    values = G_full[Pcoo.row, Pcoo.col]
    G = sp.coo_matrix((values, (Pcoo.row, Pcoo.col)), shape=H.shape).tocsr()
    G.eliminate_zeros()
    return G


def selected_inverse_by_factorized_solves(H: sp.csr_matrix, P: sp.csr_matrix) -> sp.csr_matrix:
    """Helper routine used in the numerical experiments."""
    n = H.shape[0]
    lu = spla.splu(H.tocsc(), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0)
    Pcsc = P.tocsc()

    rows_out: List[int] = []
    cols_out: List[int] = []
    vals_out: List[float] = []

    e = np.zeros(n, dtype=float)
    for j in range(n):
        start, end = Pcsc.indptr[j], Pcsc.indptr[j + 1]
        rows = Pcsc.indices[start:end]
        if rows.size == 0:
            continue
        e[j] = 1.0
        x = lu.solve(e)
        e[j] = 0.0
        rows_out.extend(rows.tolist())
        cols_out.extend([j] * int(rows.size))
        vals_out.extend(x[rows].tolist())

    G = sp.coo_matrix((vals_out, (rows_out, cols_out)), shape=(n, n)).tocsr()
                                                                   
    G = 0.5 * (G + G.T)
    G.eliminate_zeros()
    return G.tocsr()


def diag_inverse(H: sp.csr_matrix) -> sp.csr_matrix:
    d = H.diagonal()
    if np.any(d <= 0):
        raise ValueError("H has nonpositive diagonal entries.")
    return sp.diags(1.0 / d, format="csr")


def lu_fill_pattern(H: sp.csr_matrix) -> sp.csr_matrix:
    """Helper routine used in the numerical experiments."""
    lu = spla.splu(H.tocsc(), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0)
    Lp = sparse_pattern(lu.L.tocsr(), include_diagonal=True)
    Up = sparse_pattern(lu.U.T.tocsr(), include_diagonal=True)
    P = symmetrize_pattern(Lp + Up)
    return P


def fro_identity_error(H: sp.csr_matrix, G: sp.csr_matrix) -> float:
    n = H.shape[0]
    R = sp.eye(n, format="csr") - H @ G
    return float(spla.norm(R, ord="fro") / np.sqrt(n))


def action_error(H_lu: spla.SuperLU, G: sp.csr_matrix, nr: int = 20, seed: int = 2026) -> float:
    rng = np.random.default_rng(seed)
    n = G.shape[0]
    errs = []
    for _ in range(nr):
        r = rng.standard_normal(n)
        r /= np.linalg.norm(r)
        y_true = H_lu.solve(r)
        y_approx = G @ r
        denom = np.linalg.norm(y_true)
        if denom == 0.0:
            continue
        errs.append(np.linalg.norm(y_true - y_approx) / denom)
    return float(np.mean(errs)) if errs else np.nan


def diag_error(H_lu: spla.SuperLU, G: sp.csr_matrix, max_n_for_diag: int = 5000) -> float:
    """Helper routine used in the numerical experiments."""
    n = G.shape[0]
    if n > max_n_for_diag:
        return np.nan
    true_diag = np.empty(n, dtype=float)
    e = np.zeros(n, dtype=float)
    for j in range(n):
        e[j] = 1.0
        x = H_lu.solve(e)
        e[j] = 0.0
        true_diag[j] = x[j]
    approx_diag = G.diagonal()
    return float(np.linalg.norm(true_diag - approx_diag) / np.linalg.norm(true_diag))


def eig_bounds_symmetric(G: sp.csr_matrix, dense_limit: int = 2500) -> Tuple[float, float]:
    """Helper routine used in the numerical experiments."""
    n = G.shape[0]
    Gs = (0.5 * (G + G.T)).tocsr()
    if n <= dense_limit:
        vals = np.linalg.eigvalsh(Gs.toarray())
        return float(vals[0]), float(vals[-1])
    try:
        lam_min = float(spla.eigsh(Gs, k=1, which="SA", return_eigenvectors=False, tol=1e-7)[0])
        lam_max = float(spla.eigsh(Gs, k=1, which="LA", return_eigenvectors=False, tol=1e-7)[0])
        return lam_min, lam_max
    except Exception:
        return np.nan, np.nan


def dense_inverse_time(H: sp.csr_matrix, max_dense_n: int) -> Tuple[float, bool]:
    n = H.shape[0]
    if n > max_dense_n:
        return np.nan, False
    t0 = time.perf_counter()
    _ = np.linalg.inv(H.toarray())
    return time.perf_counter() - t0, True


                                                              
                       
                                                              

def run_case(nx: int, ny: int, mode: str, jitter: float, nr: int, max_dense_n: int, out_prefix: str) -> Tuple[List[dict], List[dict]]:
    mesh = build_structured_tri_mesh(nx, ny, jitter=jitter)
    H = assemble_hodge_h1(mesh).tocsr()
    n = H.shape[0]
    nv = mesh.points.shape[0]
    nf = mesh.triangles.shape[0]
    ne = mesh.edges.shape[0]

                                                                                     
    perm = reverse_cuthill_mckee(H, symmetric_mode=True)
    H_rcm = H[perm, :][:, perm].tocsr()
    H = H_rcm

    H_lu = spla.splu(H.tocsc(), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0)

    P_H = symmetrize_pattern(sparse_pattern(H, include_diagonal=True))
    P_L = lu_fill_pattern(H)

    methods = {
        "G_H_pattern": P_H,
        "G_L_fill": P_L,
    }

    results: List[dict] = []
    comparisons: List[dict] = []

                                                         
    t_full, has_dense = dense_inverse_time(H, max_dense_n=max_dense_n)

                         
    t0 = time.perf_counter()
    G_diag = diag_inverse(H)
    t_build_diag = time.perf_counter() - t0

    all_G: Dict[str, Tuple[sp.csr_matrix, float]] = {"G_diag": (G_diag, t_build_diag)}

    for name, P in methods.items():
        t0 = time.perf_counter()
        if mode == "dense":
            if n > max_dense_n:
                raise RuntimeError(
                    f"mode='dense' requested, but n={n} > max_dense_n={max_dense_n}. "
                    "Use --mode selected-solve or increase --max-dense-n."
                )
            G = inverse_project_from_dense(H, P)
        elif mode == "selected-solve":
            G = selected_inverse_by_factorized_solves(H, P)
        elif mode == "auto":
            if n <= max_dense_n:
                G = inverse_project_from_dense(H, P)
            else:
                G = selected_inverse_by_factorized_solves(H, P)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        t_build = time.perf_counter() - t0
        all_G[name] = (G, t_build)

    for name, (G, t_build) in all_G.items():
        ef = fro_identity_error(H, G)
        eact = action_error(H_lu, G, nr=nr)
        ediag = diag_error(H_lu, G, max_n_for_diag=max_dense_n)
        lam_min, lam_max = eig_bounds_symmetric(G)
        kappa = lam_max / lam_min if np.isfinite(lam_min) and lam_min > 0 else np.nan

        row = {
            "mesh": f"{nx}x{ny}",
            "nx": nx,
            "ny": ny,
            "jitter": jitter,
            "NV": nv,
            "NE": ne,
            "NF": nf,
            "n": n,
            "method": name,
            "nnz_H": H.nnz,
            "rho_H": density(H),
            "nnz_G": G.nnz,
            "rho_G": density(G),
            "EF_identity": ef,
            "Ediag": ediag,
            "Eact": eact,
            "lambda_min_symG": lam_min,
            "lambda_max_symG": lam_max,
            "kappa_symG": kappa,
            "t_build_s": t_build,
            "t_full_inverse_s": t_full,
            "mode": mode,
        }
        comparisons.append(row)
        if name == "G_H_pattern":
            results.append(row.copy())

    return results, comparisons


                                                              
     
                                                              

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Teste 01 — Inversão seletiva no padrão da matriz de Hodge.")
    parser.add_argument("--meshes", nargs="+", type=int, default=[8, 12, 16, 20],
                        help="Lista de refinamentos Nx=Ny. Ex.: --meshes 8 12 16 20")
    parser.add_argument("--mode", choices=["auto", "dense", "selected-solve"], default="auto",
                        help="dense: inverte H inteira e projeta; selected-solve: calcula entradas por solves; auto: dense até max_dense_n.")
    parser.add_argument("--jitter", type=float, default=0.0,
                        help="Perturbação dos vértices internos. Use 0 para malha estruturada.")
    parser.add_argument("--nr", type=int, default=20,
                        help="Número de vetores aleatórios para erro de ação.")
    parser.add_argument("--max-dense-n", type=int, default=2500,
                        help="Dimensão máxima para inversão densa e diagonal exata completa.")
    parser.add_argument("--outdir", type=str, default=".", help="Diretório de saída.")
    parser.add_argument("--prefix", type=str, default="teste01_selected_hodge", help="Prefixo dos CSVs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[dict] = []
    comparison_rows: List[dict] = []

    print("\nTeste 01 — Inversão seletiva no padrão da matriz de Hodge")
    print("=" * 72)
    print(f"mode        : {args.mode}")
    print(f"meshes      : {args.meshes}")
    print(f"jitter      : {args.jitter}")
    print(f"max_dense_n : {args.max_dense_n}")
    print("=" * 72)

    for m in args.meshes:
        print(f"\nRodando malha {m}x{m} ...")
        t0 = time.perf_counter()
        rows, comp = run_case(
            nx=m,
            ny=m,
            mode=args.mode,
            jitter=args.jitter,
            nr=args.nr,
            max_dense_n=args.max_dense_n,
            out_prefix=args.prefix,
        )
        elapsed = time.perf_counter() - t0
        summary_rows.extend(rows)
        comparison_rows.extend(comp)
        print(f"Concluído em {elapsed:.3f} s")
        for r in comp:
            print(
                f"  {r['method']:<12s} "
                f"n={r['n']:5d} nnzG={r['nnz_G']:8d} rhoG={r['rho_G']:.3e} "
                f"EF={r['EF_identity']:.3e} Eact={r['Eact']:.3e} "
                f"lmin={r['lambda_min_symG']:.3e} t={r['t_build_s']:.3f}s"
            )

    df_summary = pd.DataFrame(summary_rows)
    df_comp = pd.DataFrame(comparison_rows)

    summary_path = outdir / f"{args.prefix}_summary.csv"
    comp_path = outdir / f"{args.prefix}_comparison.csv"
    df_summary.to_csv(summary_path, index=False)
    df_comp.to_csv(comp_path, index=False)

    print("\nArquivos gerados:")
    print(f"  {summary_path}")
    print(f"  {comp_path}")


if __name__ == "__main__":
    main()
