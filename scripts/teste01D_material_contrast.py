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
    Algebraic validation of the selected inverse Hodge approximation under material contrast.

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
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla


                                                              
                                 
                                                              

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


                                                              
                     
                                                              

def triangle_material_eps(
    points: np.ndarray,
    tri: np.ndarray,
    eps_left: float,
    eps_right: float,
    interface_x: float,
) -> float:
    """Helper routine used in the numerical experiments."""
    centroid_x = float(points[tri, 0].mean())
    return float(eps_left if centroid_x < interface_x else eps_right)


                                                              
                  
                                                              

def triangle_area_and_grads(coords: np.ndarray) -> Tuple[float, np.ndarray]:
    """Helper routine used in the numerical experiments."""
    x0, y0 = coords[0]
    x1, y1 = coords[1]
    x2, y2 = coords[2]
    det = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    area = 0.5 * det
    if area <= 0:
        raise ValueError(
            "Triangle is not positively oriented or has zero area. "
            "Reduce the jitter or check the connectivity."
        )

    grads = np.empty((3, 2), dtype=float)
    grads[0] = np.array([y1 - y2, x2 - x1]) / (2.0 * area)
    grads[1] = np.array([y2 - y0, x0 - x2]) / (2.0 * area)
    grads[2] = np.array([y0 - y1, x1 - x0]) / (2.0 * area)
    return float(area), grads


def int_lambda_lambda(area: float, a: int, b: int) -> float:
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


def prune_sparse(A: sp.spmatrix, tol: float = 1e-14) -> sp.csr_matrix:
    """Helper routine used in the numerical experiments."""
    B = A.tocsr(copy=True)
    if tol > 0:
        mask = np.abs(B.data) <= tol
        B.data[mask] = 0.0
    B.eliminate_zeros()
    return B


def assemble_hodge_h1_material(
    mesh: TriMesh2D,
    eps_left: float = 1.0,
    eps_right: float = 10.0,
    interface_x: float = 0.5,
) -> sp.csr_matrix:
    """Helper routine used in the numerical experiments."""
    ne = mesh.edges.shape[0]
    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []

    for t, tri in enumerate(mesh.triangles):
        eps_t = triangle_material_eps(
            points=mesh.points,
            tri=tri,
            eps_left=eps_left,
            eps_right=eps_right,
            interface_x=interface_x,
        )
        coords = mesh.points[tri]
        Mloc = local_whitney_h1_mass(coords, eps=eps_t)
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
    return prune_sparse(H, tol=0.0)


                                                              
                               
                                                              

def structural_pattern(A: sp.spmatrix, tol: float = 1e-14, include_diagonal: bool = True) -> sp.csr_matrix:
    """Helper routine used in the numerical experiments."""
    B = prune_sparse(A, tol=tol)
    B.data[:] = 1.0
    if include_diagonal:
        B = (B + sp.eye(B.shape[0], format="csr"))
    B.data[:] = 1.0
    B.eliminate_zeros()
    return B.astype(np.float64).tocsr()


def boolean_product_pattern(A: sp.csr_matrix, B: sp.csr_matrix) -> sp.csr_matrix:
    """Helper routine used in the numerical experiments."""
    C = (A @ B).tocsr()
    C.data[:] = 1.0
    C.eliminate_zeros()
    return C


def radius_patterns_from_H(H: sp.csr_matrix, radii: Sequence[int], tol: float) -> Dict[int, sp.csr_matrix]:
    """Helper routine used in the numerical experiments."""
    if not radii:
        raise ValueError("radii cannot be empty.")
    if min(radii) < 1:
        raise ValueError("radii must be >= 1.")

    max_r = max(radii)
    P = structural_pattern(H, tol=tol, include_diagonal=True)
    patterns: Dict[int, sp.csr_matrix] = {}
    current = P.copy()

    for r in range(1, max_r + 1):
        if r == 1:
            current = P.copy()
        else:
            current = boolean_product_pattern(current, P)
        if r in radii:
            S = (current + current.T).tocsr()
            S.data[:] = 1.0
            S.eliminate_zeros()
            patterns[r] = S
    return patterns


def project_dense_inverse_to_pattern(Hinv: np.ndarray, pattern: sp.csr_matrix) -> sp.csr_matrix:
    """Helper routine used in the numerical experiments."""
    coo = pattern.tocoo()
    vals = Hinv[coo.row, coo.col]
    G = sp.coo_matrix((vals, (coo.row, coo.col)), shape=pattern.shape).tocsr()
    G.sum_duplicates()
    return G


def diag_inverse(H: sp.csr_matrix) -> sp.csr_matrix:
    d = H.diagonal()
    if np.any(np.abs(d) < 1e-30):
        raise ZeroDivisionError("H has zero diagonal entries.")
    return sp.diags(1.0 / d, format="csr")


def density(A: sp.spmatrix | np.ndarray) -> float:
    n, m = A.shape
    nnz = A.nnz if sp.issparse(A) else int(np.count_nonzero(A))
    return float(nnz) / float(n * m)


def identity_error(H: sp.csr_matrix, G: sp.csr_matrix) -> float:
    n = H.shape[0]
    R = sp.eye(n, format="csr") - (H @ G)
    return float(spla.norm(R, ord="fro") / np.sqrt(n))


def diagonal_error(Hinv: np.ndarray, G: sp.csr_matrix) -> float:
    ref = np.diag(Hinv)
    gd = G.diagonal()
    return float(np.linalg.norm(ref - gd) / np.linalg.norm(ref))


def action_error(Hinv: np.ndarray, G: sp.csr_matrix, nr: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    n = Hinv.shape[0]
    errs = []
    for _ in range(nr):
        r = rng.standard_normal(n)
        r /= np.linalg.norm(r)
        y_ref = Hinv @ r
        y = G @ r
        errs.append(np.linalg.norm(y_ref - y) / np.linalg.norm(y_ref))
    return float(np.mean(errs))


def spectral_bounds_symmetric(G: sp.csr_matrix, dense_limit: int = 1000, eig_max_n: int = 1000) -> Tuple[float, float, float]:
    """Helper routine used in the numerical experiments."""
    n = G.shape[0]
    if eig_max_n <= 0 or n > eig_max_n:
        return np.nan, np.nan, np.nan
    Gs = ((G + G.T) * 0.5).tocsr()
    try:
        if n <= dense_limit:
            vals = np.linalg.eigvalsh(Gs.toarray())
            lmin = float(vals[0])
            lmax = float(vals[-1])
        else:
            lmin = float(spla.eigsh(Gs, k=1, which="SA", return_eigenvectors=False, tol=1e-8, maxiter=5000)[0])
            lmax = float(spla.eigsh(Gs, k=1, which="LA", return_eigenvectors=False, tol=1e-8, maxiter=5000)[0])
        cond = float(lmax / lmin) if lmin > 0 else np.inf
        return lmin, lmax, cond
    except Exception:
        return np.nan, np.nan, np.nan


def matvec_time(G: sp.csr_matrix, nrep: int = 50, seed: int = 123) -> float:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(G.shape[0])
    t0 = time.perf_counter()
    for _ in range(nrep):
        x = G @ x
    return float((time.perf_counter() - t0) / max(nrep, 1))


def h_condition_dense(H_dense: np.ndarray) -> float:
    """Helper routine used in the numerical experiments."""
    try:
        vals = np.linalg.eigvalsh(H_dense)
        lmin = float(vals[0])
        lmax = float(vals[-1])
        return float(lmax / lmin) if lmin > 0 else np.inf
    except Exception:
        return np.nan


                                                              
                              
                                                              

def run_one_case(
    nx: int,
    ny: int,
    radii: Sequence[int],
    jitter: float,
    contrast: float,
    eps_left: float,
    interface_x: float,
    seed: int,
    tol: float,
    nr: int,
    max_dense_n: int,
    eig_max_n: int,
) -> Tuple[List[dict], dict]:
    t_case0 = time.perf_counter()
    eps_right = float(contrast)

    mesh = build_structured_tri_mesh(nx, ny, jitter=jitter, seed=seed)
    H = assemble_hodge_h1_material(
        mesh=mesh,
        eps_left=eps_left,
        eps_right=eps_right,
        interface_x=interface_x,
    )
    H = prune_sparse(H, tol=tol)
    n = H.shape[0]

    if n > max_dense_n:
        raise MemoryError(
            f"n={n} exceeds --max-dense-n={max_dense_n}. "
            "Reduza a malha ou aumente --max-dense-n conscientemente."
        )

    t_inv0 = time.perf_counter()
    H_dense = H.toarray()
    H_cond = h_condition_dense(H_dense)
    Hinv = np.linalg.inv(H_dense)
    t_inv = time.perf_counter() - t_inv0

    patterns = radius_patterns_from_H(H, radii=radii, tol=tol)

    rows: List[dict] = []
    methods: List[Tuple[str, sp.csr_matrix, float]] = []

                        
    t0 = time.perf_counter()
    methods.append(("G_diag", diag_inverse(H), time.perf_counter() - t0))

                           
    for r in radii:
        t0 = time.perf_counter()
        G = project_dense_inverse_to_pattern(Hinv, patterns[r])
        t_proj = time.perf_counter() - t0
        methods.append((f"G_H_radius_{r}", G, t_proj))

    for method, G, t_build in methods:
        ef = identity_error(H, G)
        ed = diagonal_error(Hinv, G)
        ea = action_error(Hinv, G, nr=nr, seed=seed + 77)
        lmin, lmax, cond = spectral_bounds_symmetric(G, eig_max_n=eig_max_n)
        tmv = matvec_time(G, nrep=30, seed=seed + 99)
        radius = 0 if method == "G_diag" else int(method.split("_")[-1])

        rows.append({
            "mesh": f"{nx}x{ny}",
            "nx": nx,
            "ny": ny,
            "jitter": float(jitter),
            "contrast": float(contrast),
            "eps_left": float(eps_left),
            "eps_right": float(eps_right),
            "interface_x": float(interface_x),
            "NV": int(mesh.points.shape[0]),
            "NE": int(mesh.edges.shape[0]),
            "NF": int(mesh.triangles.shape[0]),
            "n": int(n),
            "nnz_H": int(H.nnz),
            "rho_H": density(H),
            "cond_H_dense": float(H_cond),
            "method": method,
            "radius": int(radius),
            "nnz_G": int(G.nnz),
            "rho_G": density(G),
            "nnzG_over_nnzH": float(G.nnz / H.nnz),
            "E_F_identity": float(ef),
            "E_diag": float(ed),
            "E_action": float(ea),
            "lambda_min_symG": float(lmin),
            "lambda_max_symG": float(lmax),
            "cond_symG": float(cond),
            "t_inverse_dense_s": float(t_inv),
            "t_build_projection_s": float(t_build),
            "t_matvec_s": float(tmv),
            "passed_positive": bool(np.isfinite(lmin) and lmin > 0.0),
            "passed_action_1e2": bool(ea < 1.0e-2),
            "passed_action_vs_diag": np.nan,
        })

    diag_eact = next(r["E_action"] for r in rows if r["method"] == "G_diag")
    for r in rows:
        r["passed_action_vs_diag"] = bool(r["E_action"] < diag_eact) if r["method"] != "G_diag" else False

    non_diag = [r for r in rows if r["method"] != "G_diag"]
    best = min(non_diag, key=lambda z: z["E_action"])
    radius3 = next((r for r in rows if r["method"] == "G_H_radius_3"), None)

    summary = {
        "mesh": f"{nx}x{ny}",
        "nx": int(nx),
        "ny": int(ny),
        "jitter": float(jitter),
        "contrast": float(contrast),
        "eps_left": float(eps_left),
        "eps_right": float(eps_right),
        "interface_x": float(interface_x),
        "NV": int(mesh.points.shape[0]),
        "NE": int(mesh.edges.shape[0]),
        "NF": int(mesh.triangles.shape[0]),
        "n": int(n),
        "nnz_H": int(H.nnz),
        "rho_H": density(H),
        "cond_H_dense": float(H_cond),
        "best_method": best["method"],
        "best_radius": int(best["radius"]),
        "best_E_action": float(best["E_action"]),
        "best_E_F_identity": float(best["E_F_identity"]),
        "best_lambda_min_symG": float(best["lambda_min_symG"]),
        "best_rho_G": float(best["rho_G"]),
        "best_nnzG_over_nnzH": float(best["nnzG_over_nnzH"]),
        "diag_E_action": float(diag_eact),
        "r3_E_action": float(radius3["E_action"]) if radius3 is not None else np.nan,
        "r3_E_F_identity": float(radius3["E_F_identity"]) if radius3 is not None else np.nan,
        "r3_lambda_min_symG": float(radius3["lambda_min_symG"]) if radius3 is not None else np.nan,
        "r3_nnzG_over_nnzH": float(radius3["nnzG_over_nnzH"]) if radius3 is not None else np.nan,
        "r3_passed_positive": bool(radius3["passed_positive"]) if radius3 is not None else False,
        "r3_passed_action_1e2": bool(radius3["passed_action_1e2"]) if radius3 is not None else False,
        "total_time_s": float(time.perf_counter() - t_case0),
        "t_inverse_dense_s": float(t_inv),
    }
    return rows, summary


                                                              
     
                                                              

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Teste 01D — Contraste material para inversão seletiva topológica da Hodge."
    )

    p.add_argument(
        "--meshes",
        nargs="+",
        type=int,
        default=[8, 12, 16, 20],
        help="Lista de malhas quadradas nx=ny. Default: 8 12 16 20",
    )
    p.add_argument(
        "--radii",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="Raios topológicos a testar. Default: 1 2 3",
    )
    p.add_argument(
        "--contrasts",
        nargs="+",
        type=float,
        default=[10.0, 100.0, 1000.0],
        help="Valores de eps_right. Default: 10 100 1000",
    )
    p.add_argument(
        "--jitter-values",
        nargs="+",
        type=float,
        default=[0.0, 0.1],
        help="Jitter values geométrico. Default: 0.0 0.1",
    )
    p.add_argument(
        "--eps-left",
        type=float,
        default=1.0,
        help="Permissividade à esquerda da interface. Default: 1.0",
    )
    p.add_argument(
        "--interface-x",
        type=float,
        default=0.5,
        help="Posição da interface vertical. Default: 0.5",
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
        default=1e-14,
        help="Tolerância para padrão estrutural. Default: 1e-14",
    )
    p.add_argument(
        "--nr",
        type=int,
        default=20,
        help="Número de vetores aleatórios para erro de ação. Default: 20",
    )
    p.add_argument(
        "--max-dense-n",
        type=int,
        default=2500,
        help="Dimensão máxima para inversão densa. Default: 2500",
    )
    p.add_argument(
        "--eig-max-n",
        type=int,
        default=2500,
        help="Dimensão máxima para calcular autovalores de G. Default: 2500. Use 0 para pular.",
    )
    p.add_argument(
        "--out-prefix",
        type=str,
        default="teste01D_material_contrast",
        help="Output CSV prefix.",
    )
    return p.parse_args()


                                                              
      
                                                              

def main() -> None:
    args = parse_args()
    radii = sorted(set(args.radii))
    contrasts = list(args.contrasts)
    jitter_values = list(args.jitter_values)

    print("\nTeste 01D — Contraste material da inversa seletiva topológica de Hodge")
    print("=" * 96)
    print(f"meshes        : {args.meshes}")
    print(f"radii         : {radii}")
    print(f"contrasts     : {contrasts}")
    print(f"jitter_values : {jitter_values}")
    print(f"eps_left      : {args.eps_left}")
    print(f"interface_x   : {args.interface_x}")
    print(f"tol           : {args.tol:.1e}")
    print(f"nr            : {args.nr}")
    print(f"max_dense_n   : {args.max_dense_n}")
    print(f"eig_max_n     : {args.eig_max_n}")
    print(f"out_prefix    : {args.out_prefix}")
    print("=" * 96)

    all_rows: List[dict] = []
    summaries: List[dict] = []

    for contrast in contrasts:
        for jitter in jitter_values:
            print(f"\n### Contraste eps_right={contrast:g} | jitter={jitter:g} ###")
            for nx in args.meshes:
                print(f"\nRodando malha {nx}x{nx} ...")
                t0 = time.perf_counter()
                try:
                    rows, summary = run_one_case(
                        nx=nx,
                        ny=nx,
                        radii=radii,
                        jitter=float(jitter),
                        contrast=float(contrast),
                        eps_left=float(args.eps_left),
                        interface_x=float(args.interface_x),
                        seed=int(args.seed),
                        tol=float(args.tol),
                        nr=int(args.nr),
                        max_dense_n=int(args.max_dense_n),
                        eig_max_n=int(args.eig_max_n),
                    )
                except Exception as exc:
                    print(f"  ERRO: {exc}")
                    continue

                all_rows.extend(rows)
                summaries.append(summary)

                print(f"Concluído em {time.perf_counter() - t0:.3f} s")
                for r in rows:
                    print(
                        f"  {r['method']:<14s} "
                        f"n={r['n']:5d} "
                        f"nnzG={r['nnz_G']:8d} "
                        f"nnzG/nnzH={r['nnzG_over_nnzH']:.2f} "
                        f"EF={r['E_F_identity']:.3e} "
                        f"Eact={r['E_action']:.3e} "
                        f"lmin={r['lambda_min_symG']:.3e}"
                    )

                print(
                    f"  melhor raio: r={summary['best_radius']} | "
                    f"Eact={summary['best_E_action']:.3e} | "
                    f"lambda_min={summary['best_lambda_min_symG']:.3e} | "
                    f"nnzG/nnzH={summary['best_nnzG_over_nnzH']:.2f}"
                )

                if not summary["r3_passed_action_1e2"] or not summary["r3_passed_positive"]:
                    print(
                        "  ALERTA r=3: "
                        f"Eact={summary['r3_E_action']:.3e}, "
                        f"lambda_min={summary['r3_lambda_min_symG']:.3e}"
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

                            
    print("\nSíntese rápida para r=3:")
    cols = [
        "contrast", "jitter", "mesh", "r3_E_action", "r3_lambda_min_symG",
        "r3_nnzG_over_nnzH", "r3_passed_action_1e2", "r3_passed_positive"
    ]
    print(dfs[cols].to_string(index=False))


if __name__ == "__main__":
    main()
