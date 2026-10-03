"""
Proxy for the QUALITY COST of a rotation-aligned prefix (the missing half of the "net bits" question).

A rotation-aligned basis must start with d vectors x^i * v (rotations of one vector v). A normal reduced
basis starts with d independently chosen short vectors. Both are built here from the SAME real G6K sieve
database of a rank-K module lattice:

  U  = unconstrained greedy prefix: w1 = shortest db vector, then repeatedly the db vector with the
       shortest component orthogonal to what was already chosen (d steps).
  A  = aligned prefix: the best v in the db minimising the log-volume of [v, xv, ..., x^(d-1) v].

Compared: the Gram-Schmidt norms g_i of each prefix and its mean log2 g_i. log2-volume of the prefix is
what the rest of the basis inherits (the tail covolume is vol(L)/vol(prefix)), so the gap is the
quality the aligned prefix gives up in its first block. This is only a PROXY: it uses the sieve database
(short vectors only), not a full module-BKZ, and it does not model how the loss compounds over a tour.

  python rotation_prefix_proxy.py       (cwd = patched g6k checkout, or G6K_DIR)
Env: PX_D=20 PX_K=3 PX_Q=257 PX_SEEDS=3 PX_OUT=...json
"""
import json
import math
import os
import sys

sys.path.insert(0, os.environ.get("G6K_DIR", os.getcwd()))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from fpylll import IntegerMatrix, LLL, BKZ
from g6k import Siever
from g6k.siever_params import SieverParams
from g6k.siever import SaturationError

import rotation_orbit_check as R
import rotation_block_test as T

D = int(os.environ.get("PX_D", "20"))
K = int(os.environ.get("PX_K", "3"))
Q = int(os.environ.get("PX_Q", "257"))
SEEDS = int(os.environ.get("PX_SEEDS", "3"))
OUT = os.environ.get("PX_OUT", "rotation_prefix_proxy_results.json")


def gs_norms(rows):
    """Gram-Schmidt norms of the rows (in order)."""
    Qm = np.zeros((0, rows.shape[1]))
    out = []
    for v in rows:
        r = v - Qm.T @ (Qm @ v) if len(Qm) else v.copy()
        r = r - Qm.T @ (Qm @ r) if len(Qm) else r        # one re-orthogonalisation for stability
        nr = float(np.linalg.norm(r))
        out.append(nr)
        if nr > 1e-9:
            Qm = np.vstack([Qm, r / nr])
    return np.array(out)


def run_seed(seed):
    d, n = D, K * D
    B0, P = T.module_basis(d, Q, seed, K)
    A = IntegerMatrix.from_matrix(B0.tolist())
    A = LLL.reduction(A)
    BKZ.reduction(A, BKZ.Param(block_size=min(20, n)))
    g = Siever(A, SieverParams(threads=1, default_sieve="bgj1", reserved_n=n,
                               db_size_factor=3.2, saturation_ratio=0.5), seed=seed)
    g.initialize_local(0, 0, n)
    try:
        g(alg="bgj1")
    except SaturationError:
        pass
    Bn = np.array([[A[i, j] for j in range(n)] for i in range(n)], dtype=np.float64)
    X = np.array(list({R.canon(v) for v in g.itervalues()}), dtype=np.float64)
    V = X @ Bn                                     # Euclidean short vectors of the lattice
    Pf = P.astype(np.float64)
    norms = np.linalg.norm(V, axis=1)
    order = np.argsort(norms)
    V = V[order]
    # U: unconstrained greedy prefix of d vectors
    chosen = [V[0]]
    Qm = (V[0] / np.linalg.norm(V[0]))[None, :]
    for _ in range(1, d):
        res = V - (V @ Qm.T) @ Qm
        rn = np.linalg.norm(res, axis=1)
        rn[rn < 1e-6] = np.inf
        i = int(np.argmin(rn))
        chosen.append(V[i])
        Qm = np.vstack([Qm, res[i] / rn[i]])
    gU = gs_norms(np.array(chosen))
    # A: best aligned prefix among the db vectors (try the shortest few hundred)
    best = None
    for v in V[: min(len(V), 400)]:
        rows = [v]
        for _ in range(1, d):
            rows.append(rows[-1] @ Pf)
        gA = gs_norms(np.array(rows))
        if gA.min() < 1e-2 * gA[0]:
            continue    # rank-deficient orbit span (zero divisors: x^d+1 reducible for non-power-of-two d); not a valid prefix
        score = float(np.sum(np.log2(np.maximum(gA, 1e-9))))
        if best is None or score < best[0]:
            best = (score, gA)
    gA = best[1]
    lam1 = float(norms[order][0])
    slope_u = float((np.log2(gU[0]) - np.log2(gU[-1])) / max(1, d - 1))   # bits of log2 GS norm lost per index
    return {"seed": seed, "db": int(len(V)), "lambda1": lam1, "slope_U": slope_u,
            "mean_log2_U": float(np.mean(np.log2(gU))), "mean_log2_A": float(np.mean(np.log2(gA))),
            "gU": gU.tolist(), "gA": gA.tolist()}


def main():
    rows = []
    print(f"rank-{K} module lattice, d={D} (n={K*D}); prefix length d={D}; q={Q}")
    for s in range(SEEDS):
        r = run_seed(100 + s)
        rows.append(r)
        gap = r["mean_log2_A"] - r["mean_log2_U"]
        print(f"seed {r['seed']}: db {r['db']}, lambda1 {r['lambda1']:.1f} | mean log2 GS norm over the prefix: "
              f"unconstrained {r['mean_log2_U']:.3f}, aligned {r['mean_log2_A']:.3f}  "
              f"-> aligned is {gap:+.3f} bits/vector ({gap * D:+.1f} bits total over {D} vectors) larger", flush=True)
        sl = r["slope_U"]
        print(f"   slope of the unconstrained prefix: {sl:.4f} bits/index; aligned excess / (slope * d) = "
              f"{(r['mean_log2_A'] - r['mean_log2_U']) / max(sl * D, 1e-9):.3f}")
        print("   GS norms U:", " ".join(f"{x:.0f}" for x in r["gU"][:10]), "...")
        print("   GS norms A:", " ".join(f"{x:.0f}" for x in r["gA"][:10]), "...")
    with open(OUT, "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
