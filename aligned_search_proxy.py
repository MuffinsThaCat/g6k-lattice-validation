"""
Can a smarter choice of the rotation-invariant prefix lower the aligned-prefix penalty c?

A rank-d invariant sublattice of a module lattice is R*u for a generator u. Its covolume is
  vol(R u) = prod_j ||sigma_j(u)||   (j over the d canonical embeddings; sigma_j(u) in C^K),
computed here with a twisted FFT, and it is unchanged by multiplying u by a unit of R. The best aligned
prefix therefore minimises this product, which can be smaller for a generator with UNBALANCED embeddings
than for the shortest vector (flat embeddings give the largest volume for a given Euclidean norm).

Compared per vector (bits of log2 Gram-Schmidt norm): the best aligned prefix found in pools of increasing size
  top-T   : the T shortest database vectors,
  all     : the whole sieve database,
  +combos : the database plus S random sums/differences of two database vectors (non-short candidates),
against the greedy unconstrained prefix of d vectors, giving c = excess / (slope * d) as in
rotation_prefix_proxy.py. Also checks the FFT formula against the direct Gram-Schmidt value.

  python aligned_search_proxy.py      (cwd = G6K checkout, or G6K_DIR)
Env: AS_D=16 AS_K=3 AS_Q=257 AS_SEEDS=3 AS_COMBOS=300000 AS_OUT=...json
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
import rotation_prefix_proxy as PX

D = int(os.environ.get("AS_D", "16"))
K = int(os.environ.get("AS_K", "3"))
Q = int(os.environ.get("AS_Q", "257"))
SEEDS = int(os.environ.get("AS_SEEDS", "3"))
COMBOS = int(os.environ.get("AS_COMBOS", "300000"))
OUT = os.environ.get("AS_OUT", "aligned_search_proxy_results.json")


def log2_vol_orbit(U, d, k):
    """log2 vol(R u) for rows u of U (each = k blocks of d coefficients), negacyclic ring. Returns per-row value."""
    t = np.arange(d)
    tw = np.exp(1j * np.pi * t / d)
    B = U.reshape(len(U), k, d).astype(np.complex128) * tw
    F = np.fft.fft(B, axis=2)                        # (rows, k, d): embeddings sigma_j(u_k)
    lam = (np.abs(F) ** 2).sum(axis=1)               # (rows, d): ||sigma_j(u)||^2
    return 0.5 * np.log2(np.maximum(lam, 1e-300)).sum(axis=1)


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
    V = X @ Bn
    V = V[np.argsort(np.linalg.norm(V, axis=1))]
    Pf = P.astype(np.float64)
    # unconstrained greedy prefix (as in rotation_prefix_proxy)
    chosen = [V[0]]
    Qm = (V[0] / np.linalg.norm(V[0]))[None, :]
    for _ in range(1, d):
        res = V - (V @ Qm.T) @ Qm
        rn = np.linalg.norm(res, axis=1)
        rn[rn < 1e-6] = np.inf
        i = int(np.argmin(rn))
        chosen.append(V[i])
        Qm = np.vstack([Qm, res[i] / rn[i]])
    gU = PX.gs_norms(np.array(chosen))
    mean_U = float(np.mean(np.log2(gU)))
    slope = float((np.log2(gU[0]) - np.log2(gU[-1])) / max(1, d - 1))
    # check of the FFT volume formula against direct Gram-Schmidt for a few vectors
    for v in V[:3]:
        rows = [v]
        for _ in range(1, d):
            rows.append(rows[-1] @ Pf)
        direct = float(np.sum(np.log2(PX.gs_norms(np.array(rows)))))
        fft = float(log2_vol_orbit(v[None, :], d, K)[0])
        assert abs(direct - fft) < 1e-6 * max(1, abs(direct)), (direct, fft)
    rng = np.random.RandomState(seed)
    pools = {}
    pools["top-400"] = V[: min(400, len(V))]
    pools["all"] = V
    ii = rng.randint(0, len(V), COMBOS)
    jj = rng.randint(0, len(V), COMBOS)
    sg = rng.choice([-1.0, 1.0], COMBOS)
    C = V[ii] + sg[:, None] * V[jj]
    C = C[np.linalg.norm(C, axis=1) > 1e-6]
    pools["all+combos"] = np.vstack([V, C])
    out = {"seed": seed, "db": len(V), "slope_U": slope, "mean_log2_U": mean_U}
    for name, pool in pools.items():
        lv = log2_vol_orbit(pool, d, K)
        i = int(np.argmin(lv))
        excess = float(lv[i] / d - mean_U)
        out[name] = {"pool": len(pool), "best_log2vol_per_vec": float(lv[i] / d), "excess": excess,
                     "euclid_norm": float(np.linalg.norm(pool[i])), "c": excess / max(slope * d, 1e-9)}
    return out


def main():
    rows = []
    print(f"rank-{K} module, d={D} (n={K*D}), q={Q}: best aligned prefix by pool; c = excess / (slope * d)")
    for s in range(SEEDS):
        r = run_seed(100 + s)
        rows.append(r)
        print(f"seed {r['seed']}: slope {r['slope_U']:.4f} bits/index, unconstrained mean log2 GS {r['mean_log2_U']:.3f}")
        for name in ("top-400", "all", "all+combos"):
            x = r[name]
            print(f"   {name:11s} pool {x['pool']:7d}: best aligned {x['best_log2vol_per_vec']:.3f}  "
                  f"excess {x['excess']:+.3f}  c {x['c']:+.3f}  (|u| {x['euclid_norm']:.1f})", flush=True)
    with open(OUT, "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
