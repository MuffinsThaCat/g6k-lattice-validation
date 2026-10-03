"""
Aligned-prefix penalty for a rotation SUBGROUP of order t (orbit length t) instead of the full ring group.

The lattice is invariant under y = x^(d/t) for every divisor t of d. An invariant prefix is then R_t*u: the
t rotations u, yu, ..., y^(t-1) u (rank t), a smaller staircase than the full d. Compared per vector (bits of
log2 Gram-Schmidt norm): the best such prefix over the whole sieve database against the greedy unconstrained
prefix of t vectors; c_t = excess / (slope * t) as in rotation_prefix_proxy.py.

  python subgroup_penalty_proxy.py      (cwd = G6K checkout, or G6K_DIR)
Env: SP_D=32 SP_K=2 SP_TS="4,8,16,32" SP_Q=257 SP_SEEDS=4 SP_OUT=...json
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

D = int(os.environ.get("SP_D", "32"))
K = int(os.environ.get("SP_K", "2"))
Q = int(os.environ.get("SP_Q", "257"))
TS = [int(x) for x in os.environ.get("SP_TS", "4,8,16,32").split(",")]
SEEDS = int(os.environ.get("SP_SEEDS", "4"))
OUT = os.environ.get("SP_OUT", "subgroup_penalty_proxy_results.json")


def log2_vol_sub(U, Pt, t):
    """log2 of the volume of span{u, yu, ..., y^(t-1) u} for each row u of U (Gram determinant)."""
    rots = [U]
    for _ in range(t - 1):
        rots.append(rots[-1] @ Pt)
    Rs = np.array(rots)                                  # (t, rows, n)
    G = np.einsum("rin,sin->irs", Rs, Rs)                # (rows, t, t)
    sign, logdet = np.linalg.slogdet(G)
    out = 0.5 * logdet / math.log(2)
    out[sign <= 0] = np.inf
    return out


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
    out = {"seed": seed, "db": len(V), "by_t": {}}
    for t in TS:
        Pt = np.linalg.matrix_power(P.astype(np.float64), d // t)
        chosen = [V[0]]
        Qm = (V[0] / np.linalg.norm(V[0]))[None, :]
        for _ in range(1, t):
            res = V - (V @ Qm.T) @ Qm
            rn = np.linalg.norm(res, axis=1)
            rn[rn < 1e-6] = np.inf
            i = int(np.argmin(rn))
            chosen.append(V[i])
            Qm = np.vstack([Qm, res[i] / rn[i]])
        gU = PX.gs_norms(np.array(chosen))
        mean_U = float(np.mean(np.log2(gU)))
        slope = float((np.log2(gU[0]) - np.log2(gU[-1])) / max(1, t - 1))
        lv = log2_vol_sub(V, Pt, t)
        # reject rank-deficient orbit spans
        ok = np.isfinite(lv)
        lv_pool = np.where(ok, lv, np.inf)
        i = int(np.argmin(lv_pool))
        best = float(lv_pool[i] / t)
        excess = best - mean_U
        out["by_t"][t] = {"slope": slope, "excess": excess, "c": excess / max(slope * t, 1e-9),
                          "best_per_vec": best, "mean_U": mean_U}
    return out


def main():
    print(f"ring d={D}, rank {K} (n={K*D}), q={Q}: subgroup orders {TS}")
    rows = []
    for s in range(SEEDS):
        r = run_seed(100 + s)
        rows.append(r)
        line = "  ".join(f"t={t}: c={r['by_t'][t]['c']:+.2f}" for t in TS)
        print(f"seed {r['seed']}: {line}", flush=True)
    print("mean c:  " + "  ".join(f"t={t}: {np.mean([r['by_t'][t]['c'] for r in rows]):+.3f}" for t in TS))
    with open(OUT, "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
