"""
Does ring-rotation augmentation survive PROJECTION (what BKZ actually sieves)?

Lattice: rank-K module over R = Z[x]/(x^d+1), q-ary NTRU/MLWE-type, dimension K*d, basis rows
  x^i * g1 (i < d),  q*e_j (the remaining rows),  with g1 = (e_0, h_1, ..., h_{K-1}).
The first d rows span R*g1, a rotation-invariant sublattice S. G6K sieves the block [l, r) = [d, K*d),
i.e. the lattice projected orthogonally to S. Because S is invariant, the rotation x commutes with
the projection, so it acts on the projected block; its local matrix is the corner block
M[l:r, l:r] of M = B P B^-1. The projected block is again a ring lattice (rank K-1 module).

Needs the patched G6K (Siever.set_rotation) and a basis whose prefix is NOT scrambled: the
constructor's full LLL is skipped here and only a local LLL on [l, r) is run (it never touches S).

First the premise is CHECKED numerically (prefix invariant, projected norms preserved), then base vs
rotate-short vs rotate-everything are compared on the projected block, as in rotation_sieve_ab.py.

  python rotation_block_test.py            (cwd = patched g6k checkout, or G6K_DIR)
Env: BT_DS="20,24" BT_K=3 BT_Q=257 BT_SEEDS=3 BT_OUT=...json
"""
import json
import math
import os
import sys
import time

sys.path.insert(0, os.environ.get("G6K_DIR", os.getcwd()))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from fpylll import IntegerMatrix
from g6k import Siever
from g6k.siever_params import SieverParams
from g6k.siever import SaturationError

import rotation_orbit_check as R

DS = [int(x) for x in os.environ.get("BT_DS", "20,24").split(",")]
K = int(os.environ.get("BT_K", "3"))
Q = int(os.environ.get("BT_Q", "257"))
SEEDS = int(os.environ.get("BT_SEEDS", "3"))
OUT = os.environ.get("BT_OUT", "rotation_block_results.json")


class AlignedSiever(Siever):
    """Siever whose constructor does NOT run the full-lattice LLL (it would scramble the module-aligned
    prefix). The local LLL is run explicitly with Siever.lll(g, l, r)."""
    def lll(self, l, r):
        return None


def module_basis(d, q, seed, k):
    rng = np.random.RandomState(seed)
    N = R.neg_shift(d)
    n = k * d
    B = np.zeros((n, n), dtype=np.int64)
    B[:d, :d] = np.eye(d, dtype=np.int64)
    for t in range(1, k):
        h = rng.randint(0, q, size=d).astype(np.int64)
        row = h.copy()
        for i in range(d):
            B[i, t * d:(t + 1) * d] = row      # EXACT signed rotation x^i*h (not reduced mod q): the prefix must be exactly invariant
            row = row @ N
        B[t * d:(t + 1) * d, t * d:(t + 1) * d] = q * np.eye(d, dtype=np.int64)
    P = np.zeros((n, n), dtype=np.int64)
    for t in range(k):
        P[t * d:(t + 1) * d, t * d:(t + 1) * d] = N
    return B, P


def run_one(B0, P, d, k, mode, seed, sat, below=None):
    n = k * d
    l, r = d, n
    m = r - l
    A = IntegerMatrix.from_matrix(B0.tolist())
    g = AlignedSiever(A, SieverParams(threads=1, default_sieve="bgj1", reserved_n=m,
                                      db_size_factor=3.2, saturation_ratio=sat), seed=seed)
    g.M.update_gso()                         # the skipped constructor LLL would have computed the GSO
    Siever.lll(g, l, r)                      # local LLL only; rows < l (the invariant prefix) untouched
    g.initialize_local(l, l, r)
    Bn = np.array([[A[i, j] for j in range(n)] for i in range(n)], dtype=np.int64)
    Mfull = R.rotation_matrix(Bn, P)         # raises if the lattice is not invariant in this basis
    assert not Mfull[:l, l:].any(), "prefix is not rotation-invariant: block structure lost"
    M22 = Mfull[l:r, l:r]
    # projected Gram matrix of the block, from G6K's own GSO
    Mu = np.eye(m)
    rr = np.array([g.M.get_r(l + i, l + i) for i in range(m)])
    for i in range(m):
        for j in range(i):
            Mu[i, j] = g.M.get_mu(l + i, l + j)
    G = Mu @ np.diag(rr) @ Mu.T
    # premise check: rotation preserves projected lengths
    rng = np.random.RandomState(seed)
    X = rng.randint(-3, 4, size=(200, m)).astype(np.float64)
    a = np.einsum("ij,jk,ik->i", X, G, X)
    Y = X @ M22
    b = np.einsum("ij,jk,ik->i", Y, G, Y)
    prem = float(np.max(np.abs(a - b) / np.maximum(a, 1e-9)))
    assert prem < 1e-6, f"rotation does not preserve projected norms (rel err {prem:.2e})"
    order = 2 * d
    if mode != "base":
        g.set_rotation(M22, order, below if below is not None else g.params.saturation_radius)
    saturated = True
    t0 = time.perf_counter()
    try:
        g(alg="bgj1")
    except SaturationError:
        saturated = False
    wall = time.perf_counter() - t0
    xpc = g.get_stat("xorpopcnt_total") or 0
    fsp = g.get_stat("fullscprods_total") or 0
    gh2 = math.exp(2.0 * math.lgamma(m / 2.0 + 1.0) / m) / math.pi * math.exp(float(np.sum(np.log(rr))) / m)
    R2 = (4.0 / 3.0) * gh2
    db = list({R.canon(v) for v in g.itervalues()})
    Xdb = np.array(db, dtype=np.float64)
    L2 = np.einsum("ij,jk,ik->i", Xdb, G, Xdb) if len(db) else np.array([])
    short = {c for c, v in zip(db, L2) if v <= R2}
    closure = R.closure(short, M22, order)
    orbits = len({min(R.closure([c], M22, order)) for c in short})
    expected = (4.0 / 3.0) ** (m / 2.0) / 2.0
    return {"mode": mode, "wall": wall, "ops": xpc + fsp, "short": len(short), "closure": len(closure),
            "orbits": orbits, "expected": expected, "clones": g.rotation_clones_added,
            "ok": len(short) >= 0.95 * sat * expected, "premise_err": prem, "saturation_error": not saturated}


def main():
    out = []
    print(f"module rank K={K}, q={Q}; sieve block = projected lattice beyond the first ring block")
    for d in DS:
        m = (K - 1) * d
        B0, P = module_basis(d, Q, 1, K)
        print(f"\nd={d}: block dimension {m}; ~{(4/3)**(m/2)/2:.0f} short +- pairs expected", flush=True)
        variants = [("base", 0.5, None), ("rot-short", 1.0, None), ("rot-all", 1.0, 1e9)]
        base_w = base_o = base_orb = None
        for name, sat, below in variants:
            rs = [run_one(B0, P, d, K, "base" if name == "base" else "rot", 100 + s, sat, below)
                  for s in range(SEEDS)]
            wall = float(np.mean([r["wall"] for r in rs]))
            ops = float(np.mean([r["ops"] for r in rs]))
            orb = float(np.mean([r["orbits"] for r in rs]))
            if name == "base":
                base_w, base_o, base_orb = wall, ops, orb
            print(f"  {name:9s} orbits {orb:6.1f}/{base_orb:.1f}  wall {wall:8.3f}s ({math.log2(base_w/wall):+.2f} bits)  "
                  f"ops {ops:.3g} ({math.log2(base_o/ops):+.2f} bits)  premise err "
                  f"{max(r['premise_err'] for r in rs):.1e}  clones {np.mean([r['clones'] for r in rs]):.0f}", flush=True)
            out.append({"d": d, "m": m, "variant": name, "wall": wall, "ops": ops, "orbits": orb,
                        "base_orbits": base_orb, "runs": rs})
        with open(OUT, "w") as fh:
            json.dump(out, fh, indent=1, default=str)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
