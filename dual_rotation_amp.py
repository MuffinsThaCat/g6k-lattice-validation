"""
Does rotation amplify a DUAL-lattice distinguisher built from real sieve output?  (toy module-LWE)

Instance: A in R_q^{m x k} (R = Z[x]/(x^d+1)), b = A s + e. Unfolded: T (m*d x k*d integer), b = T s + e mod q.
Dual lattice  L = { w in Z^{m d} : w^T T = 0 mod q }  has dimension m*d and is invariant under the
block-diagonal negacyclic shift w -> w P, so one short w yields d vectors x^i w, and <x^i w, b> are the
d coefficients of the single ring element conj(w)*b = conj(w)*e (the secret cancels).

Distinguisher: S = sum_w cos(2 pi <w,b>/q): mean > 0 for LWE b, mean 0 for uniform b.
We sieve L with G6K (bgj1), take the database W, and compare
   S_db : the database vectors themselves,
   S_cl : the rotation closure of the database (every x^i w, deduplicated),
each with its own best length cut-off, using T_TRIALS fresh (s, e) and the same number of uniform b.
Reported: distinguishing power d' = (mean_LWE - mean_null) / std_null, the factor (d'_cl/d'_db)^2 (the
effective sample gain), and the null variance relative to the independent-samples value |set|/2
(the independence heuristic check; a ratio > 1 means the samples are positively correlated).

  python dual_rotation_amp.py       (cwd = G6K checkout, or G6K_DIR)
Env: DR_D=16 DR_M=3 DR_K=2 DR_Q=257 DR_SIGMA="1.0,1.5,2.0" DR_TRIALS=300 DR_SEEDS=2 DR_OUT=...json
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

D_ = int(os.environ.get("DR_D", "16"))
M_ = int(os.environ.get("DR_M", "3"))
K_ = int(os.environ.get("DR_K", "2"))
Q = int(os.environ.get("DR_Q", "257"))
SIGMAS = [float(x) for x in os.environ.get("DR_SIGMA", "1.0,1.5,2.0").split(",")]
TRIALS = int(os.environ.get("DR_TRIALS", "300"))
SEEDS = int(os.environ.get("DR_SEEDS", "2"))
OUT = os.environ.get("DR_OUT", "dual_rotation_amp_results.json")


def modinv_matrix(A, q):
    """Inverse of a square integer matrix mod prime q (Gauss-Jordan); None if singular."""
    n = A.shape[0]
    M = np.concatenate([A % q, np.eye(n, dtype=np.int64)], axis=1)
    for c in range(n):
        piv = next((r for r in range(c, n) if M[r, c] % q), None)
        if piv is None:
            return None
        M[[c, piv]] = M[[piv, c]]
        M[c] = (M[c] * pow(int(M[c, c]), -1, q)) % q
        for r in range(n):
            if r != c and M[r, c]:
                M[r] = (M[r] - M[r, c] * M[c]) % q
    return M[:, n:]


def build_instance(d, m, k, q, seed):
    rng = np.random.RandomState(seed)
    N = R.neg_shift(d)
    Npow = [np.eye(d, dtype=np.int64)]
    for _ in range(d - 1):
        Npow.append(Npow[-1] @ N)
    while True:
        a = rng.randint(0, q, size=(m, k, d)).astype(np.int64)
        T = np.zeros((m * d, k * d), dtype=np.int64)
        for j in range(m):
            for i in range(k):
                for c in range(d):
                    T[j * d:(j + 1) * d, i * d + c] = (a[j, i] @ Npow[c]) % q
        Ttop, Tbot = T[:k * d], T[k * d:]
        inv = modinv_matrix(Ttop, q)
        if inv is not None:
            break
        seed += 1000
        rng = np.random.RandomState(seed)
    # dual basis rows: [ -e_j Tbot Ttop^-1 mod q | e_j ]  and  [ q e_i | 0 ]
    nb = m * d - k * d
    B = np.zeros((m * d, m * d), dtype=np.int64)
    X = (-(Tbot @ inv)) % q                      # (nb x kd): w_top = w_bot @ X ... transposed use below
    # w^T T = w_top Ttop + w_bot Tbot = 0  =>  w_top = -w_bot Tbot Ttop^-1
    for jj in range(nb):
        B[jj, :k * d] = (-(Tbot[jj] @ inv)) % q
        B[jj, k * d + jj] = 1
    for ii in range(k * d):
        B[nb + ii, ii] = q
    # sanity: every basis row is dual
    assert not ((B @ T) % q).any(), "dual basis construction failed"
    P = np.zeros((m * d, m * d), dtype=np.int64)
    for j in range(m):
        P[j * d:(j + 1) * d, j * d:(j + 1) * d] = N
    return T, B, P


def sieve_dual(B0, seed):
    n = B0.shape[0]
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
    Bn = np.array([[A[i, j] for j in range(n)] for i in range(n)], dtype=np.int64)
    X = np.array(list({R.canon(v) for v in g.itervalues()}), dtype=np.int64)
    return X @ Bn, Bn


def closure_vectors(W, P, d):
    seen = set()
    out = []
    cur = W.copy()
    for _ in range(d):
        for v in cur:
            c = R.canon(tuple(int(x) for x in v))
            if c not in seen:
                seen.add(c)
                out.append(c)
        cur = cur @ P
    return np.array(out, dtype=np.int64)


def best_dprime(W, Bs_lwe, Bs_null, q, ncuts=8):
    """Max over length cut-offs of d' = (mean_LWE - mean_null) / std_null for S = sum cos(2 pi <w,b>/q)."""
    norms = np.linalg.norm(W, axis=1)
    order = np.argsort(norms)
    W = W[order]
    res = []
    for frac in np.linspace(0.1, 1.0, ncuts):
        cnt = max(8, int(frac * len(W)))
        Wc = W[:cnt]
        s1 = np.cos(2 * np.pi * ((Wc @ Bs_lwe) % q) / q).sum(axis=0)
        s0 = np.cos(2 * np.pi * ((Wc @ Bs_null) % q) / q).sum(axis=0)
        dp = (s1.mean() - s0.mean()) / s0.std()
        res.append((dp, cnt, float(s0.var()), cnt / 2.0))
    return max(res, key=lambda r: r[0])


def main():
    d, m, k = D_, M_, K_
    n = m * d
    rows = []
    print(f"d={d} m={m} k={k} q={Q}: dual dimension {n}; trials {TRIALS}")
    for seed in range(SEEDS):
        T, B0, P = build_instance(d, m, k, Q, 10 + seed)
        W, Bn = sieve_dual(B0, 100 + seed)
        Mrot = R.rotation_matrix(Bn, P)           # raises if the dual lattice is not rotation-invariant
        C = closure_vectors(W, P, d)
        print(f"\ninstance {seed}: sieve database {len(W)} vectors, rotation closure {len(C)} "
              f"(x{len(C)/len(W):.1f}); shortest |w| {np.linalg.norm(W, axis=1).min():.1f}", flush=True)
        rng = np.random.RandomState(1000 + seed)
        for sigma in SIGMAS:
            S = rng.randint(-1, 2, size=(k * d, TRIALS)).astype(np.int64)
            E = np.rint(rng.normal(0, sigma, size=(n, TRIALS))).astype(np.int64)
            B_lwe = (T @ S + E) % Q
            B_null = rng.randint(0, Q, size=(n, TRIALS)).astype(np.int64)
            dp_db, cnt_db, var_db, nom_db = best_dprime(W, B_lwe, B_null, Q)
            dp_cl, cnt_cl, var_cl, nom_cl = best_dprime(C, B_lwe, B_null, Q)
            gain = (dp_cl / dp_db) ** 2 if dp_db > 0 else float("nan")
            # independent-samples prediction of d' for the closure at its chosen cut-off
            Cn = C[np.argsort(np.linalg.norm(C, axis=1))][:cnt_cl]
            bias = np.exp(-2 * np.pi ** 2 * (sigma ** 2 + 1.0 / 12) * np.sum(Cn.astype(float) ** 2, axis=1) / Q ** 2)
            dp_ind = bias.sum() / math.sqrt(cnt_cl / 2.0)
            Dn = W[np.argsort(np.linalg.norm(W, axis=1))][:cnt_db]
            bias_db = np.exp(-2 * np.pi ** 2 * (sigma ** 2 + 1.0 / 12) * np.sum(Dn.astype(float) ** 2, axis=1) / Q ** 2)
            dp_ind_db = bias_db.sum() / math.sqrt(cnt_db / 2.0)
            gain_pred = (dp_ind / dp_ind_db) ** 2
            print(f"  sigma={sigma:<4} d'(db)={dp_db:5.2f} [{cnt_db} vecs]   d'(closure)={dp_cl:5.2f} [{cnt_cl} vecs]   "
                  f"sample gain (d'_cl/d'_db)^2 = {gain:5.2f}   | null var / independent: db {var_db/nom_db:.2f}, "
                  f"closure {var_cl/nom_cl:.2f} | independent model: d'(db) {dp_ind_db:.2f}, d'(cl) {dp_ind:.2f}, gain {gain_pred:.2f}", flush=True)
            rows.append({"seed": seed, "sigma": sigma, "dp_db": dp_db, "dp_cl": dp_cl, "gain": gain,
                         "n_db": len(W), "n_cl": len(C), "null_var_ratio_db": var_db / nom_db,
                         "null_var_ratio_cl": var_cl / nom_cl, "dp_indep_cl": dp_ind, "dp_indep_db": dp_ind_db, "gain_pred": gain_pred})
    with open(OUT, "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
