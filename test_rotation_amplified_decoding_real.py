"""
REAL test of rotation-amplified decoding (see rotation_amplified_decoding_estimate.py
for the idea and the model).

What is real here
-----------------
  * Real MLWE instances over Z_q[x]/(x^n+1), true negacyclic arithmetic.
  * Real fpylll LLL + progressive BKZ-beta of the TARGET-FREE q-ary lattice
    Lambda = {(y, x) : y = (A_coeff)_I x mod q}; reduced ONCE per (A, beta).
  * For each secret (s, e): the n displacement vectors ((x^j e)_I, -x^j s).
    Decoding t_j = v_j + disp_j succeeds iff decoding disp_j returns 0
    (Babai and CVP are exactly translation-invariant by lattice vectors), so we
    decode the displacements directly.
  * Three success criteria per view, all evaluated on the real reduced basis:
      norm : ||pi_blk(disp)||^2 <= ||b*_{d-beta}||^2       (the ADPS16 "2016 estimate" event)
      cvp  : exact CVP by real enumeration in the projected last-beta block returns the
             true point, AND Babai lifts it correctly through the first d-beta indices
      babai: plain Babai nearest-plane on the whole basis succeeds (Ogilvie's p_NP event)

Compared, at every beta:
  p1     = success probability of ONE view (the published analyses' p_NP / p_bdd)
  p_any  = probability that AT LEAST ONE of the n rotated views succeeds
  indep  = per-secret independence prediction, using the Beta(beta/2,(d-beta)/2) law
           conditional on the true ||disp||^2 (norm criterion only)
and the block size at which each crosses 50%.

Usage:
  python test_rotation_amplified_decoding_real.py pilot  <set>
  python test_rotation_amplified_decoding_real.py run    <set> <sigma> [n_seeds] [n_secrets]
"""
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy import stats

from fpylll import BKZ, GSO, IntegerMatrix, LLL, Enumeration, EnumerationError, FPLLL
from fpylll.algorithms.bkz2 import BKZReduction

Q = 3329
# name: (n, k, m, beta list)
SETS = {
    "n32k2":   (32, 2, 64, list(range(12, 43, 2))),     # d = 128, 32 rotations
    "n64k1":   (64, 1, 64, list(range(12, 43, 2))),     # d = 128, 64 rotations
    "n64k2p":  (64, 2, 96, list(range(12, 43, 2))),     # d = 224, m NOT a multiple of n (row subset)
    "n128k1":  (128, 1, 128, list(range(12, 43, 2))),   # d = 256, 128 rotations
}
# The two larger sets need a larger modulus: at q = 3329 their marginal error width is
# sigma ~ 0.5 (pilot), where a rounded Gaussian degenerates to a sparse ternary vector.
Q_OF = {"n32k2": 3329, "n64k1": 3329, "n64k2p": 262139, "n128k1": 131071}


def set_modulus(set_name):
    global Q
    Q = Q_OF[set_name]
STRATEGIES = next((p for p in (
    "/opt/homebrew/Cellar/fplll/5.5.0/share/fplll/strategies/default.json",
    "/opt/homebrew/share/fplll/strategies/default.json",
    BKZ.DEFAULT_STRATEGY.decode() if isinstance(BKZ.DEFAULT_STRATEGY, bytes) else BKZ.DEFAULT_STRATEGY,
) if os.path.exists(p)), BKZ.DEFAULT_STRATEGY)   # fallback: fpylll resolves its own default
CVP_SCALE = 1024
CVP_PREFILTER = 1.2   # skip enumeration if ||proj|| > 1.2 * GH(block): expected # closer points >= 1.2^beta


def negacyclic_matrix(a):
    """n x n integer matrix of multiplication by a in Z[x]/(x^n+1): column i = a * x^i."""
    n = len(a)
    M = np.zeros((n, n), dtype=np.int64)
    col = a.copy()
    for i in range(n):
        M[:, i] = col
        col = np.concatenate(([-col[-1]], col[:-1]))
    return M


def rot(a, j):
    """x^j * a in Z[x]/(x^n+1)."""
    if j == 0:
        return a.copy()
    n = len(a)
    return np.concatenate((-a[n - j:], a[:n - j]))


def build_basis(rng, n, k, m):
    A = np.zeros((k * n, k * n), dtype=np.int64)
    for r in range(k):
        for c in range(k):
            A[r * n:(r + 1) * n, c * n:(c + 1) * n] = negacyclic_matrix(rng.integers(0, Q, n))
    A = A[:m, :] % Q                       # row subset I = first m coefficient rows
    ns = k * n
    d = m + ns
    B = np.zeros((d, d), dtype=np.int64)
    B[:m, :m] = Q * np.eye(m, dtype=np.int64)
    B[m:, :m] = A.T
    B[m:, m:] = np.eye(ns, dtype=np.int64)
    return A, B


def displacements(rng, n, k, m, sigma, count):
    """count x n x d array: all n rotated displacement vectors of `count` secrets."""
    ns = k * n
    out = np.zeros((count, n, m + ns), dtype=np.int64)
    for t in range(count):
        e = np.rint(rng.normal(0, sigma, (k, n))).astype(np.int64)
        s = np.rint(rng.normal(0, sigma, (k, n))).astype(np.int64)
        for j in range(n):
            ej = np.concatenate([rot(e[r], j) for r in range(k)])[:m]
            sj = np.concatenate([rot(s[c], j) for c in range(k)])
            out[t, j, :m] = ej
            out[t, j, m:] = -sj
    return out


def check_algebra(rng, n, k, m):
    """(x^j b)_I - (A_I (x^j s) + (x^j e)_I) == 0 mod q: the n targets really share the lattice."""
    A, _ = build_basis(rng, n, k, m)
    Afull_rng = np.random.default_rng(12345)
    # rebuild with full rows to form b over the ring, then restrict
    Af = np.zeros((k * n, k * n), dtype=np.int64)
    for r in range(k):
        for c in range(k):
            Af[r * n:(r + 1) * n, c * n:(c + 1) * n] = negacyclic_matrix(Afull_rng.integers(0, Q, n))
    s = Afull_rng.integers(-2, 3, (k, n))
    e = Afull_rng.integers(-2, 3, (k, n))
    b = (Af @ s.reshape(-1) + e.reshape(-1)) % Q
    for j in range(n):
        bj = np.concatenate([rot(b[r * n:(r + 1) * n], j) for r in range(k)])
        sj = np.concatenate([rot(s[c], j) for c in range(k)])
        ej = np.concatenate([rot(e[r], j) for r in range(k)])
        assert np.all((bj[:m] - (Af[:m] @ sj + ej[:m])) % Q == 0), j
    return True


def log_gh(log_r_diag):
    """ln of the Gaussian-heuristic radius of a block given ln|r_ii|."""
    b = len(log_r_diag)
    return (np.sum(log_r_diag) / b) + (math.lgamma(b / 2 + 1) - (b / 2) * math.log(math.pi)) / b


def block_cvp_is_zero(Rblk, yblk):
    """Exact test: is 0 the closest vector of the projected block lattice to the target?"""
    b = Rblk.shape[0]
    Bi = IntegerMatrix.from_matrix(np.rint(Rblk.T * CVP_SCALE).astype(np.int64).tolist())
    M = GSO.Mat(Bi, float_type="double")
    M.update_gso()
    out = []
    for y in yblk:
        t = [int(v) for v in np.rint(y * CVP_SCALE)]
        r2 = float(sum(v * v for v in t))
        try:
            # a FRESH Enumeration object per query: a reused object returns stale solutions
            Enumeration(M).enumerate(0, b, r2 * (1 - 1e-6), 0, M.from_canonical(t))
            out.append(False)       # found a lattice point strictly closer than 0
        except EnumerationError:
            out.append(True)
    return out


def check_block_cvp(rng):
    """Negative/positive control: block_cvp_is_zero must agree with brute force in dimension 5."""
    import itertools
    agree = total = zeros = 0
    for _ in range(6):
        Bi = IntegerMatrix.from_matrix(rng.integers(-40, 40, (5, 5)).tolist())
        LLL.reduction(Bi)
        Bm = np.array([[Bi[i, j] for j in range(5)] for i in range(5)], dtype=np.float64)
        _, R = np.linalg.qr(Bm.T)
        Qm = np.linalg.qr(Bm.T)[0]
        for _ in range(10):
            t = rng.integers(-12, 12, 5).astype(np.float64)
            best = min(((t - np.array(c) @ Bm) ** 2).sum() for c in itertools.product(range(-4, 5), repeat=5))
            truth = abs(best - (t ** 2).sum()) < 1e-9
            got = block_cvp_is_zero(R, (t @ Qm)[None, :])[0]
            agree += (got == truth)
            zeros += truth
            total += 1
    assert agree == total, (agree, total)
    return zeros, total


def worker(args):
    set_name, seed, sigma, n_secrets, do_cvp = args
    n, k, m, betas = SETS[set_name]
    set_modulus(set_name)
    rng = np.random.default_rng(seed)
    FPLLL.set_random_seed(seed)
    A, B = build_basis(rng, n, k, m)
    d = B.shape[0]
    disp = displacements(rng, n, k, m, sigma, n_secrets) if sigma else None
    Bi = IntegerMatrix.from_matrix(B.tolist())
    LLL.reduction(Bi)
    # double-precision GSO loops in size reduction at d >= 224 on this lattice family;
    # arm64 fplll has no long double / dd, so use mpfr there
    if d > 160:
        FPLLL.set_precision(90)
    bkz = BKZReduction(GSO.Mat(Bi, float_type="double" if d <= 160 else "mpfr"))
    res = {"set": set_name, "seed": seed, "sigma": sigma, "d": d, "n": n, "betas": {}}
    t0 = time.time()
    for beta in betas:
        for _ in range(2):
            bkz(BKZ.Param(beta, strategies=STRATEGIES, max_loops=2, flags=BKZ.MAX_LOOPS))
        Bn = np.array([[Bi[i, j] for j in range(d)] for i in range(d)], dtype=np.float64)
        Qm, R = np.linalg.qr(Bn.T)
        rd = np.abs(np.diag(R))
        first = d - beta
        gh = math.exp(log_gh(np.log(rd[first:])))
        entry = {"r_first": float(rd[first]), "gh": gh, "bkz_seconds": time.time() - t0}
        if disp is not None:
            Y = disp.reshape(-1, d).astype(np.float64) @ Qm          # (n_secrets*n) x d
            Y = Y.reshape(n_secrets, n, d)
            v2 = (disp[:, 0, :].astype(np.float64) ** 2).sum(axis=1)
            r2 = (Y[:, :, first:] ** 2).sum(axis=2)                   # projected squared norms
            ok_norm = r2 <= rd[first] ** 2
            ok_prefix = np.all(np.abs(Y[:, :, :first]) < rd[:first] / 2, axis=2)
            ok_babai = ok_prefix & np.all(np.abs(Y[:, :, first:]) < rd[first:] / 2, axis=2)
            entry.update({
                "v2": v2.tolist(),
                "ratio_mean": float(r2.mean() / (v2.mean() * beta / d)),
                "ratio_min": float(r2.min(axis=1).mean() / r2.mean()),
                "norm_p1": float(ok_norm.mean()), "norm_any": float(ok_norm.any(axis=1).mean()),
                "babai_p1": float(ok_babai.mean()), "babai_any": float(ok_babai.any(axis=1).mean()),
                "norm_view0": float(ok_norm[:, 0].mean()), "babai_view0": float(ok_babai[:, 0].mean()),
            })
            # Beta-law independence prediction, conditional on each secret's true ||disp||^2
            F = stats.beta.cdf(np.minimum(rd[first] ** 2 / v2, 1.0), beta / 2.0, (d - beta) / 2.0)
            entry["model_p1"] = float(F.mean())
            entry["model_any"] = float((1 - (1 - F) ** n).mean())
            # empirical correlation of projected norms across views of the same secret, given V
            z = r2 / v2[:, None]
            entry["view_var_within"] = float(z.var(axis=1, ddof=1).mean())
            entry["view_var_beta"] = float(stats.beta.var(beta / 2.0, (d - beta) / 2.0))
            if do_cvp:
                cand = ok_prefix & (r2 <= (CVP_PREFILTER * gh) ** 2)
                ok_cvp = np.zeros_like(cand)
                idx = np.argwhere(cand)
                if len(idx):
                    yb = Y[idx[:, 0], idx[:, 1], first:]
                    got = block_cvp_is_zero(R[first:, first:], yb)
                    ok_cvp[idx[:, 0], idx[:, 1]] = got
                entry.update({"cvp_p1": float(ok_cvp.mean()), "cvp_any": float(ok_cvp.any(axis=1).mean()),
                              "cvp_view0": float(ok_cvp[:, 0].mean()), "cvp_enums": int(len(idx))})
        res["betas"][beta] = entry
        print(f"[{set_name} seed {seed}] beta {beta} done, {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    return res


def crossing(betas, ps, level=0.5):
    """Linear-interpolated beta at which p first reaches `level` (None if never)."""
    for i, p in enumerate(ps):
        if p >= level:
            if i == 0:
                return float(betas[0])
            p0, b0, b1 = ps[i - 1], betas[i - 1], betas[i]
            return b0 + (level - p0) / (p - p0) * (b1 - b0) if p > p0 else float(b1)
    return None


def main():
    mode, set_name = sys.argv[1], sys.argv[2]
    n, k, m, betas = SETS[set_name]
    set_modulus(set_name)
    assert check_algebra(np.random.default_rng(1), n, k, m)
    print(f"algebra check passed: all {n} rotated targets are BDD instances of the same lattice "
          f"(n={n}, k={k}, m={m}, m%n={m % n})")
    z, tot = check_block_cvp(np.random.default_rng(2))
    print(f"block-CVP check passed: enumeration agrees with brute force on {tot}/{tot} targets "
          f"({z} with 0 closest, {tot - z} with another point closer)")
    if mode == "pilot":
        r = worker((set_name, 1, 0.0, 0, False))
        for b in betas:
            e = r["betas"][b]
            print(f"beta {b:3d}  r_first {e['r_first']:8.2f}  gh {e['gh']:8.2f}  "
                  f"sigma@norm-threshold {e['r_first'] / math.sqrt(b):6.3f}  t={e['bkz_seconds']:.0f}s")
        return
    sigma = float(sys.argv[3])
    n_seeds = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    n_secrets = int(sys.argv[5]) if len(sys.argv) > 5 else 200
    do_cvp = (len(sys.argv) <= 6) or sys.argv[6] != "nocvp"
    with Pool(min(n_seeds, max(1, os.cpu_count() - 2))) as pool:
        results = pool.map(worker, [(set_name, 1000 + i, sigma, n_secrets, do_cvp) for i in range(n_seeds)])
    os.makedirs("data", exist_ok=True)
    out = f"data/rotation_amplified_decoding_{set_name}_sigma{sigma}.json"
    with open(out, "w") as f:
        json.dump(results, f)
    d = results[0]["d"]
    print(f"\n{set_name}: n={n} k={k} m={m} d={d} q={Q} sigma={sigma}  "
          f"{n_seeds} lattices x {n_secrets} secrets x {n} rotations")
    crits = ["norm", "cvp", "babai"] if do_cvp else ["norm", "babai"]
    hdr = f"{'beta':>4} {'E[min]/E':>8} {'var ratio':>9} |" + "".join(
        f" {c + ' p1':>9} {c + ' any':>9}" for c in crits) + f" | {'model p1':>8} {'model any':>9}"
    print(hdr)
    agg = {}
    for b in betas:
        es = [r["betas"][b] if b in r["betas"] else r["betas"][str(b)] for r in results]
        row = {key: float(np.mean([e[key] for e in es])) for key in es[0] if isinstance(es[0][key], float)}
        agg[b] = row
        print(f"{b:>4} {row['ratio_min']:>8.3f} {row['view_var_within'] / row['view_var_beta']:>9.3f} |"
              + "".join(f" {row[c + '_p1']:>9.4f} {row[c + '_any']:>9.4f}" for c in crits)
              + f" | {row['model_p1']:>8.4f} {row['model_any']:>9.4f}")
    print("\nblock size at 50% success (linear interpolation):")
    for c in crits + ["model"]:
        b1 = crossing(betas, [agg[b][c + "_p1"] for b in betas])
        bn = crossing(betas, [agg[b][c + "_any"] for b in betas])
        gain = f"{b1 - bn:5.2f}" if (b1 is not None and bn is not None) else "  n/a"
        print(f"  {c:<6} one view: {b1}   any of {n} views: {bn}   delta beta = {gain}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
