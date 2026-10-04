"""
Measure c, the number of independent chances the REAL primal baseline (embedded-target
progressive BKZ) already gets, which Round 47 left unmeasured (scope limit 2).

Pre-registered (written before any run):
  * Same instance family, sigma and progressive schedule as test_rotation_amplified_decoding_real.py.
  * Per instance (fresh A, fresh secret), paired arms on the same A:
      E    : Kannan-embedded target t0 = (b_I, 0 | M), real fpylll progressive BKZ; success = some
             basis row equals +-(disp_0, M). This is the real baseline with ALL its implicit chances
             (every tour, every position).
      T1   : target-free BKZ, same schedule; norm event of view 0 in the last block (c = 1 event).
      Tany : same basis, any of the n rotated views (the Round 47 attack, last block only).
    All three are CUMULATIVE over the progressive schedule (success by beta).
  * c_hat = maximum-likelihood c in  P(E found by beta_j) = E_V[1 - prod_{i<=j} (1-F_i)^c],
    F_i = Beta(beta_i/2,(d-beta_i)/2) law at the target-free profile of the same instance.
    95% CI by bootstrap over instances.
  * Readout: beta50 of E, T1, Tany; c_hat. Decision (user's call, recorded only):
      CI upper < 16  -> the adopted c=16 credit is conservative; c_hat credit can be computed
      CI contains 16 -> keep the adopted credit
      c_hat > 16     -> the adopted credit is too high.

Usage: .venv/bin/python3 measure_baseline_chances_c.py <set> <sigma> <n_inst> [workers]
"""
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar

from fpylll import BKZ, GSO, IntegerMatrix, LLL, FPLLL
from fpylll.algorithms.bkz2 import BKZReduction

import test_rotation_amplified_decoding_real as R


def reduce_schedule(Bi, d, betas, after_each):
    bkz = BKZReduction(GSO.Mat(Bi, float_type="double" if d <= 160 else "mpfr"))
    for beta in betas:
        for _ in range(2):
            bkz(BKZ.Param(beta, strategies=R.STRATEGIES, max_loops=2, flags=BKZ.MAX_LOOPS))
            if after_each(beta):
                return


def worker(args):
    set_name, seed, sigma = args
    n, k, m, betas = R.SETS[set_name]
    R.set_modulus(set_name)
    q = R.Q
    rng = np.random.default_rng(seed)
    FPLLL.set_random_seed(seed)
    A, B = R.build_basis(rng, n, k, m)
    d = B.shape[0]
    disp = R.displacements(rng, n, k, m, sigma, 1)[0]          # n x d
    if d > 160:
        FPLLL.set_precision(90)
    v2 = float((disp[0].astype(np.float64) ** 2).sum())

    # ---- target-free arm: T1 / Tany per beta, plus model F per beta ----
    Bi = IntegerMatrix.from_matrix(B.tolist())
    LLL.reduction(Bi)
    per_beta = {}

    def probe_tf(beta):
        if beta in per_beta and per_beta[beta]["pass"] == 1:
            Bn = np.array([[Bi[i, j] for j in range(d)] for i in range(d)], dtype=np.float64)
            Qm, Rr = np.linalg.qr(Bn.T)
            rd = np.abs(np.diag(Rr))
            first = d - beta
            Y = disp.astype(np.float64) @ Qm
            r2 = (Y[:, first:] ** 2).sum(axis=1)
            ok = r2 <= rd[first] ** 2
            F = float(stats.beta.cdf(min(rd[first] ** 2 / v2, 1.0), beta / 2.0, (d - beta) / 2.0))
            per_beta[beta].update({"t1": bool(ok[0]), "tany": bool(ok.any()), "F": F})
        else:
            per_beta[beta] = {"pass": 1}
        return False

    reduce_schedule(Bi, d, betas, probe_tf)

    # ---- embedded arm: real baseline ----
    s = -disp[0, m:]
    e = disp[0, :m]
    b = (A @ s + e) % q
    M = max(1, int(round(sigma)))
    Be = np.zeros((d + 1, d + 1), dtype=np.int64)
    Be[:d, :d] = B
    Be[d, :m] = b
    Be[d, d] = M
    target = np.concatenate([disp[0], [M]])
    Bie = IntegerMatrix.from_matrix(Be.tolist())
    found = {"beta": None, "call": 0}

    def check(beta):
        found["call"] += 1
        for i in range(d + 1):
            row = np.array([Bie[i, j] for j in range(d + 1)], dtype=np.int64)
            if np.array_equal(row, target) or np.array_equal(row, -target):
                found["beta"] = beta
                return True
        return False

    LLL.reduction(Bie)
    if not check(0):
        reduce_schedule(Bie, d + 1, betas, check)
    return {"seed": seed, "v2": v2, "E_beta": found["beta"], "per_beta": per_beta}


def cum(flags):
    out, acc = [], False
    for f in flags:
        acc = acc or f
        out.append(acc)
    return out


def loglik(c, recs, betas):
    ll = 0.0
    for r in recs:
        surv = 1.0
        prev_surv = 1.0
        hit = None
        for bt in betas:
            F = r["per_beta"][str(bt) if str(bt) in r["per_beta"] else bt]["F"]
            prev_surv = surv
            surv *= (1.0 - F) ** c
            if r["E_beta"] is not None and r["E_beta"] <= bt and hit is None:
                hit = max(prev_surv - surv, 1e-300)
        if r["E_beta"] is not None and r["E_beta"] == 0:
            continue                                   # found by LLL: uninformative
        ll += math.log(hit) if hit is not None else math.log(max(surv, 1e-300))
    return ll


def fit_c(recs, betas):
    res = minimize_scalar(lambda lc: -loglik(math.exp(lc), recs, betas), bounds=(-3, 9), method="bounded")
    return math.exp(res.x)


def beta50(curve, betas):
    for i, p in enumerate(curve):
        if p >= 0.5:
            if i == 0:
                return betas[0]
            p0 = curve[i - 1]
            return betas[i - 1] + (0.5 - p0) / (p - p0) * (betas[i] - betas[i - 1])
    return float("nan")


def main():
    set_name, sigma, n_inst = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    betas = R.SETS[set_name][3]
    t0 = time.time()
    with Pool(workers) as pool:
        recs = pool.map(worker, [(set_name, 5000 + i, sigma) for i in range(n_inst)])
    path = f"data/baseline_chances_c_{set_name}_sigma{sigma}.json"
    with open(path, "w") as f:
        json.dump(recs, f)

    E = np.array([[r["E_beta"] is not None and r["E_beta"] <= bt for bt in betas] for r in recs]).mean(0)
    T1 = np.array([cum([r["per_beta"][bt]["t1"] for bt in betas]) for r in recs]).mean(0)
    TA = np.array([cum([r["per_beta"][bt]["tany"] for bt in betas]) for r in recs]).mean(0)
    print(f"{set_name} sigma={sigma} n_inst={n_inst} wall={time.time()-t0:.0f}s")
    print("beta  E(real)  T1(1 view)  Tany(n views)")
    for i, bt in enumerate(betas):
        print(f"{bt:4d}  {E[i]:.2f}     {T1[i]:.2f}        {TA[i]:.2f}")
    print(f"beta50: E={beta50(E, betas):.1f}  T1={beta50(T1, betas):.1f}  Tany={beta50(TA, betas):.1f}")
    c_hat = fit_c(recs, betas)
    rng = np.random.default_rng(1)
    boots = [fit_c([recs[j] for j in rng.integers(0, len(recs), len(recs))], betas) for _ in range(200)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    c1 = fit_c([dict(r, E_beta=next((bt for bt in betas if r["per_beta"][bt]["t1"]), None)) for r in recs], betas)
    print(f"c_hat (real embedded baseline) = {c_hat:.2f}  95% CI [{lo:.2f}, {hi:.2f}]")
    print(f"control: same fit on T1 (true c=1 event) = {c1:.2f}")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
