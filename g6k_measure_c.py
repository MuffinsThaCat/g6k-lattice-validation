"""
Measure c (baseline chances) and the rotation saving with G6K's real attack schedule
(pump-and-jump BKZ, dimensions for free), at block sizes where G6K actually sieves.
Same pre-registered readout as measure_baseline_chances_c.py:
  E    : Kannan-embedded target, progressive G6K pnj-BKZ (one tour per blocksize, jump 1);
         success = basis contains +-(disp_0, M). The real baseline, all implicit chances.
  T1   : target-free lattice, same schedule; norm event of view 0 in the last beta block.
  Tany : same basis, any of the n rotated views.
Cumulative success by blocksize, beta50 of each, ML c with T1 as the c = 1 control.

Run with the latticefail venv:
  ~/latticefail_env/venv312/bin/python g6k_measure_c.py <n> <k> <m> <q> <sigma> <bmin> <bmax> <n_inst> [workers]
"""
import json
import math
import os
import sys
import time
from multiprocessing import Pool

G6K_DIR = os.environ.get("G6K_DIR", os.path.expanduser("~/latticefail_env/g6k-stock"))
sys.path.insert(0, G6K_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy import stats
from fpylll import IntegerMatrix, FPLLL
from fpylll.tools.bkz_stats import dummy_tracer
from g6k import Siever, SieverParams
from g6k.algorithms.bkz import pump_n_jump_bkz_tour

import test_rotation_amplified_decoding_real as R
from measure_baseline_chances_c import fit_c, beta50, cum


def basis_np(g6k, d):
    return np.array([[g6k.M.B[i, j] for j in range(d)] for i in range(d)], dtype=np.float64)


def run_schedule(B, betas, after_tour, threads=1, seed=None):
    A = IntegerMatrix.from_matrix(B.tolist())
    from fpylll import LLL
    LLL.reduction(A)          # precision-adaptive wrapper; double may then hold on the reduced basis
    # arm64 fplll has no long double / dd, which G6K picks above d = 160: use mpfr there
    ft = os.environ.get("G6K_FT")
    M = Siever.MatGSO(A, float_type=(ft if ft else ("double" if A.nrows <= 160 else 80)))
    g6k = Siever(M, SieverParams(threads=threads), seed=seed)
    g6k.lll(0, g6k.full_n)
    if after_tour(g6k, 0):
        return
    for beta in betas:
        if beta < int(os.environ.get("G6K_FPLLL_BELOW", "30")):
            from fpylll import BKZ
            from fpylll.algorithms.bkz2 import BKZReduction
            BKZReduction(g6k.M)(BKZ.Param(beta, strategies=R.STRATEGIES, max_loops=1))
        else:
            pump_n_jump_bkz_tour(g6k, dummy_tracer, beta, jump=1)
        g6k.lll(0, g6k.full_n)
        if after_tour(g6k, beta):
            return


def worker(args):
    n, k, m, q, sigma, betas, seed = args
    R.Q = q
    rng = np.random.default_rng(seed)
    FPLLL.set_random_seed(seed)
    A, B = R.build_basis(rng, n, k, m)
    d = B.shape[0]
    disp = R.displacements(rng, n, k, m, sigma, 1)[0].astype(np.float64)
    v2 = float((disp[0] ** 2).sum())
    per_beta = {}

    # Embedded arm (the real baseline) runs first and stops at success; the target-free arm then
    # stops once both outcomes are settled: view 0 has hit (so cumulative T1 = Tany = 1 from there)
    # and the schedule has passed E's success block size (so F is known up to it, which is all the
    # c fit uses). G6K_EARLY_STOP=0 runs the target-free arm to bmax as before.
    s = (-disp[0, m:]).astype(np.int64)
    e = disp[0, :m].astype(np.int64)
    b = (A @ s + e) % q
    M = max(1, int(round(sigma)))
    Be = np.zeros((d + 1, d + 1), dtype=np.int64)
    Be[:d, :d] = B
    Be[d, :m] = b
    Be[d, d] = M
    target = np.concatenate([disp[0], [M]]).astype(np.int64)
    found = {"beta": None}

    def check(g6k, beta):
        for i in range(d + 1):
            row = np.array([g6k.M.B[i, j] for j in range(d + 1)], dtype=np.int64)
            if np.array_equal(row, target) or np.array_equal(row, -target):
                found["beta"] = beta
                return True
        return False

    t0 = time.time()
    run_schedule(Be, betas, check, seed=seed)
    t_emb = time.time() - t0

    early = os.environ.get("G6K_EARLY_STOP", "1") != "0"
    e_stop = found["beta"] if found["beta"] is not None else betas[-1]
    t1_seen = {"v": False}

    def probe(g6k, beta):
        if beta == 0:
            return False
        Bn = basis_np(g6k, d)
        Qm, Rr = np.linalg.qr(Bn.T)
        rd = np.abs(np.diag(Rr))
        first = d - beta
        r2 = ((disp @ Qm)[:, first:] ** 2).sum(axis=1)
        ok = r2 <= rd[first] ** 2
        F = float(stats.beta.cdf(min(rd[first] ** 2 / v2, 1.0), beta / 2.0, (d - beta) / 2.0))
        per_beta[beta] = {"t1": bool(ok[0]), "tany": bool(ok.any()), "F": F,
                          "nviews": int(ok.sum())}
        t1_seen["v"] = t1_seen["v"] or bool(ok[0])
        return early and t1_seen["v"] and beta >= e_stop

    t0 = time.time()
    run_schedule(B, betas, probe, seed=seed + 1)
    return {"seed": seed, "d": d, "v2": v2, "E_beta": found["beta"], "per_beta": per_beta,
            "t_tf": time.time() - t0, "t_emb": t_emb, "early_stop": early}


def main():
    n, k, m, q = map(int, sys.argv[1:5])
    sigma = float(sys.argv[5])
    bmin, bmax, n_inst = int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8])
    workers = int(sys.argv[9]) if len(sys.argv) > 9 else 4
    betas = list(range(10, bmax + 1)) if bmin <= 10 else list(range(10, bmin, 2)) + list(range(bmin, bmax + 1))
    t0 = time.time()
    with Pool(workers) as pool:
        recs = pool.map(worker, [(n, k, m, q, sigma, betas, 7000 + i) for i in range(n_inst)])
    tag = f"n{n}k{k}m{m}q{q}s{sigma}"
    path = f"data/g6k_c_{tag}.json"
    with open(path, "w") as f:
        json.dump(recs, f)
    # E is only probed until success; per_beta exists for every beta in the target-free arm
    E = np.array([[r["E_beta"] is not None and r["E_beta"] <= bt for bt in betas] for r in recs]).mean(0)
    T1 = np.array([cum([r["per_beta"][bt]["t1"] for bt in betas]) for r in recs]).mean(0)
    TA = np.array([cum([r["per_beta"][bt]["tany"] for bt in betas]) for r in recs]).mean(0)
    print(f"{tag} d={recs[0]['d']} n_inst={n_inst} wall={time.time()-t0:.0f}s "
          f"(per inst tf {np.mean([r['t_tf'] for r in recs]):.0f}s emb {np.mean([r['t_emb'] for r in recs]):.0f}s)")
    print("beta  E(real)  T1(1 view)  Tany(n views)")
    for i, bt in enumerate(betas):
        if 0 < TA[i] or 0 < E[i]:
            print(f"{bt:4d}  {E[i]:.2f}     {T1[i]:.2f}        {TA[i]:.2f}")
    print(f"beta50: E={beta50(E, betas):.1f}  T1={beta50(T1, betas):.1f}  Tany={beta50(TA, betas):.1f}")
    if n_inst >= 4:
        c_hat = fit_c(recs, betas)
        c1 = fit_c([dict(r, E_beta=next((bt for bt in betas if r["per_beta"][bt]["t1"]), None)) for r in recs], betas)
        rng = np.random.default_rng(1)
        ratios = []
        for _ in range(200):
            idx = rng.integers(0, len(recs), len(recs))
            sub = [recs[j] for j in idx]
            ce = fit_c(sub, betas)
            c1b = fit_c([dict(r, E_beta=next((bt for bt in betas if r["per_beta"][bt]["t1"]), None)) for r in sub], betas)
            ratios.append(ce / c1b)
        lo, hi = np.percentile(ratios, [2.5, 97.5])
        print(f"c_hat E = {c_hat:.2f}, control T1 = {c1:.2f}, ratio c_eff = {c_hat/c1:.2f}  95% CI [{lo:.2f}, {hi:.2f}]")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
