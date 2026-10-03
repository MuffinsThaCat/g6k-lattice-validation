"""
Round-2 item E' (PREREGISTERED_CRITERIA_R2.md): how many independent chances c does the standard progressive-BKZ
baseline already get on a planted LWE secret?   c_eff = ln(1 - P_full) / ln(1 - P_1).

  P_full(beta): fraction of instances whose planted vector (+-) is a ROW of the basis after progressive BKZ up to
                block size beta (cumulative over every earlier stage, position and tour: the real baseline).
  P_1(beta):    mean over the same instances of the model probability of ONE view at beta, from the REAL
                Gram-Schmidt profile after that stage: P[V^2 X <= r_{d-beta}], X ~ Beta(beta/2, (d-beta)/2).

Needs only fpylll + numpy + scipy.   Env: CB_N=40 CB_Q=97 CB_ETA=3 CB_M="28,32,36,40,44" CB_SEEDS=48 CB_SEED0=11
CB_BMAX=40 CB_TOURS=2 CB_NPROC=4 CB_OUT=res_c.json
"""
import json
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np
from fpylll import IntegerMatrix, LLL, BKZ, GSO, FPLLL
from scipy import special

N = int(os.environ.get("CB_N", "40"))
Q = int(os.environ.get("CB_Q", "97"))
ETA = int(os.environ.get("CB_ETA", "3"))
MS = [int(x) for x in os.environ.get("CB_M", "28,32,36,40,44").split(",")]
SEEDS = int(os.environ.get("CB_SEEDS", "48"))
SEED0 = int(os.environ.get("CB_SEED0", "11"))   # seeds 1-4 were used for parameter calibration and are never reused
BMAX = int(os.environ.get("CB_BMAX", "40"))
TOURS = int(os.environ.get("CB_TOURS", "2"))
NPROC = int(os.environ.get("CB_NPROC", "4"))
OUT = os.environ.get("CB_OUT", "res_c.json")


def cbd(eta, size, rng):
    return rng.integers(0, 2, (size, eta)).sum(1) - rng.integers(0, 2, (size, eta)).sum(1)


def make_instance(n, m, q, eta, seed):
    rng = np.random.default_rng(seed)
    A = rng.integers(0, q, (n, m))
    s, e = cbd(eta, n, rng), cbd(eta, m, rng)
    b = (s @ A + e) % q
    d = m + n + 1
    B = np.zeros((d, d), dtype=np.int64)
    B[:m, :m] = q * np.eye(m, dtype=np.int64)
    B[m:m + n, :m] = A
    B[m:m + n, m:m + n] = np.eye(n, dtype=np.int64)
    B[m + n, :m] = b
    B[m + n, m + n] = 1
    v = np.zeros(d, dtype=np.int64)
    v[:m], v[m:m + n], v[m + n] = e, -s, 1
    comb = B[m + n] - s @ B[m:m + n]
    comb[:m] -= q * np.rint(comb[:m] / q).astype(np.int64)
    assert np.array_equal(comb, v), "planted vector not in lattice"
    return B, v


def has_row(IM, v):
    t, tn = tuple(int(x) for x in v), tuple(-int(x) for x in v)
    for i in range(IM.nrows):
        row = tuple(IM[i, j] for j in range(IM.ncols))
        if row == t or row == tn:
            return True
    return False


def profile(IM):
    for ft in ("long double", "double"):
        try:
            M = GSO.Mat(IM, float_type=ft)
            M.update_gso()
            return np.array([M.get_r(i, i) for i in range(IM.nrows)])
        except (ValueError, RuntimeError):
            continue
    raise RuntimeError("no usable GSO float type")


def p_one(r, V2, beta, d):
    x = min(1.0, r[d - beta] / V2)
    return float(special.betainc(beta / 2.0, (d - beta) / 2.0, x))


def tour(IM, b):
    for ft in ("double", "long double", "mpfr"):
        try:
            BKZ.reduction(IM, BKZ.Param(block_size=b, strategies=BKZ.DEFAULT_STRATEGY, max_loops=1,
                                        flags=BKZ.MAX_LOOPS), float_type=ft)
            return
        except (RuntimeError, ValueError):
            continue
    raise RuntimeError("BKZ failed at every precision")


def one(args):
    m, seed = args
    FPLLL.set_random_seed(1000 * m + seed)
    B, v = make_instance(N, m, Q, ETA, 1000 * m + seed)
    d = B.shape[0]
    V2 = float(v @ v)
    IM = IntegerMatrix.from_matrix(B.tolist())
    LLL.reduction(IM)
    rec, p1 = {}, {}
    t0 = time.time()
    for b in range(10, min(BMAX, d - 1) + 1, 2):
        for _ in range(TOURS):
            tour(IM, b)
        rec[b] = has_row(IM, v)
        p1[b] = p_one(profile(IM), V2, b, d)
    return dict(m=m, seed=seed, d=d, V2=V2, rec=rec, p1=p1, sec=round(time.time() - t0, 1))


def main():
    tasks = [(m, s) for m in MS for s in range(SEED0, SEED0 + SEEDS)]
    print(f"{len(tasks)} instances, n={N} q={Q} eta={ETA}, m in {MS}, progressive BKZ 10..{BMAX} step 2 x{TOURS} tours", flush=True)
    rows = []
    with mp.get_context("fork").Pool(NPROC) as pool:
        for r in pool.imap_unordered(one, tasks):
            rows.append(r)
            print(f"m={r['m']} seed={r['seed']} d={r['d']} {r['sec']}s recovered_at={min([b for b, x in r['rec'].items() if x], default=None)}", flush=True)
            json.dump(rows, open(OUT, "w"))
    summarize(rows)


def summarize(rows):
    rng = np.random.default_rng(0)
    print("\n=== c_eff = ln(1-P_full)/ln(1-P_1), cells with both probabilities in (0.03, 0.97) ===")
    print(f"{'m':>4} {'d':>4} {'beta':>5} {'P_full':>7} {'P_1':>7} {'c_eff':>7}   95% CI")
    cells = []
    for m in sorted({r["m"] for r in rows}):
        rs = [r for r in rows if r["m"] == m]
        for b in sorted(rs[0]["rec"], key=int):
            pf = np.array([float(r["rec"][b]) for r in rs])
            p1 = np.array([r["p1"][b] for r in rs])
            Pf, P1 = pf.mean(), p1.mean()
            if not (0.03 < Pf < 0.97 and 0.03 < P1 < 0.97):
                continue
            c = math.log(1 - Pf) / math.log(1 - P1)
            cs = []
            for _ in range(2000):
                ix = rng.integers(0, len(rs), len(rs))
                a, bb = pf[ix].mean(), p1[ix].mean()
                if 0.0 < a < 1.0 and 0.0 < bb < 1.0:
                    cs.append(math.log(1 - a) / math.log(1 - bb))
            lo, hi = (np.percentile(cs, 2.5), np.percentile(cs, 97.5)) if len(cs) > 200 else (float("nan"),) * 2
            cells.append(c)
            print(f"{m:>4} {rs[0]['d']:>4} {b:>5} {Pf:>7.2f} {P1:>7.3f} {c:>7.2f}   [{lo:.2f}, {hi:.2f}]")
    if len(cells) >= 3:
        print(f"\nqualifying cells: {len(cells)}; median c_eff = {np.median(cells):.2f} (min {min(cells):.2f}, max {max(cells):.2f})")
    else:
        print(f"\nonly {len(cells)} qualifying cells (< 3): c is UNMEASURED")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summarize":
        summarize(json.load(open(sys.argv[2])))
    else:
        main()
