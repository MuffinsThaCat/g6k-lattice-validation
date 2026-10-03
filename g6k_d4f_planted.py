"""
Real-G6K measurement of DIMENSIONS FOR FREE on a PLANTED vector (the quantity a primal uSVP attack needs).

Why this exists (see PREREGISTERED_CRITERIA.md, item D): the paper's baseline uses Ducas's formula
f = beta*ln(4/3)/ln(beta/2 pi e).  G6K's own fit f = 11.46 + 0.0757 d (eprint 2019/089, Fig. 3b) is for
EXACT SVP at d = 60..100 and the authors call it a local approximation (footnote 17); the GPU-sieving table
(eprint 2021/141) shows D4F 28-35 at n = 158-180 but for 1.05-approximate SVP, which is easier.  Neither
tells us the free dimensions for recovering a planted LWE secret.  This script measures exactly that.

Method
  1. Build an LWE-style Kannan embedding  L = rows [[q I_m,0,0],[A,I_n,0],[b,0,1]],  b = sA + e (mod q),
     with CBD(eta) secret and error.  The planted lattice vector is v = (e, -s, 1).
  2. Pre-reduce with fpylll BKZ (progressive, up to beta_pre).
  3. For every block size beta_blk compute rho = ||pi_kappa(v)|| / gh(block), kappa = d - beta_blk (v is the
     shortest vector of the last block iff rho < ~1).  Pick the smallest beta_blk with rho <= RHO
     (threshold regime, the same regime Ducas's f is derived for).
  4. For each f in F_LIST run G6K's pump on that block with dim4free = f (sieve context dimension
     beta_blk - f, G6K's on-the-fly lifting active) and test whether the block's first Gram-Schmidt norm
     now equals ||pi_kappa(v)||, i.e. whether the planted vector was found and inserted.
  5. f_obs = largest f with success.  Compare with Ducas's f(beta_blk) and G6K's fit.

Run inside the activated G6K environment (cwd = the g6k checkout, like g6k_opcount_vs_walltime.py).
Environment knobs:
  G6K_D4F_INST="110:134,120:132"   n:m pairs (q=1009, CBD(2)); see calibration in the findings note
  G6K_D4F_Q=1009  G6K_D4F_ETA=2  G6K_D4F_SEEDS=2  G6K_D4F_RHO=0.97  G6K_D4F_BPRE=44
  G6K_D4F_FS="0,4,8,10,12,14,16,18,20,22,24,26"  G6K_THREADS=1  G6K_MAX_SECONDS=9000
  G6K_D4F_OUT=res_d4f.json

STATUS: written against G6K's public API (g6k.algorithms.pump.pump, g6k.utils.stats.dummy_tracer,
Siever, SieverParams); NOT executed by the author. Anything that does not match the installed
version is reported, not assumed.
"""
import json
import math
import os
import sys
import time

sys.path.insert(0, os.environ.get("G6K_DIR", os.getcwd()))

import numpy as np
from fpylll import IntegerMatrix, LLL, BKZ, FPLLL
from g6k import Siever

try:
    from g6k.siever_params import SieverParams
except ImportError:
    from g6k.siever import SieverParams
try:
    from g6k.siever import SaturationError
except ImportError:
    class SaturationError(RuntimeError):
        pass
try:
    from g6k.algorithms.pump import pump
    from g6k.utils.stats import dummy_tracer
except ImportError as exc:  # report, do not guess
    print("FATAL: cannot import g6k.algorithms.pump / g6k.utils.stats.dummy_tracer:", exc)
    sys.exit(2)

INST = [tuple(int(t) for t in p.split(":")) for p in os.environ.get("G6K_D4F_INST", "110:134,120:132").split(",")]
Q = int(os.environ.get("G6K_D4F_Q", "1009"))
ETA = int(os.environ.get("G6K_D4F_ETA", "2"))
SEEDS = int(os.environ.get("G6K_D4F_SEEDS", "2"))
RHO = float(os.environ.get("G6K_D4F_RHO", "0.97"))
RHO_MIN = float(os.environ.get("G6K_D4F_RHO_MIN", "0.60"))   # below this the planted vector is already (nearly) in the reduced prefix: trivial instance
BPRE = int(os.environ.get("G6K_D4F_BPRE", "44"))
FS = [int(x) for x in os.environ.get("G6K_D4F_FS", "0,4,8,10,12,14,16,18,20,22,24,26").split(",")]
THREADS = int(os.environ.get("G6K_THREADS", "1"))
MAX_SECONDS = float(os.environ.get("G6K_MAX_SECONDS", "9000"))
OUT = os.environ.get("G6K_D4F_OUT", "res_d4f.json")
T0 = time.time()


def cbd(eta, size, rng):
    return rng.integers(0, 2, (size, eta)).sum(1) - rng.integers(0, 2, (size, eta)).sum(1)


def make_instance(n, m, q, eta, seed):
    rng = np.random.default_rng(seed)
    A = rng.integers(0, q, (n, m))
    s = cbd(eta, n, rng)
    e = cbd(eta, m, rng)
    b = (s @ A + e) % q
    d = m + n + 1
    B = np.zeros((d, d), dtype=np.int64)
    B[:m, :m] = q * np.eye(m, dtype=np.int64)
    B[m:m + n, :m] = A
    B[m:m + n, m:m + n] = np.eye(n, dtype=np.int64)
    B[m + n, :m] = b
    B[m + n, m + n] = 1
    v = np.zeros(d, dtype=np.int64)
    v[:m] = e
    v[m:m + n] = -s
    v[m + n] = 1
    # self-check: v must be an integer combination of the rows (last row - s*A rows - q*k)
    comb = B[m + n] - s @ B[m:m + n]
    comb[:m] -= q * np.rint(comb[:m] / q).astype(np.int64)
    assert np.array_equal(comb, v), "planted vector is not in the lattice (instance construction bug)"
    return B, v


def to_np(IM):
    d = IM.nrows
    return np.array([[IM[i, j] for j in range(IM.ncols)] for i in range(d)], dtype=np.float64)


def prereduce(B):
    IM = IntegerMatrix.from_matrix(B.tolist())
    LLL.reduction(IM)
    strat = os.environ.get("G6K_STRATEGIES") or BKZ.DEFAULT_STRATEGY
    for bs in range(20, BPRE + 1, 4):
        BKZ.reduction(IM, BKZ.Param(block_size=bs, strategies=strat, max_loops=2,
                                    flags=BKZ.MAX_LOOPS | BKZ.AUTO_ABORT))
    return IM


def gso_norms_and_Q(Bnp):
    Qm, R = np.linalg.qr(Bnp.T)
    return np.abs(np.diag(R)), Qm


def gh(r_block):
    k = len(r_block)
    return math.exp(math.lgamma(k / 2 + 1) / k) / math.sqrt(math.pi) * math.exp(np.mean(np.log(r_block)))


def f_ducas(b):
    return b * math.log(4 / 3) / math.log(b / (2 * math.pi * math.e))


def f_g6k_fit(b):
    return 11.46 + 0.0757 * b


def pick_block(Bnp, v):
    d = Bnp.shape[0]
    r, Qm = gso_norms_and_Q(Bnp)
    y = Qm.T @ v.astype(np.float64)
    out = None
    for beta in range(30, min(d, 140) + 1):
        k = d - beta
        proj = math.sqrt(float(np.sum(y[k:] ** 2)))
        rho = proj / gh(r[k:])
        if rho <= RHO and out is None:
            out = (beta, rho, proj)
    if out is not None and out[1] < RHO_MIN:
        return None          # trivial: pi_kappa(v) is tiny, the block choice would say nothing about d4f
    return out


def one_pump(IM, v, kappa, beta, f):
    """Run G6K pump on block [kappa, kappa+beta) with f dimensions for free; success iff the planted
    vector's projection now is the block's first Gram-Schmidt vector."""
    params = SieverParams(threads=THREADS, default_sieve="bgj1", reserved_n=beta + 2)
    M = IntegerMatrix.from_matrix([[IM[i, j] for j in range(IM.ncols)] for i in range(IM.nrows)])
    g6k = Siever(M, params)
    t = time.time()
    status = "ok"
    try:
        pump(g6k, dummy_tracer, kappa, beta, f, down_sieve=True, verbose=False)
    except SaturationError:
        status = "saturation_error"
    except Exception as exc:  # report the API/mismatch, do not hide it
        status = "error:" + type(exc).__name__ + ":" + str(exc)[:80]
    now = to_np(g6k.M.B)
    r, Qm = gso_norms_and_Q(now)
    d = now.shape[0]
    y = Qm.T @ v.astype(np.float64)
    target = float(np.sum(y[kappa:] ** 2))
    # Success means the PLANTED projection was found: the block's first GS norm equals ||pi_kappa(v)||.
    # A strictly shorter first GS norm means some OTHER lattice vector shorter than the planted one exists
    # in the block (expected for ~rho^beta of instances); that run says nothing about planted recovery, so it
    # is reported as inconclusive and excluded from f_obs instead of being counted as a success.
    tol = 1e-4
    if r[kappa] ** 2 < target * (1 - tol):
        status = (status + "|other_shorter_vector") if status != "ok" else "other_shorter_vector"
        found = None
    else:
        found = bool(r[kappa] ** 2 <= target * (1 + tol))
    return found, status, time.time() - t


def main():
    FPLLL.set_random_seed(0x1337)
    results = []
    for (n, m) in INST:
        for seed in range(SEEDS):
            if time.time() - T0 > MAX_SECONDS:
                print("time budget reached; stopping", flush=True)
                break
            B, v = make_instance(n, m, Q, ETA, 1000 * n + seed)
            t = time.time()
            IM = prereduce(B)
            pick = pick_block(to_np(IM), v)
            print(f"n={n} m={m} seed={seed} d={B.shape[0]} prereduce {time.time() - t:.0f}s pick={pick}", flush=True)
            if pick is None:
                results.append(dict(n=n, m=m, seed=seed, note="no usable block: rho never <= RHO, or planted vector trivial (rho < RHO_MIN)"))
                continue
            beta, rho, proj = pick
            kappa = B.shape[0] - beta
            row = dict(n=n, m=m, seed=seed, d=B.shape[0], beta=beta, rho=rho,
                       f_ducas=f_ducas(beta), f_g6k_fit=f_g6k_fit(beta), per_f={})
            for f in FS:
                if f >= beta - 20 or time.time() - T0 > MAX_SECONDS:
                    break
                found, status, sec = one_pump(IM, v, kappa, beta, f)
                row["per_f"][f] = dict(found=found, status=status, sec=round(sec, 1))
                print(f"   f={f:>3} sieve dim {beta - f:>3}  found={found}  {status}  {sec:.1f}s", flush=True)
            if any(r["found"] is None for r in row["per_f"].values()):
                row["note"] = "a strictly shorter non-planted vector exists in the block: instance inconclusive"
                results.append(row)
                print("   => inconclusive (another vector shorter than the planted one); excluded", flush=True)
                continue
            ok = [f for f, r in row["per_f"].items() if r["found"]]
            row["f_obs"] = max(ok) if ok else None
            results.append(row)
            with open(OUT, "w") as fh:
                json.dump(results, fh, indent=1)
            print(f"   => f_obs={row['f_obs']}  Ducas f={row['f_ducas']:.1f}  G6K fit={row['f_g6k_fit']:.1f}", flush=True)
    done = [r for r in results if r.get("f_obs") is not None]
    print("\n=== planted-vector dimensions for free (f_obs = largest f with the planted vector found) ===")
    print(f"{'beta':>5} {'rho':>5} {'f_obs':>6} {'Ducas':>6} {'excess':>7} {'G6K fit':>8}")
    for r in sorted(done, key=lambda r: r["beta"]):
        print(f"{r['beta']:>5} {r['rho']:>5.2f} {r['f_obs']:>6} {r['f_ducas']:>6.1f} "
              f"{r['f_obs'] - r['f_ducas']:>7.1f} {r['f_g6k_fit']:>8.1f}")
    if done:
        ex = [r["f_obs"] - r["f_ducas"] for r in done]
        print(f"mean excess over Ducas: {np.mean(ex):.2f} dims  (= {0.292 * np.mean(ex):.2f} bits at 0.292/dim); n={len(done)}")
        bs = np.array([r["beta"] for r in done], float)
        if len(done) >= 4 and bs.max() > bs.min():
            slope = np.polyfit(bs, np.array(ex), 1)[0]
            print(f"excess vs beta slope: {slope:+.3f} dims per unit beta (negative => excess shrinks with beta)")
    json.dump(results, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
