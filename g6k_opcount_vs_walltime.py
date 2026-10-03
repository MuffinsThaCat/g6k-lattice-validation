"""
Real-G6K measurement of operation counts and wall-clock for the sieve, on x86-64/AVX2.

Purpose (see paper Round 40, Reconciliation 1): the paper credits wall-clock speedups measured on
a numpy sieve against gate counts. G6K exposes its own operation counters
(Siever.all_statistics: xorpopcnt_total, fullscprods_total, reds_total, ...), which are the
operations a circuit-level count (the Kyber specification's XOR-popcount + inner-product loop)
models. This script measures, in real G6K:

  E1  Scaling and constants. Time and operation counts of one full sieve at block sizes BETAS.
      Fits log2(time) and log2(ops) against beta (slope, intercept). Reports time per operation,
      which shows how fast a G6K operation already is (the numpy baseline in the paper's tests
      is far slower per operation; that gap is why batching showed a large wall-clock credit and
      zero operation credit).
  E2  Approximate termination in real G6K. Same lattices and seeds, saturation_ratio varied
      (G6K's default 0.5 is already an early stop). Reports the time and operation ratio against
      the default, i.e. the termination gain measured against the production baseline instead of a
      run-to-completion numpy baseline.

Run inside the activated G6K environment (see colab_g6k_setup.sh). Environment knobs:
  G6K_BETAS="30,36,42,48,54,60"  G6K_SEEDS=5  G6K_SATS="0.3,0.5,0.7,0.9"  G6K_E2_BETA=48
  G6K_ALG=bgj1  G6K_THREADS=1  G6K_MAX_SECONDS=2400  G6K_OUT=g6k_opcount_results.json

STATUS: written against G6K's public source (Siever.__call__, all_statistics, SieverParams names as
listed in siever.pyx); NOT executed by the author (no x86 machine available). API details that could
differ on the installed version are handled defensively and reported, not assumed.
"""
import json
import math
import os
import sys
import time

# G6K's bootstrap builds the extension IN PLACE (never installs it), and Python puts the script's
# own directory on sys.path, not the current one. Run from the g6k checkout (or set G6K_DIR).
sys.path.insert(0, os.environ.get("G6K_DIR", os.getcwd()))

import numpy as np
from fpylll import IntegerMatrix, LLL, BKZ, FPLLL
from g6k import Siever

try:
    from g6k.siever_params import SieverParams
except ImportError:
    from g6k.siever import SieverParams

try:  # raised by Siever.bgj1_sieve -> check_saturation() for dim >= 50 when saturation is not reached
    from g6k.siever import SaturationError
except ImportError:
    class SaturationError(RuntimeError):
        pass

BETAS = [int(x) for x in os.environ.get("G6K_BETAS", "40,46,52,58,64,70").split(",")]
SEEDS = int(os.environ.get("G6K_SEEDS", "5"))
SATS = [float(x) for x in os.environ.get("G6K_SATS", "0.3,0.5,0.7,0.9").split(",")]
E2_BETA = int(os.environ.get("G6K_E2_BETA", "48"))
ALG = os.environ.get("G6K_ALG", "bgj1")
THREADS = int(os.environ.get("G6K_THREADS", "1"))
MAX_SECONDS = float(os.environ.get("G6K_MAX_SECONDS", "2400"))
OUT = os.environ.get("G6K_OUT", "g6k_opcount_results.json")
RACE_WARN = float(os.environ.get("G6K_RACE_WARN", "1e-3"))  # untuned; check first multi-thread output
COUNTERS = ["xorpopcnt_total", "fullscprods_total", "reds_total", "2reds_total", "3reds",
            "dataraces_total", "dataraces_replaced_was_saturated", "replacementfailures_total"]
T_START = time.time()


def over_budget():
    return time.time() - T_START > MAX_SECONDS


def make_lattice(dim, seed):
    FPLLL.set_random_seed(seed)
    A = IntegerMatrix.random(dim, "qary", k=dim // 2, bits=11)
    A = LLL.reduction(A)
    BKZ.reduction(A, BKZ.Param(block_size=min(20, dim)))  # in place; do not rely on its return value
    return A


def read_counters(g6k):
    out = {}
    try:
        s = g6k.stats
        if isinstance(s, dict):
            for k in COUNTERS:
                if k in s:
                    out[k] = float(s[k])
    except Exception:
        pass
    for k in COUNTERS:
        if k not in out:
            try:
                out[k] = float(g6k.get_stat(k))
            except Exception:
                out[k] = float("nan")
    return out


def best_lift_norm(g6k):
    # best_lifts() returns triples (index, squared length, vector); take the entry at index 0
    # (the shortest vector of the whole sieved context).
    try:
        for item in g6k.best_lifts():
            if item[0] == 0:
                return float(item[1])
    except Exception:
        pass
    return float("nan")


def shortest_db_norm(g6k, A, dim):
    """Squared length of the shortest vector in the sieve database. best_lifts() only lists
    positions where a lift improves on the basis, so it is often empty; the database minimum is
    always available. itervalues() gives coordinates wrt the current basis A (reduced in place)."""
    try:
        B = np.array([[A[i, j] for j in range(dim)] for i in range(dim)], dtype=np.float64)
        X = np.array(list(g6k.itervalues()), dtype=np.float64)
        if X.size == 0:
            return float("nan")
        V = X @ B
        return float(np.min(np.einsum("ij,ij->i", V, V)))
    except Exception:
        return float("nan")


def nanmean_safe(xs):
    xs = [x for x in xs if not math.isnan(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def run_sieve(dim, seed, saturation_ratio=None):
    A = make_lattice(dim, seed)
    # reserved_n=dim sizes the database for this dimension (as G6K's own plain_sieve.py does).
    kw = dict(threads=THREADS, default_sieve=ALG, reserved_n=dim)
    if saturation_ratio is not None:
        kw["saturation_ratio"] = saturation_ratio
    g6k = Siever(A, SieverParams(**kw), seed=seed)
    g6k.initialize_local(0, 0, dim)
    saturated = True
    t0 = time.perf_counter()
    try:
        g6k(alg=ALG)
    except SaturationError:
        # G6K's own pump() catches this too. The work was done and the counters are valid, but the
        # run stopped before reaching the requested saturation: record it, never mix it in silently.
        saturated = False
    wall = time.perf_counter() - t0
    c = read_counters(g6k)
    c["saturated"] = saturated
    if math.isnan(c.get("xorpopcnt_total", float("nan"))) and ALG in ("bgj1", "triple_mt"):
        sys.exit("FATAL: G6K operation counters are not compiled in. Rebuild with -DENABLE_STATS "
                 "(colab_g6k_setup.sh does this); without them the op-count results are empty.")
    c["wall"] = wall
    c["best_norm"] = shortest_db_norm(g6k, A, dim)
    if math.isnan(c["best_norm"]):
        c["best_norm"] = best_lift_norm(g6k)
    return c


def lg(x):
    return math.log2(x) if x and x > 0 and not math.isnan(x) else float("nan")


def fit(xs, ys):
    pts = [(x, y) for x, y in zip(xs, ys) if not math.isnan(y)]
    if len(pts) < 3:
        return float("nan"), float("nan")
    slope, icpt = np.polyfit([p[0] for p in pts], [p[1] for p in pts], 1)
    return float(slope), float(icpt)


def main():
    print(f"G6K op-count vs wall-clock | alg={ALG} threads={THREADS} betas={BETAS} seeds={SEEDS}")
    cpu = "unknown"
    try:
        with open("/proc/cpuinfo") as f:
            cpu = next((l.split(":", 1)[1].strip() for l in f if l.startswith("model name")), "unknown")
    except OSError:
        pass
    results = {"env": {"alg": ALG, "threads": THREADS, "cpu": cpu, "nproc": os.cpu_count(),
                       "seeds": SEEDS, "betas": BETAS, "e2_beta": E2_BETA, "sats": SATS},
               "E1": {}, "E2": {}}
    print(f"cpu={cpu} nproc={os.cpu_count()}")

    print("\nE1: scaling (default saturation_ratio)")
    print(f"  {'beta':>4} {'wall_s':>9} {'log2 time':>10} {'xorpopcnt':>12} {'fullscprod':>12} "
          f"{'log2 ops':>9} {'ns/op':>8}")
    rows = []
    for beta in BETAS:
        if over_budget():
            print("  time budget reached; stopping E1")
            break
        if rows:  # project this beta's cost from the last one (~2^(0.4*dbeta) per step) before starting it
            proj = rows[-1][1] * SEEDS * 2 ** (0.4 * (beta - rows[-1][0]))
            if time.time() - T_START + proj > MAX_SECONDS:
                print(f"  beta={beta} projected {proj:.0f}s would exceed the budget; stopping E1")
                break
        runs = [run_sieve(beta, 1000 + s) for s in range(SEEDS)]
        wall = float(np.mean([r["wall"] for r in runs]))
        xpc = float(np.nanmean([r["xorpopcnt_total"] for r in runs]))
        fsp = float(np.nanmean([r["fullscprods_total"] for r in runs]))
        ops = (0 if math.isnan(xpc) else xpc) + (0 if math.isnan(fsp) else fsp)
        rows.append((beta, wall, ops, xpc))
        ns = 1e9 * wall / ops if ops else float("nan")
        nsat = sum(1 for r in runs if r["saturated"])
        if nsat < len(runs):
            print(f"  WARNING beta={beta}: {len(runs) - nsat}/{len(runs)} runs stopped before reaching "
                  "the requested saturation (SaturationError); their times/counts are still recorded")
        dr = nanmean_safe([r["dataraces_total"] for r in runs])
        drs = nanmean_safe([r["dataraces_replaced_was_saturated"] for r in runs])  # nan at stats level 1
        if THREADS == 1 and dr and not math.isnan(dr):
            print(f"  NOTE beta={beta}: {dr:.3g} data races with 1 thread; unexpected, investigate")
        if not math.isnan(dr) and ops and dr / ops > RACE_WARN:
            print(f"  WARNING beta={beta}: data races {dr:.3g} = {dr/ops:.1e} of ops "
                  f"(saturated-overwrites {drs:.3g}); saturation/termination may be unreliable")
        results["E1"][str(beta)] = {"saturated_runs": nsat, "datarace_frac": (dr / ops if ops else float("nan")),
                                     "wall": wall, "xorpopcnt": xpc, "fullscprods": fsp,
                                     "ns_per_op": ns, "runs": runs}
        print(f"  {beta:>4} {wall:>9.3f} {lg(wall):>10.2f} {xpc:>12.3g} {fsp:>12.3g} "
              f"{lg(ops):>9.2f} {ns:>8.2f}", flush=True)
    if len(rows) >= 3:
        b = [r[0] for r in rows]
        st, ci = fit(b, [lg(r[1]) for r in rows])
        so, co = fit(b, [lg(r[2]) for r in rows])
        sx, cx = fit(b, [lg(r[3]) for r in rows])   # xor-popcounts alone: the cheap, dominant unit
        results["E1"]["fit"] = {"time_slope": st, "time_intercept": ci,
                                "ops_slope": so, "ops_intercept": co,
                                "xorpopcnt_slope": sx, "xorpopcnt_intercept": cx}
        print(f"  fit log2(xorpopcnt only) = {sx:.3f}*beta + {cx:.2f}  "
              "(ops = xorpopcnt + fullscprods adds two units of very different cost; "
              "quote the components, not only the sum)")
        print(f"\n  fit log2(time) = {st:.3f}*beta + {ci:.2f}   |   fit log2(ops) = {so:.3f}*beta + {co:.2f}")
        print("  (0.292 is the asymptotic core-SVP slope; the intercept is the constant c the paper fits.)")

    print(f"\nE2: termination (beta={E2_BETA}); ratios vs saturation_ratio=0.5 on identical lattices/seeds")
    base = [run_sieve(E2_BETA, 2000 + s, 0.5) for s in range(SEEDS)]
    print(f"  {'sat':>5} {'time bits':>10} {'op bits':>9} {'same best vector':>17}")
    for sat in SATS:
        if over_budget():
            print("  time budget reached; stopping E2")
            break
        runs = base if sat == 0.5 else [run_sieve(E2_BETA, 2000 + s, sat) for s in range(SEEDS)]
        tb = [lg(b["wall"] / r["wall"]) for b, r in zip(base, runs)]
        ob = [lg((b["xorpopcnt_total"] + b["fullscprods_total"]) /
                 (r["xorpopcnt_total"] + r["fullscprods_total"])) for b, r in zip(base, runs)]
        same = [abs(b["best_norm"] - r["best_norm"]) <= 1e-6 * max(1.0, abs(b["best_norm"]))
                if not (math.isnan(b["best_norm"]) or math.isnan(r["best_norm"])) else None
                for b, r in zip(base, runs)]
        known = [s for s in same if s is not None]
        frac = f"{sum(known)}/{len(known)}" if known else "n/a"
        if not known:
            print("  WARNING: best_lifts() returned nothing, so the 'same best vector' check cannot "
                  "validate this termination credit")
        sat_ok = f"{sum(1 for r in runs if r['saturated'])}/{len(runs)}"
        results["E2"][str(sat)] = {"time_bits": tb, "op_bits": ob, "same_best": same,
                                   "saturated_runs": sat_ok}
        if sat_ok.split("/")[0] != sat_ok.split("/")[1]:
            print(f"  NOTE sat={sat}: only {sat_ok} runs reached the requested saturation")
        print(f"  {sat:>5} {np.nanmean(tb):>+10.2f} {np.nanmean(ob):>+9.2f} {frac:>17}", flush=True)

    with open(OUT, "w") as f:
        json.dump(results, f, indent=1, default=str)
    print(f"\nWrote {OUT}. Bring this file back for the paper update.")


if __name__ == "__main__":
    main()
