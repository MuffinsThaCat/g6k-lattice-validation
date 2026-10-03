"""
Approximate-SVP (plateau) termination on the G6K-like bucketed sieve.

The paper's termination credit: stop the sieve once the best norm has plateaued, accepting a vector within a factor
alpha of the shortest, instead of running to saturation. Here the bucketed sieve of orbit_bucketed_sieve.py (calibrated
within 1.5x of real G6K bgj1 counters at n = 48) is run to saturation in short chunks while the database minimum is
recorded against cumulative operations (xor-popcounts + scalar products). For each alpha:
    ops_alpha = operations at which the minimum first falls within alpha x (the final minimum)
    gain      = log2(total ops / ops_alpha)   <- the most an ideal plateau detector could save (it must still
                                                 detect the plateau, which costs extra rounds)
An oracle stop at the right moment is an UPPER bound on any real plateau rule.

  python plateau_termination.py     Env: PT_CFGS="8:5,16:3" (ring d : rank) PT_SEEDS=3 PT_CHUNK=200
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import orbit_sieve_proto as O
import orbit_bucketed_sieve as B

CFGS = [tuple(int(x) for x in c.split(":")) for c in os.environ.get("PT_CFGS", "8:5,16:3").split(",")]
SEEDS = int(os.environ.get("PT_SEEDS", "3"))
CHUNK = int(os.environ.get("PT_CHUNK", "200"))
Q = int(os.environ.get("PT_Q", "257"))
ALPHAS = [1.02, 1.05, 1.10, 1.20]
OUT = os.environ.get("PT_OUT", "plateau_termination_results.json")


def one(d, k, seed):
    n = d * k
    Bn, P = O.make_lattice(d, k, Q, 50 + seed)
    rng = np.random.RandomState(seed)
    gh2 = O.gh_sq(Bn)
    R2 = 4.0 / 3.0 * gh2
    N = int(3.2 * (4 / 3) ** (n / 2))
    target = 0.5 * (4 / 3) ** (n / 2) / 2
    alpha_b = B.alpha_for(n, N)
    Hm = rng.normal(size=(n, B.BITS))
    V = O.sample_db(Bn, N, 3.0 * math.sqrt(gh2), rng)
    B.MAXROUNDS = CHUNK
    cnt = B.Counter()
    trace = []                                         # (cumulative ops, min norm)
    short = 0
    for _ in range(400):
        V, rounds, short = B.run_baseline(V, Hm, n, R2, target, alpha_b, rng, cnt)
        trace.append((cnt.pop + cnt.dots, float(np.sqrt(((V * V).sum(1)).min()))))
        if short >= target:
            break
    total = trace[-1][0]
    final = trace[-1][1]
    res = {"n": n, "d": d, "seed": seed, "total_ops": total, "final_min_over_gh": final / math.sqrt(gh2),
           "reached": bool(short >= target), "alphas": {}}
    for a in ALPHAS:
        ops_a = next(o for o, m in trace if m <= a * final)
        res["alphas"][a] = {"ops": ops_a, "frac": ops_a / total, "bits": math.log2(total / ops_a)}
    return res


def main():
    rows = []
    print("oracle plateau stop: ops when the best norm first reaches alpha x final, as a fraction of the ops to saturation")
    print(f"{'n':>3s} {'seed':>4s} {'reached':>7s} {'final/GH':>8s}  " + "  ".join(f"a={a:<4}   " for a in ALPHAS))
    for d, k in CFGS:
        for s in range(SEEDS):
            r = one(d, k, s)
            rows.append(r)
            cells = "  ".join(f"{r['alphas'][a]['frac']:.2f} ({r['alphas'][a]['bits']:+.2f}b)" for a in ALPHAS)
            print(f"{r['n']:3d} {s:4d} {str(r['reached']):>7s} {r['final_min_over_gh']:8.3f}  {cells}", flush=True)
    for a in ALPHAS:
        bits = [r["alphas"][a]["bits"] for r in rows if r["reached"]]
        print(f"alpha {a}: mean oracle gain {np.mean(bits):+.2f} bits (range {min(bits):+.2f} .. {max(bits):+.2f}, {len(bits)} runs)")
    with open(OUT, "w") as fh:
        json.dump(rows, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
