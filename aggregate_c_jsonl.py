"""
Tally a c-measurement JSONL (from either shard runner): cumulative success curves of
E (real embedded baseline), T1 (one view), Tany (all rotations), beta50 of each, and
c_eff = c_hat(E) / c_hat(T1 control) with a bootstrap 95% CI.

  python aggregate_c_jsonl.py <file.jsonl>
"""
import json
import math
import sys

import numpy as np
from scipy.optimize import minimize_scalar


# Same definitions as measure_baseline_chances_c.py, copied so this script needs only numpy + scipy
# (that module imports fpylll).
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
            F = r["per_beta"][bt]["F"]
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
    recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
    for r in recs:
        r["per_beta"] = {int(b): v for b, v in r["per_beta"].items()}
    betas = sorted(set.union(*[set(r["per_beta"]) for r in recs]))
    # Early-stopped target-free arms (g6k_measure_c.py, G6K_EARLY_STOP) end only after view 0 has hit
    # and after E's success block size, so past their last probed beta: cumulative T1 = Tany = 1, and
    # F is never read by the c fit (it stops at E's success). Fill those betas accordingly.
    for r in recs:
        last = max(r["per_beta"])
        for bt in betas:
            if bt not in r["per_beta"]:
                if bt < last:
                    raise SystemExit(f"seed {r['seed']}: gap at beta {bt} below last probed {last}")
                assert any(v["t1"] for v in r["per_beta"].values()), r["seed"]
                r["per_beta"][bt] = {"t1": True, "tany": True, "F": 1.0, "nviews": -1}
    E = np.array([[r["E_beta"] is not None and r["E_beta"] <= bt for bt in betas] for r in recs]).mean(0)
    T1 = np.array([cum([r["per_beta"][bt]["t1"] for bt in betas]) for r in recs]).mean(0)
    TA = np.array([cum([r["per_beta"][bt]["tany"] for bt in betas]) for r in recs]).mean(0)
    print(f"{sys.argv[1]}: {len(recs)} instances, d={recs[0].get('d')}")
    print("beta  E(real)  T1(1 view)  Tany(all rot)")
    for i, bt in enumerate(betas):
        if 0 < max(E[i], T1[i], TA[i]) and (i == 0 or min(E[i - 1], T1[i - 1]) < 1):
            print(f"{bt:4d}  {E[i]:.2f}     {T1[i]:.2f}        {TA[i]:.2f}")
    print(f"beta50: E={beta50(E, betas):.1f}  T1={beta50(T1, betas):.1f}  Tany={beta50(TA, betas):.1f}")
    print(f"E found per instance: {sorted(r['E_beta'] for r in recs if r['E_beta'] is not None)}"
          f"  not found: {sum(r['E_beta'] is None for r in recs)}")
    if len(recs) >= 4:
        def ctl(rs):
            return [dict(r, E_beta=next((bt for bt in betas if r["per_beta"][bt]["t1"]), None)) for r in rs]
        ce, c1 = fit_c(recs, betas), fit_c(ctl(recs), betas)
        rng = np.random.default_rng(1)
        ratios = []
        for _ in range(200):
            sub = [recs[j] for j in rng.integers(0, len(recs), len(recs))]
            ratios.append(fit_c(sub, betas) / fit_c(ctl(sub), betas))
        lo, hi = np.percentile(ratios, [2.5, 97.5])
        print(f"c_hat E={ce:.2f}  control T1={c1:.2f}  c_eff={ce / c1:.2f}  95% CI [{lo:.2f}, {hi:.2f}]")


if __name__ == "__main__":
    main()
