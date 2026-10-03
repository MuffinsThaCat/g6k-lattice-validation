"""
Applies PREREGISTERED_CRITERIA_R2.md item C' to res_tts_d*.json.  No judgement calls are made here by hand:
the variant is chosen from TUNING runs at d = 28 only, then scored on HOLDOUT runs.

  python rotation_tts_analyze.py [dir_with_res_tts_files]
"""
import glob
import json
import math
import os
import random
import sys

DIR = sys.argv[1] if len(sys.argv) > 1 else "."
TUNE_D = 28
ORDER = ["rot", "rotmid", "rotall"]          # tie-break: fewer clones first
COSTS = {"wall": "wall", "ops_dense": "ops_incl_clones_dense", "ops_opt": "ops_incl_clones_opt"}
TIMEOUT_CHARGE = 3000.0


def load():
    rows = []
    for p in sorted(glob.glob(os.path.join(DIR, "**", "res_tts_d*.json"), recursive=True)):
        rows += json.load(open(p))["rows"]
    return rows


def lref(rows):
    out = {}
    for r in rows:
        if r.get("timeout") or r.get("error") or "minlen2" not in r:
            continue
        out[r["d"]] = min(out.get(r["d"], float("inf")), r["minlen2"])
    return out


def success(r, L):
    return (not r.get("timeout")) and (not r.get("error")) and r["minlen2"] <= L[r["d"]] * (1 + 1e-9)


def cost(r, key):
    if r.get("timeout"):
        return TIMEOUT_CHARGE if key == "wall" else float("inf")   # ops of a killed run are unknown: infinite -> never helps
    if r.get("error"):
        return None
    return r[COSTS[key]]


def tts(runs, L, key):
    cs = [cost(r, key) for r in runs]
    if any(c is None for c in cs) or not runs:
        return None
    s = sum(1 for r in runs if success(r, L))
    if s == 0:
        return float("inf")
    return sum(cs) / s


def boot_gain(base, var, L, key, n=2000, seed=1):
    rng = random.Random(seed)
    g = []
    for _ in range(n):
        b = [rng.choice(base) for _ in base]
        v = [rng.choice(var) for _ in var]
        tb, tv = tts(b, L, key), tts(v, L, key)
        if tb is None or tv is None or math.isinf(tb) or math.isinf(tv):
            continue
        g.append(math.log2(tb / tv))
    g.sort()
    if len(g) < n // 2:
        return None, None
    return g[int(0.025 * len(g))], g[int(0.975 * len(g))]


def slope(xs, ys):
    if len(xs) < 2:
        return None
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    den = sum((x - mx) ** 2 for x in xs)
    return None if den == 0 else sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


def main():
    rows = load()
    if not rows:
        print("no results found")
        return
    L = lref(rows)
    errs = [r for r in rows if r.get("error")]
    tos = [r for r in rows if r.get("timeout")]
    print(f"{len(rows)} runs, {len(errs)} errors, {len(tos)} timeouts; dimensions: {sorted(L)}")
    for r in errs[:5]:
        print("  error:", r["mode"], r["d"], r["seed"], r["error"])

    tune = [r for r in rows if r["set"] == "tune" and r["d"] == TUNE_D]
    if not tune:
        print("no tuning runs at d =", TUNE_D, "-> variant cannot be chosen (unmeasured)")
        return
    print(f"\n== TUNING (d={TUNE_D}, seeds 100-107): wall time-to-solution by variant ==")
    best, best_t = None, float("inf")
    for m in ["base"] + ORDER:
        rs = [r for r in tune if r["mode"] == m]
        t = tts(rs, L, "wall")
        ns = sum(1 for r in rs if success(r, L))
        print(f"  {m:7} runs {len(rs)} successes {ns}  TTS_wall {t if t is None else round(t, 3)}")
        if m != "base" and t is not None and t < best_t:
            best, best_t = m, t
    if best is None:
        print("no variant has a finite tuning TTS: credit 0 (unmeasured)")
        return
    print(f"-> chosen variant (tuning only): {best}")

    print(f"\n== HOLDOUT: gross gain g(d) = log2(TTS_base / TTS_{best}), 95% bootstrap CI ==")
    print(f"{'d':>3} {'n':>3} {'runs b/v':>9} {'succ b/v':>9}  " + "  ".join(f"{k:>24}" for k in COSTS))
    table = {k: {} for k in COSTS}
    for d in sorted(L):
        b = [r for r in rows if r["set"] == "holdout" and r["d"] == d and r["mode"] == "base"]
        v = [r for r in rows if r["set"] == "holdout" and r["d"] == d and r["mode"] == best]
        if not b or not v:
            continue
        sb = sum(1 for r in b if success(r, L))
        sv = sum(1 for r in v if success(r, L))
        cells = []
        for k in COSTS:
            tb, tv = tts(b, L, k), tts(v, L, k)
            if tb is None or tv is None or math.isinf(tb) or math.isinf(tv):
                cells.append(f"{'n/a':>24}")
                continue
            g = math.log2(tb / tv)
            lo, hi = boot_gain(b, v, L, k)
            table[k][d] = (g, lo, hi)
            cells.append(f"{g:+6.2f} [{lo:+.2f},{hi:+.2f}]".rjust(24) if lo is not None else f"{g:+6.2f} [n/a]".rjust(24))
        print(f"{d:>3} {2 * d:>3} {len(b):>4}/{len(v):<4} {sb:>4}/{sv:<4}  " + "  ".join(cells))

    print("\n== PRE-REGISTERED READOUT ==")
    for k, label in (("wall", "wall-clock gain"), ("ops_dense", "gate-count gain (clone work as implemented, dense)"),
                     ("ops_opt", "POTENTIAL with a cyclic-shift kernel (model, never a credit)")):
        ok = {d: t for d, t in table[k].items() if t[1] is not None and t[1] > 0}
        if not ok:
            print(f"{label}: no dimension has a CI lower bound > 0 -> credit 0")
            continue
        dmax = max(ok)
        xs = [d for d in table[k] if d >= TUNE_D]
        sl = slope(xs, [table[k][d][0] for d in xs])
        capped = ""
        val = ok[dmax][0]
        if sl is not None and sl <= 0:
            val = ok.get(TUNE_D, ok[dmax])[0]
            capped = "  (slope <= 0: non-growing, capped at the d=28 value)"
        sls = "n/a" if sl is None else f"{sl:+.3f} bits/d"
        print(f"{label}: {val:+.2f} b at d={dmax if not capped else TUNE_D} (n={2 * (dmax if not capped else TUNE_D)}), "
              f"slope over d>=28: {sls}{capped}")
    gd = table["ops_dense"]
    ok = [t[0] for d, t in gd.items() if t[1] is not None and t[1] > 0]
    if not ok or max(ok) < 0.5:
        print("gate-count credit: < 0.5 b in the dense metric -> 0 (rule C)")
    print("GROSS only: the aligned-basis quality cost is not netted (see PREREGISTERED_CRITERIA_R2.md).")


if __name__ == "__main__":
    main()
