"""Applies PREREGISTERED_CRITERIA_R4.md.  python rotation_tts_r4_analyze.py R2_DIR R4_DIR"""
import glob, json, math, os, random, sys
D, LAT = 32, 1
def load(dirs):
    rows = []
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "**", "res_tts_d*.json"), recursive=True)):
            for r in json.load(open(p))["rows"]:
                r.setdefault("lat", 1); r.setdefault("group", 0); rows.append(r)
    return rows
rows = load(sys.argv[1:])
Lr = min(r["minlen2"] for r in rows if r["d"] == D and r["lat"] == LAT and "minlen2" in r)
ok = lambda r: "minlen2" in r and r["minlen2"] <= Lr * (1 + 1e-9)
def tts(rs):
    s = sum(ok(r) for r in rs)
    return float("inf") if s == 0 else sum(3000.0 if r.get("timeout") else r["wall"] for r in rs) / s
def sel(mode, g):
    return [r for r in rows if r["d"] == D and r["lat"] == LAT and r["set"] == "holdout" and r["mode"] == mode
            and r["group"] == g and 200 <= r["seed"] <= 215]
def boot(b, v, n=2000):
    rng = random.Random(1); g = []
    for _ in range(n):
        tb, tv = tts([rng.choice(b) for _ in b]), tts([rng.choice(v) for _ in v])
        if math.isfinite(tb) and math.isfinite(tv): g.append(math.log2(tb / tv))
    g.sort(); return (g[int(.025 * len(g))], g[int(.975 * len(g))]) if len(g) > n // 2 else (None, None)
base = sel("base", 0)
print(f"d={D} (n={2*D}) lattice {LAT}; base runs {len(base)}, successes {sum(ok(r) for r in base)}, TTS {tts(base):.1f}s")
print(f"{'g':>3} {'runs':>4} {'succ':>4}  {'gain G [95% CI]':<22} {'ceil':>5} {'eps':>5} {'A(g)':>5} {'N=G-A':>6}")
best = None
for g in (2, 4, 8, 16, 32, 64):
    v = sel("rotall", 0 if g == 64 else g)
    if not v: print(f"{g:>3}  no runs"); continue
    t = tts(v)
    if not math.isfinite(t): print(f"{g:>3} {len(v):>4}    0  n/a"); continue
    G = math.log2(tts(base) / t); lo, hi = boot(base, v)
    ceil = math.log2(g / 2) if g > 2 else 0.0
    eps = G / ceil if ceil > 0 else float("nan"); A = 0.146 * (g / 2)
    print(f"{g:>3} {len(v):>4} {sum(ok(r) for r in v):>4}  {G:+6.2f} [{lo:+.2f},{hi:+.2f}]".ljust(34) + f" {ceil:5.2f} {eps:5.2f} {A:5.2f} {G - A:+6.2f}")
    if lo is not None and lo > 0 and g > 2 and (best is None or G - A > best[1]): best = (g, G - A, G)
c2 = sel("rotall", 2)
if c2:
    G2 = math.log2(tts(base) / tts(c2)) if math.isfinite(tts(c2)) else float("nan"); lo2, _ = boot(base, c2)
    print(f"\nNULL CONTROL g=2: G={G2:+.2f}, CI lower bound {lo2}: " + ("FAILED (CI lower bound > 0): do not trust this round" if lo2 and lo2 > 0 else "passed"))
print("\nREADOUT: " + (f"g*={best[0]}, net before prefix-quality cost N={best[1]:+.2f} b (gross {best[2]:+.2f}); toy scale n=64, not extrapolated"
      if best and best[1] > 0 else "max N(g) <= 0 (or no g with CI lower bound > 0): rotation credit under d4f = 0"))
