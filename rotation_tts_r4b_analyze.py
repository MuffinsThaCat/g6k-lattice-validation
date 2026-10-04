"""Applies PREREGISTERED_CRITERIA_R4B.md.  python rotation_tts_r4b_analyze.py R4B_DIR
Same quantities and rules as rotation_tts_r4_analyze.py; the only change is that each g is compared with the
baseline runs of its OWN job (its own res_tts_d32_lat1_g<g>.json, same runner), not with a pooled Round 2 baseline."""
import glob, json, math, os, random, re, sys
D, LAT = 32, 1
jobs = {}
for p in sorted(glob.glob(os.path.join(sys.argv[1], "**", "res_tts_d32_lat1_g*.json"), recursive=True)):
    g = int(re.search(r"_g(\d+)\.json$", p).group(1))
    assert g not in jobs, f"two result files for g={g}"
    jobs[g] = [dict(r, lat=r.get("lat", 1)) for r in json.load(open(p))["rows"]]
rows = [r for v in jobs.values() for r in v]
Lr = min(r["minlen2"] for r in rows if r["d"] == D and r["lat"] == LAT and "minlen2" in r)
ok = lambda r: "minlen2" in r and r["minlen2"] <= Lr * (1 + 1e-9)
def tts(rs):
    s = sum(ok(r) for r in rs)
    return float("inf") if s == 0 else sum(3000.0 if r.get("timeout") else r["wall"] for r in rs) / s
def sel(rs, mode):
    return [r for r in rs if r["d"] == D and r["lat"] == LAT and r["set"] == "holdout" and r["mode"] == mode
            and 200 <= r["seed"] <= 215]
def boot(b, v, n=2000):
    rng = random.Random(1); g = []
    for _ in range(n):
        tb, tv = tts([rng.choice(b) for _ in b]), tts([rng.choice(v) for _ in v])
        if math.isfinite(tb) and math.isfinite(tv): g.append(math.log2(tb / tv))
    g.sort(); return (g[int(.025 * len(g))], g[int(.975 * len(g))]) if len(g) > n // 2 else (None, None)
print(f"d={D} (n={2*D}) lattice {LAT}; jobs found: {sorted(jobs)}")
print(f"{'g':>3} {'base':>4} {'TTSb':>7} {'rot':>4} {'succ':>4}  {'gain G [95% CI]':<22} {'ceil':>5} {'eps':>5} {'A(g)':>5} {'N=G-A':>6}")
best, res = None, {}
for g in (2, 4, 8, 16, 32, 64):
    if g not in jobs: print(f"{g:>3}  no job"); continue
    b, v = sel(jobs[g], "base"), sel(jobs[g], "rotall")
    tb, t = tts(b), tts(v)
    if not (b and v and math.isfinite(tb) and math.isfinite(t)):
        print(f"{g:>3} {len(b):>4} {tb:7.1f} {len(v):>4}  n/a"); continue
    G = math.log2(tb / t); lo, hi = boot(b, v); res[g] = (G, lo)
    ceil = math.log2(g / 2) if g > 2 else 0.0
    eps = G / ceil if ceil > 0 else float("nan"); A = 0.146 * (g / 2)
    print(f"{g:>3} {len(b):>4} {tb:7.1f} {len(v):>4} {sum(ok(r) for r in v):>4}  {G:+6.2f} [{lo:+.2f},{hi:+.2f}]".ljust(48)
          + f" {ceil:5.2f} {eps:5.2f} {A:5.2f} {G - A:+6.2f}")
    if lo is not None and lo > 0 and g > 2 and (best is None or G - A > best[1]): best = (g, G - A, G)
null_failed = 2 in res and res[2][1] is not None and res[2][1] > 0
print(f"\nNULL CONTROL g=2: " + ("no result" if 2 not in res else
      f"G={res[2][0]:+.2f}, CI lower bound {res[2][1]}: " + ("FAILED (CI lower bound > 0): do not trust this round" if null_failed else "passed")))
if null_failed:
    print("\nREADOUT: none (null control failed)")
else:
    print("\nREADOUT: " + (f"g*={best[0]}, net before prefix-quality cost N={best[1]:+.2f} b (gross {best[2]:+.2f}); toy scale n=64, not extrapolated"
          if best and best[1] > 0 else "max N(g) <= 0 (or no g with CI lower bound > 0): rotation credit under d4f = 0"))
