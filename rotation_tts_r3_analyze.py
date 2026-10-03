"""
Applies PREREGISTERED_CRITERIA_R3.md (R3a, R3b, R3c) to res_tts_d*.json found under the given directories
(pass the Round-2 download dir and the Round-3 download dir).   python rotation_tts_r3_analyze.py DIR [DIR ...]
Variant fixed to rotall; nothing is re-tuned.
"""
import glob, json, math, os, random, statistics as st, sys

TO = {"wall": 3000.0}
def load(dirs):
    rows = []
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "**", "res_tts_d*.json"), recursive=True)):
            for r in json.load(open(p))["rows"]:
                r.setdefault("lat", 1)
                rows.append(r)
    return rows

def lref(rows):
    L = {}
    for r in rows:
        if "minlen2" in r:
            k = (r["d"], r["lat"]); L[k] = min(L.get(k, 9e99), r["minlen2"])
    return L

def ok(r, L): return "minlen2" in r and r["minlen2"] <= L[(r["d"], r["lat"])] * (1 + 1e-9)
def cost(r, key, charge):
    if r.get("timeout"): return charge if key == "wall" else float("inf")
    return r["wall"] if key == "wall" else r["ops_incl_clones_dense"]
def tts(rs, L, key, charge):
    s = sum(ok(r, L) for r in rs)
    return float("inf") if s == 0 else sum(cost(r, key, charge) for r in rs) / s
def boot(b, v, L, key, charge, n=2000):
    rng = random.Random(1); g = []
    for _ in range(n):
        tb = tts([rng.choice(b) for _ in b], L, key, charge); tv = tts([rng.choice(v) for _ in v], L, key, charge)
        if math.isfinite(tb) and math.isfinite(tv): g.append(math.log2(tb / tv))
    g.sort()
    return (g[int(.025 * len(g))], g[int(.975 * len(g))]) if len(g) > n // 2 else (None, None)

def sel(rows, **kw):
    return [r for r in rows if r["set"] == "holdout" and all(r.get(k) == v for k, v in kw.items())]

def main():
    rows = load(sys.argv[1:])
    L = lref(rows)
    print(f"{len(rows)} runs, {sum(1 for r in rows if r.get('timeout'))} timeouts, {sum(1 for r in rows if r.get('error'))} errors")

    print("\n== R3a: d=40, lattice 1 ==")
    b = sel(rows, d=40, lat=1, mode="base"); b = [r for r in b if r["seed"] <= 203]
    v = sel(rows, d=40, lat=1, mode="rotall")
    nb = sum(ok(r, L) for r in b)
    print(f"base runs {len(b)} successes {nb} timeouts {sum(1 for r in b if r.get('timeout'))}; rotall runs {len(v)} successes {sum(ok(r, L) for r in v)}")
    rw = st.median([r["wall"] for r in v if "minlen2" in r]) if v else None
    if nb >= 2:
        charge = 17000.0
        g = math.log2(tts(b, L, "wall", charge) / tts(v, L, "wall", charge)); lo, hi = boot(b, v, L, "wall", charge)
        print(f"MEASURED gain(d=40) wall = {g:+.2f} b  CI [{lo:+.2f},{hi:+.2f}]" if lo is not None else f"MEASURED gain(d=40) wall = {g:+.2f} b")
    else:
        print(f"< 2 base successes: lower bound only, >= {math.log2(17000 / rw):+.2f} b (not a credit)" if rw else "no data")

    print("\n== R3b: rotall wall-clock vs d (lattice 1, median over successes) and base trend ==")
    rt = {}
    for d in (24, 28, 32, 36, 40, 44, 48):
        w = [r["wall"] for r in sel(rows, d=d, lat=1, mode="rotall") if ok(r, L)]
        if w: rt[d] = st.median(w)
    bt = {}
    for d in (24, 28, 32, 36, 40):
        w = [r["wall"] for r in sel(rows, d=d, lat=1, mode="base") if ok(r, L)]
        if w: bt[d] = st.median(w)
    prev = None
    for d in sorted(rt):
        sl = "" if prev is None else f"  slope {(math.log2(rt[d]) - math.log2(rt[prev])) / (d - prev):+.3f} log2(s)/d since d={prev}"
        print(f"  rotall d={d}: {rt[d]:9.1f} s{sl}"); prev = d
    xs = [d for d in bt if d <= 36] if 40 not in bt else list(bt)
    if len(xs) >= 3:
        mx = sum(xs) / len(xs); my = sum(math.log2(bt[x]) for x in xs) / len(xs)
        b1 = sum((x - mx) * (math.log2(bt[x]) - my) for x in xs) / sum((x - mx) ** 2 for x in xs)
        print(f"  base trend fitted on d={min(xs)}..{max(xs)}: {b1:+.3f} log2(s)/d  (extrapolation beyond that is NOT a measurement)")
        for d in (44, 48):
            if d in rt:
                ext = 2 ** (my + b1 * (d - mx))
                print(f"    extrapolated base wall d={d}: {ext:,.0f} s -> implied gain {math.log2(ext / rt[d]):+.2f} b (EXTRAPOLATED)")

    print("\n== R3c: lattice-to-lattice variance (seeds 200-207, rotall vs base) ==")
    print(f"{'d':>3} {'lat':>3} {'succ b/v':>9}   gain wall [CI]            gain ops_dense [CI]")
    per = {}
    for d in (28, 32, 36):
        for lat in (1, 2, 3, 4):
            b = [r for r in sel(rows, d=d, lat=lat, mode="base") if r["seed"] <= 207]
            v = [r for r in sel(rows, d=d, lat=lat, mode="rotall") if r["seed"] <= 207]
            if not b or not v: continue
            cells = []
            for key in ("wall", "ops"):
                tb, tv = tts(b, L, key, 3000.0), tts(v, L, key, 3000.0)
                if not (math.isfinite(tb) and math.isfinite(tv)): cells.append("n/a"); continue
                g = math.log2(tb / tv); lo, hi = boot(b, v, L, key, 3000.0)
                cells.append(f"{g:+.2f} [{lo:+.2f},{hi:+.2f}]" if lo is not None else f"{g:+.2f}")
                if key == "wall": per.setdefault(d, []).append((lat, g, lo))
            print(f"{d:>3} {lat:>3} {sum(ok(r, L) for r in b):>4}/{sum(ok(r, L) for r in v):<4}   {cells[0]:<24}  {cells[1]}")
    for d, xs in sorted(per.items()):
        gs = [g for _, g, _ in xs]
        robust = all(lo is not None and lo > 0 for _, _, lo in xs)
        fails = [lat for lat, _, lo in xs if lo is None or lo <= 0]
        print(f"d={d}: lattices {len(xs)}, wall gain mean {st.mean(gs):+.2f} sd {st.pstdev(gs):.2f}; "
              f"{'robust on every lattice' if robust else 'NOT robust: CI lower bound <= 0 on lattice(s) ' + str(fails)}")
    print("GROSS only (aligned-basis quality cost not netted).")

if __name__ == "__main__":
    main()
