"""
How many bits does in-sieve rotation augmentation save, and how does that scale with dimension?

Attack-bit gain at MATCHED OUTCOME: log2(cost_base / cost_variant), counting only variants that found
(almost) all rotation orbits of short vectors, i.e. the same information as the unmodified sieve.
Cost = wall time (includes the cloning overhead) and, separately, counted operations (excludes it).
Base: unmodified bgj1 at saturation 0.5 (it finds every orbit). Variants: rotate with clone threshold
BELOW in VARIANTS at saturation SAT_ROT.  Fits gain(bits) against n = 2d.

  python rotation_scaling.py        (cwd = patched g6k checkout, or G6K_DIR)
Env: SC_DS="16,20,24,28,32" SC_Q=257 SC_SEEDS=3 SC_SAT_ROT=1.0 SC_BELOW="1.3333,1.6,2.0,1e9" SC_SATS_ROT="1.0,2.0" SC_OUT=...json
"""
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import rotation_sieve_ab as AB
import rotation_orbit_check as R

DS = [int(x) for x in os.environ.get("SC_DS", "16,20,24,28,32").split(",")]
Q = int(os.environ.get("SC_Q", "257"))
SEEDS = int(os.environ.get("SC_SEEDS", "3"))
SATS_ROT = [float(x) for x in os.environ.get("SC_SATS_ROT", "1.0,2.0").split(",")]
BELOWS = [float(x) for x in os.environ.get("SC_BELOW", "1.3333,1.6,2.0,1e9").split(",")]
OUT = os.environ.get("SC_OUT", "rotation_scaling_results.json")
COMPLETE = 0.90   # a variant is "matched" if it finds >= 95% of the orbits the base sieve finds


def mean(xs):
    return float(np.mean(xs))


def main():
    results = []
    print(f"q={Q} seeds={SEEDS}; variant saturations={SATS_ROT}; matched = >= {COMPLETE:.0%} of base orbits")
    for d in DS:
        n = 2 * d
        B0, P = R.ideal_basis(d, Q, 1)
        t0 = time.time()
        base = [AB.run_one(B0, P, 3.2, "base", 100 + s, sat=0.5) for s in range(SEEDS)]
        b_orb, b_wall, b_ops = mean([r["orbits"] for r in base]), mean([r["wall"] for r in base]), mean([r["ops"] for r in base])
        row = {"d": d, "n": n, "base": {"orbits": b_orb, "wall": b_wall, "ops": b_ops,
                                        "ok": sum(r["ok"] for r in base)}, "variants": {}}
        print(f"\nd={d} n={n}: base orbits {b_orb:.1f}, wall {b_wall:.3f}s, ops {b_ops:.3g} ({time.time()-t0:.0f}s so far)", flush=True)
        for below, sat_rot in [(b, s_) for b in BELOWS for s_ in SATS_ROT]:
            rs = [AB.run_one(B0, P, 3.2, "rot", 100 + s, sat=sat_rot, below=below) for s in range(SEEDS)]
            orb, wall, ops = mean([r["orbits"] for r in rs]), mean([r["wall"] for r in rs]), mean([r["ops"] for r in rs])
            matched = orb >= COMPLETE * b_orb
            row["variants"][f"{below:g}@{sat_rot:g}"] = {"orbits": orb, "wall": wall, "ops": ops, "matched": matched,
                                           "clones": mean([r["clones"] for r in rs]),
                                           "wall_bits": math.log2(b_wall / wall), "ops_bits": math.log2(b_ops / ops)}
            print(f"  below={below:<8g} sat={sat_rot:<4g} orbits {orb:5.1f}/{b_orb:.1f} {'MATCHED' if matched else 'partial'}  "
                  f"wall {wall:.3f}s ({math.log2(b_wall/wall):+.2f} bits)  "
                  f"ops {ops:.3g} ({math.log2(b_ops/ops):+.2f} bits)", flush=True)
        results.append(row)
        with open(OUT, "w") as fh:
            json.dump(results, fh, indent=1)
    # scaling fit of the best MATCHED variant per d
    ns, wb, ob = [], [], []
    print("\nBest matched variant per dimension (max wall bits):")
    for row in results:
        m = {k: v for k, v in row["variants"].items() if v["matched"]}
        if not m:
            print(f"  n={row['n']}: no variant matched the base's orbit coverage")
            continue
        k = max(m, key=lambda kk: m[kk]["wall_bits"])
        print(f"  n={row['n']}: below={k}  wall {m[k]['wall_bits']:+.2f} bits   ops {m[k]['ops_bits']:+.2f} bits")
        ns.append(row["n"]); wb.append(m[k]["wall_bits"]); ob.append(m[k]["ops_bits"])
    if len(ns) >= 3:
        sw, cw = np.polyfit(ns, wb, 1)
        so, co = np.polyfit(ns, ob, 1)
        print(f"fit: wall bits ~ {sw:+.4f}*n {cw:+.2f}   |   ops bits ~ {so:+.4f}*n {co:+.2f}")
        print("A slope >> 0 would mean the gain grows with dimension (an asymptotic effect); ~0 means a constant factor.")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
