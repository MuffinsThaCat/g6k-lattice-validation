"""
Orbit-aware BUCKETED sieve (bgj1-style) vs the same sieve without symmetry, in G6K's own counting units.

Baseline  : bgj1-like rounds on N vectors. Round = pick a centre c; bucketing: xor-popcount of every vector's
            256-bit SimHash against c's (threshold XPC_BUCKET = 102, passes if distance <= 102 or >= 154),
            exact inner product for sketch-passers, bucket = {|<c,v>| >= alpha |c||v|}; inside the bucket all
            pairs: popcount (threshold 96) -> inner product -> reduction if shorter than the 0.65-quantile length.
            Replacement: new vectors displace the longest entries. Stop at G6K's saturation rule
            (0.5 * (4/3)^(n/2)/2 vectors with |v|^2 <= 4/3 GH^2).
Orbit     : the same algorithm on R = N/d representatives, each standing for its d rotations x^r v.
            Sketches are ROTATION-EQUIVARIANT: the hash set is {x^s h_i}, so sketch(x^r v) is the
            negacyclic bit-rotation of sketch(v) (checked numerically below). A pair class (a, x^r b) therefore
            costs one popcount like a baseline pair. A round's centre c stands for its d rotated centres, and a new
            representative inserts d vectors at once.
Reported: xorpopcnt tests + full scalar products to saturation (G6K's counters), rounds, and the
baseline's counts next to REAL G6K bgj1 counters for the same n (calibration).
Toy numpy sieve; counts only. Run inside the venv with fpylll/scipy.

  python orbit_bucketed_sieve.py      Env: OB_D=16 OB_K=3 OB_Q=257 OB_SEEDS=2 OB_MAXROUNDS=40000 OB_OUT=...json
"""
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.special import betaincinv

import orbit_sieve_proto as O

D = int(os.environ.get("OB_D", "16"))
K = int(os.environ.get("OB_K", "3"))
Q = int(os.environ.get("OB_Q", "257"))
SEEDS = int(os.environ.get("OB_SEEDS", "2"))
MAXROUNDS = int(os.environ.get("OB_MAXROUNDS", "40000"))
OUT = os.environ.get("OB_OUT", "orbit_bucketed_sieve_results.json")
T_SUB = int(os.environ.get("OB_T", "0"))   # subgroup order t (0 = the full ring rotation group of order d)
BITS = 256
T_BUCKET, T_INNER = 102, 96
QUANT = 0.65
POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def pack(bits):
    return np.packbits(bits.astype(np.uint8), axis=-1)


def ham(a, b):
    """Hamming distance between packed sketches a (.., W) and b (W,) or (.., W)."""
    return POP[np.bitwise_xor(a, b)].sum(axis=-1, dtype=np.int32)


def passes(h, t):
    return (h <= t) | (h >= BITS - t)


def alpha_for(n, N):
    B = 3.2 * math.sqrt(N)
    x = betaincinv((n + 1.0) / 2.0, 0.5, min(0.999, B / N))
    return math.sqrt(1.0 - x)


class Counter:
    def __init__(self):
        self.pop = 0
        self.dots = 0


def run_baseline(V, Hm, n, R2, target, alpha, rng, cnt):
    N = len(V)
    S = pack(V @ Hm >= 0)
    nm = (V * V).sum(1)
    seen = {O.canon_key(v) for v in V}
    rounds = 0
    while rounds < MAXROUNDS:
        short = int((nm <= R2).sum())
        if short >= target:
            return V, rounds, short
        rounds += 1
        maxlen = np.quantile(nm, QUANT)
        order = np.argsort(nm)
        ci = int(order[rng.randint(N // 4, N // 2)])
        c, nc = V[ci], nm[ci]
        h = ham(S, S[ci])
        cnt.pop += N - 1
        cand = np.flatnonzero(passes(h, T_BUCKET))
        cand = cand[cand != ci]
        new = []
        if len(cand):
            dots = V[cand] @ c
            cnt.dots += len(cand)
            sel = np.abs(dots) >= alpha * math.sqrt(nc) * np.sqrt(nm[cand])
            for j, dj in zip(cand, dots):
                if nc + nm[j] - 2 * abs(dj) < maxlen:
                    new.append(c - np.sign(dj) * V[j])
            bidx = cand[sel]
            M = len(bidx)
            if M > 1:
                SB = S[bidx]
                iu, ju = np.triu_indices(M, 1)
                hh = ham(SB[iu], SB[ju])
                cnt.pop += len(iu)
                ok = passes(hh, T_INNER)
                iu, ju = iu[ok], ju[ok]
                if len(iu):
                    A, Bv = V[bidx[iu]], V[bidx[ju]]
                    dd = (A * Bv).sum(1)
                    cnt.dots += len(iu)
                    nn = nm[bidx[iu]] + nm[bidx[ju]] - 2 * np.abs(dd)
                    good = nn < maxlen
                    for a_, b_, d_ in zip(iu[good], ju[good], dd[good]):
                        new.append(V[bidx[a_]] - np.sign(d_) * V[bidx[b_]])
        if new:
            new.sort(key=lambda u: float(u @ u))
            slots = list(np.argsort(-nm))
            si = 0
            for u in new:
                if si >= len(slots) // 3:
                    break
                un = float(u @ u)
                k = O.canon_key(u)
                if un < 1e-9 or k in seen:
                    continue
                t = slots[si]
                if un >= nm[t]:
                    break
                seen.discard(O.canon_key(V[t]))
                seen.add(k)
                V[t] = u
                nm[t] = un
                S[t] = pack((u @ Hm >= 0)[None, :])[0]
                si += 1
    return V, rounds, int((nm <= R2).sum())


def run_orbit(U, Hm, P, d, n, R2, target, alpha, rng, cnt):
    R = len(U)
    N = R * d
    nm = (U * U).sum(1)
    seen = {O.orbit_key(u, P, d) for u in U}
    E = O.rotations(U, P, d)                        # (d, R, n); E[r, b] = x^r u_b
    SE = pack(E.reshape(d * R, n) @ Hm >= 0).reshape(d, R, -1)
    rounds = 0

    def expanded_short():
        keys = set()
        for b in np.flatnonzero(nm <= R2):
            for r in range(d):
                keys.add(O.canon_key(E[r, b]))
        return len(keys)

    while rounds < MAXROUNDS:
        if rounds % 10 == 0 or rounds < 10:
            short = expanded_short()
            if short >= target:
                return U, rounds, short
        rounds += 1
        maxlen = np.quantile(nm, QUANT)
        order = np.argsort(nm)
        ci = int(order[rng.randint(R // 4, max(R // 4 + 1, R // 2))])
        c, nc = U[ci], nm[ci]
        sc = SE[0, ci]
        h = ham(SE, sc)                              # (d, R)
        cnt.pop += N - 1
        pr, pb = np.nonzero(passes(h, T_BUCKET))
        keep = ~((pr == 0) & (pb == ci))
        pr, pb = pr[keep], pb[keep]
        new = []
        if len(pr):
            Vc = E[pr, pb]
            dots = Vc @ c
            cnt.dots += len(pr)
            vn = nm[pb]
            for jr, jb, dj, vj in zip(pr, pb, dots, vn):
                if nc + vj - 2 * abs(dj) < maxlen:
                    new.append(c - np.sign(dj) * E[jr, jb])
            sel = np.abs(dots) >= alpha * math.sqrt(nc) * np.sqrt(vn)
            br, bb = pr[sel], pb[sel]
            M = len(br)
            if M > 1:
                SB = SE[br, bb]
                iu, ju = np.triu_indices(M, 1)
                hh = ham(SB[iu], SB[ju])
                cnt.pop += len(iu)
                ok = passes(hh, T_INNER)
                iu, ju = iu[ok], ju[ok]
                if len(iu):
                    A, Bv = E[br[iu], bb[iu]], E[br[ju], bb[ju]]
                    dd = (A * Bv).sum(1)
                    cnt.dots += len(iu)
                    nn = nm[bb[iu]] + nm[bb[ju]] - 2 * np.abs(dd)
                    good = nn < maxlen
                    for a_, b_, d_ in zip(iu[good], ju[good], dd[good]):
                        new.append(A[a_] - np.sign(d_) * Bv[b_] if False else
                                   E[br[a_], bb[a_]] - np.sign(d_) * E[br[b_], bb[b_]])
        if new:
            new.sort(key=lambda u: float(u @ u))
            slots = list(np.argsort(-nm))
            si = 0
            for u in new:
                if si >= len(slots) // 3:
                    break
                un = float(u @ u)
                if un < 1e-9:
                    continue
                k = O.orbit_key(u, P, d)
                if k in seen:
                    continue
                t = slots[si]
                if un >= nm[t]:
                    break
                seen.discard(O.orbit_key(U[t], P, d))
                seen.add(k)
                U[t] = u
                nm[t] = un
                cur = u
                for r in range(d):
                    E[r, t] = cur
                    cur = cur @ P
                SE[:, t] = pack(E[:, t] @ Hm >= 0)
                si += 1
    return U, rounds, expanded_short()


def check_equivariance(Bn, P, d, n, m, rng):
    """sketch(x^r v)[i,s] == (complement if wrapped) sketch(v)[i,(s-r) mod d], for hash set {x^s h_i}."""
    G = rng.normal(size=(m, n))
    Hm = np.zeros((n, m * d))
    for i in range(m):
        cur = G[i]
        for s in range(d):
            Hm[:, i * d + s] = cur
            cur = cur @ P
    v = rng.normal(size=n)
    sk = (v @ Hm >= 0).reshape(m, d)
    for r in range(1, d):
        vr = v.copy()
        for _ in range(r):
            vr = vr @ P
        got = (vr @ Hm >= 0).reshape(m, d)
        want = np.zeros_like(got)
        for s in range(d):
            src = s - r
            want[:, s] = sk[:, src % d] ^ (src < 0)
        assert (got == want).all(), f"equivariance fails at r={r}"
    return Hm


def main():
    ring_d, k = D, K
    n = k * ring_d
    d = T_SUB or ring_d                           # orbit length actually used (subgroup order t)
    assert ring_d % d == 0
    m = max(1, round(BITS / d))
    print(f"ring d={ring_d} k={k} n={n} q={Q}; subgroup order t={d}; orbit sketch = {m} base hashes x {d} rotations = {m*d} bits; baseline 256-bit SimHash")
    rows = []
    for seed in range(SEEDS):
        Bn, P = O.make_lattice(ring_d, k, Q, 30 + seed)
        if d != ring_d:
            P = np.linalg.matrix_power(P, ring_d // d)      # y = x^(d_ring/t): a rotation of order 2t (t on +- classes)
        rng = np.random.RandomState(seed)
        Hm_orbit = check_equivariance(Bn, P, d, n, m, rng)
        print("  equivariant sketch verified: sketch of a rotated vector is the negacyclic bit-rotation of the sketch")
        Hm_plain = rng.normal(size=(n, BITS))
        gh2 = O.gh_sq(Bn)
        R2 = 4.0 / 3.0 * gh2
        N = int(3.2 * (4 / 3) ** (n / 2))
        R = max(8, N // d)
        target = 0.5 * (4 / 3) ** (n / 2) / 2
        alpha = alpha_for(n, N)
        radius = 3.0 * math.sqrt(gh2)
        res = {"seed": seed, "N": N, "R": R, "target": target, "alpha": alpha}
        V = O.sample_db(Bn, N, radius, rng)
        c1 = Counter()
        t0 = time.time()
        V, rounds, short = run_baseline(V, Hm_plain, n, R2, target, alpha, rng, c1)
        res["baseline"] = {"pop": c1.pop, "dots": c1.dots, "rounds": rounds, "short": short, "secs": time.time() - t0}
        U = O.sample_db(Bn, R, radius, rng)
        c2 = Counter()
        t0 = time.time()
        U, rounds2, short2 = run_orbit(U, Hm_orbit, P, d, n, R2, target, alpha_for(n, N), rng, c2)
        res["orbit"] = {"pop": c2.pop, "dots": c2.dots, "rounds": rounds2, "short": short2, "secs": time.time() - t0}
        rows.append(res)
        for name in ("baseline", "orbit"):
            r = res[name]
            ok = "reached" if r["short"] >= target else "NOT reached"
            print(f"  seed {seed} {name:8s}: popcounts {r['pop']:.3g}, scalar products {r['dots']:.3g}, rounds {r['rounds']}, "
                  f"short {r['short']} (target {target:.0f}, {ok}), {r['secs']:.0f}s", flush=True)
        b, o = res["baseline"], res["orbit"]
        if b["short"] >= target and o["short"] >= target:
            tb, to = b["pop"] + b["dots"], o["pop"] + o["dots"]
            print(f"  ops ratio baseline/orbit = {tb/to:.1f}  ({math.log2(tb/to):+.2f} bits); popcounts only {b['pop']/o['pop']:.1f}x; "
                  f"memory (full vectors) {N/R:.0f}x")
    with open(OUT, "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
