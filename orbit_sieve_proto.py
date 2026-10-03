"""
Symmetric (orbit) sieve prototype: does a database of N/d rotation-orbit REPRESENTATIVES converge like a
plain database of N vectors, while testing d times fewer pair classes?

Both sieves use the same simple pass rule (the best reduction for every vector, replace the longer one):
  plain : DB of N vectors, all N(N-1)/2 pairs tested per pass.
  orbit : DB of R = N/d representatives; each represents its d rotations x^r v (negacyclic block shift).
          A rep pair (a, b) is tested through the d inner products <u_a, x^r u_b>, r = 0..d-1, i.e. the
          cyclic cross-correlation (an FFT in a real implementation; here computed directly, only the
          NUMBER of pair classes is counted, not the arithmetic).
Pair class = one inner product that decides a reduction; the orbit sieve tests ~ N^2/(2d) of them.
Also run: plain with N = R vectors (equal memory to the orbit sieve) to show it does not converge.

Metrics at convergence: shortest vector, short vectors (|v|^2 <= 4/3 GH^2, rotations expanded), pair
classes tested, passes.  Numbers are OPERATION COUNTS of a toy numpy sieve, not G6K timings, and a
sieve with real bucketing/filters would show different constants. Run inside the venv with fpylll.

  python orbit_sieve_proto.py        Env: OS_D=16 OS_K=3 OS_Q=257 OS_SEEDS=2 OS_PASSES=400 OS_OUT=...json
"""
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from fpylll import IntegerMatrix, LLL, BKZ

import rotation_block_test as T

D = int(os.environ.get("OS_D", "16"))
K = int(os.environ.get("OS_K", "3"))
Q = int(os.environ.get("OS_Q", "257"))
SEEDS = int(os.environ.get("OS_SEEDS", "2"))
PASSES = int(os.environ.get("OS_PASSES", "400"))
OUT = os.environ.get("OS_OUT", "orbit_sieve_proto_results.json")


def make_lattice(d, k, q, seed):
    B0, P = T.module_basis(d, q, seed, k)
    A = IntegerMatrix.from_matrix(B0.tolist())
    LLL.reduction(A)
    BKZ.reduction(A, BKZ.Param(block_size=min(20, k * d)))
    n = k * d
    Bn = np.array([[A[i, j] for j in range(n)] for i in range(n)], dtype=np.float64)
    return Bn, P.astype(np.float64)


def gh_sq(Bn):
    n = Bn.shape[0]
    logdet = np.linalg.slogdet(Bn)[1]
    return math.exp(2.0 * math.lgamma(n / 2.0 + 1.0) / n) / math.pi * math.exp(2.0 * logdet / n)


def sample_db(Bn, count, radius, rng):
    n = Bn.shape[0]
    Binv = np.linalg.inv(Bn)
    out, seen = [], set()
    while len(out) < count:
        t = rng.normal(0, radius / math.sqrt(n), size=n)
        c = np.rint(t @ Binv)
        v = c @ Bn
        key = canon_key(v)
        if v.any() and key not in seen:
            seen.add(key)
            out.append(v)
    return np.array(out)


def canon_key(v):
    iv = np.rint(v).astype(np.int64)
    nz = np.flatnonzero(iv)
    if len(nz) and iv[nz[0]] < 0:
        iv = -iv
    return iv.tobytes()


def rotations(U, P, d):
    """All d rotations of every row: array (d, R, n); rot[r] = U @ P^r."""
    out = [U]
    for _ in range(d - 1):
        out.append(out[-1] @ P)
    return np.array(out)


def orbit_key(v, P, d):
    best = None
    cur = v
    for _ in range(d):
        k = canon_key(cur)
        if best is None or k < best:
            best = k
        cur = cur @ P
    return best


def plain_sieve(V, rng):
    N = len(V)
    seen = {canon_key(v) for v in V}
    tests = 0
    for p in range(PASSES):
        nm = (V * V).sum(1)
        G = V @ V.T
        np.fill_diagonal(G, 0.0)
        tests += N * (N - 1) // 2
        red = nm[:, None] + nm[None, :] - 2.0 * np.abs(G)
        mx = np.maximum(nm[:, None], nm[None, :])
        imp = mx - red
        np.fill_diagonal(imp, -1.0)
        jb = imp.argmax(1)
        ib = imp[np.arange(N), jb]
        replaced = np.zeros(N, dtype=bool)
        changed = 0
        for i in np.argsort(-ib):
            if ib[i] <= 1e-6 * nm[i]:
                break
            j = int(jb[i])
            tgt = i if nm[i] >= nm[j] else j
            if replaced[tgt]:
                continue
            u = V[i] - np.sign(G[i, j]) * V[j]
            if not u.any():
                continue
            k = canon_key(u)
            if k in seen:
                continue
            seen.discard(canon_key(V[tgt]))
            seen.add(k)
            V[tgt] = u
            replaced[tgt] = True
            changed += 1
        if changed == 0:
            return V, tests, p + 1
    return V, tests, PASSES


def orbit_sieve(U, P, d, rng):
    R = len(U)
    seen = {orbit_key(u, P, d) for u in U}
    classes = 0
    for p in range(PASSES):
        nm = (U * U).sum(1)
        rot = rotations(U, P, d)                      # (d, R, n)
        Urot = rot.reshape(d * R, -1)                 # index r*R + b
        C = U @ Urot.T                                # (R, d*R): <u_a, x^r u_b>
        classes += R * (R - 1) // 2 * d + R * d // 2
        nm_b = np.tile(nm, d)                         # norm of x^r u_b = norm of u_b
        red = nm[:, None] + nm_b[None, :] - 2.0 * np.abs(C)
        mx = np.maximum(nm[:, None], nm_b[None, :])
        imp = mx - red
        # forbid a == b with r == 0 (the vector with itself)
        for a in range(R):
            imp[a, a] = -1.0
        jb = imp.argmax(1)
        ib = imp[np.arange(R), jb]
        replaced = np.zeros(R, dtype=bool)
        changed = 0
        for a in np.argsort(-ib):
            if ib[a] <= 1e-6 * nm[a]:
                break
            col = int(jb[a])
            r, b = divmod(col, R)
            tgt = a if nm[a] >= nm[b] else b
            if replaced[tgt]:
                continue
            u = U[a] - np.sign(C[a, col]) * rot[r, b]
            if not u.any():
                continue
            k = orbit_key(u, P, d)
            if k in seen:
                continue
            seen.discard(orbit_key(U[tgt], P, d))
            seen.add(k)
            U[tgt] = u
            replaced[tgt] = True
            changed += 1
        if changed == 0:
            return U, classes, p + 1
    return U, classes, PASSES


def summarize(V, P, d, R2, expand):
    if expand:
        V = rotations(V, P, d).reshape(-1, V.shape[1])
    keys = {}
    for v in V:
        keys[canon_key(v)] = v
    W = np.array(list(keys.values()))
    nm = (W * W).sum(1)
    return {"min_norm": float(math.sqrt(nm.min())), "short": int((nm <= R2).sum()), "distinct": len(W)}


def main():
    d, k = D, K
    n = k * d
    rows = []
    print(f"d={d} k={k} n={n} q={Q}; expected ~{(4/3)**(n/2)/2:.0f} short +- pairs, default DB N ~ {3.2*(4/3)**(n/2):.0f}")
    for seed in range(SEEDS):
        Bn, P = make_lattice(d, k, Q, 20 + seed)
        gh2 = gh_sq(Bn)
        R2 = (4.0 / 3.0) * gh2
        N = int(3.2 * (4 / 3) ** (n / 2))
        R = max(8, N // d)
        rng = np.random.RandomState(seed)
        radius = 3.0 * math.sqrt(gh2)
        res = {"seed": seed, "N": N, "R": R, "lam_gh": math.sqrt(gh2)}
        skip_plain = os.environ.get("OS_SKIP_PLAIN") == "1"   # n too large for the O(N^2) plain baseline
        for name, cnt in ((() if skip_plain else (("plain", N), ("plain-equal-memory", R)))):
            V = sample_db(Bn, cnt, radius, rng)
            t0 = time.time()
            V, tests, passes = plain_sieve(V, rng)
            s = summarize(V, P, d, R2, expand=False)
            s.update({"pair_tests": tests, "passes": passes, "secs": time.time() - t0, "stored": cnt})
            res[name] = s
        U = sample_db(Bn, R, radius, rng)
        t0 = time.time()
        U, classes, passes = orbit_sieve(U, P, d, rng)
        s = summarize(U, P, d, R2, expand=True)
        s.update({"pair_tests": classes, "passes": passes, "secs": time.time() - t0, "stored": R})
        res["orbit"] = s
        rows.append(res)
        print(f"\nseed {seed}: GH {math.sqrt(gh2):.1f}")
        for name in (("orbit",) if skip_plain else ("plain", "plain-equal-memory", "orbit")):
            s = res[name]
            print(f"  {name:19s} stored {s['stored']:5d}  min |v| {s['min_norm']:7.1f} ({s['min_norm']/math.sqrt(gh2):.3f} GH)  "
                  f"short {s['short']:5d}  pair tests {s['pair_tests']:.3g}  passes {s['passes']:3d}  {s['secs']:.1f}s", flush=True)
        if skip_plain:
            est = N * (N - 1) // 2 * res["orbit"]["passes"]
            print(f"  (plain baseline skipped; one plain pass would test {N*(N-1)//2:.3g} pairs, so ~{est:.3g} over the same "
                  f"number of passes: ratio ~{est/res['orbit']['pair_tests']:.0f}, expected short pairs ~{(4/3)**(n/2)/2:.0f})")
        elif res["plain"]["pair_tests"] and res["orbit"]["pair_tests"]:
            print(f"  pair-test ratio plain/orbit = {res['plain']['pair_tests']/res['orbit']['pair_tests']:.1f}  "
                  f"(= {math.log2(res['plain']['pair_tests']/res['orbit']['pair_tests']):.2f} bits), "
                  f"memory ratio {res['plain']['stored']/res['orbit']['stored']:.1f}")
    with open(OUT, "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
