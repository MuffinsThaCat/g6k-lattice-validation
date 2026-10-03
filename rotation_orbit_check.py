"""
Does ring rotation inflate a G6K sieve database?  (R1, no kernel change needed)

Lattice: NTRU-type q-ary ideal lattice in R_q = Z_q[x]/(x^d+1), dim 2d, basis rows
[e_i | x^i*h] and [0 | q*e_i].  Multiplication by x (negacyclic shift on both halves) is a
norm-preserving automorphism P of order 2d, so every short vector v gives 2d short vectors v*P^k.
In coefficients wrt the sieve's basis B:  c -> c*M with  M = B P B^-1  (must be an integer matrix).

Measured (R1), after a real G6K run (ARM build, d = 20/24/26, bgj1, default parameters):
  * The sieve at saturation 0.5 finds about half of the ~(4/3)^(n/2)/2 short +-pairs
    (|v|^2 <= 4/3 GH^2), with NO rotation bias: a found vector's non-trivial rotation images are in
    the found set with probability ~ the saturation fraction.
  * Closing the found short vectors under the rotation recovers the full short set
    (d=24: 249 -> 504 vs ~498 expected; d=20: 80 -> 180 vs ~158; d=26: 442 -> 780 vs ~886), from
    only ~n/2 orbits. So the rotation structure is real and complete in G6K's own output.
  * Post-hoc closure of a SMALLER sieve (m >= 2, db_size_factor / m) is NOT a valid test: a
    database that small cannot sieve at all (it found 0 short vectors in ~0.04 s). A database needs
    about (4/3)^(n/2) points just to find reducing pairs, so any saving has to come from rotated
    images entering the database DURING the sieve (see ROTATION_KERNEL_SPEC.md). The m >= 2 runs are
    kept only to document this; expect "0 -> 0".

  python rotation_orbit_check.py --selftest            # math only, needs numpy; runs anywhere
  python rotation_orbit_check.py                       # needs the G6K checkout as cwd (or G6K_DIR)
Env: ROT_D=24  ROT_Q=257  ROT_MS="1"  ROT_SEED=1  ROT_THREADS=1  ROT_TIMEOUT=1800 ROT_OUT=...json
     G6K_DIR=<g6k checkout> if not run from inside it

STATUS: run on real G6K (ARM build, 2026-10-02), not yet on x86. The selftest also passes.
"""
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

# G6K is built in place, not installed: make the g6k checkout (cwd, or G6K_DIR) importable.
sys.path.insert(0, os.environ.get("G6K_DIR", os.getcwd()))

D = int(os.environ.get("ROT_D", "24"))
Q = int(os.environ.get("ROT_Q", "257"))
MS = [int(x) for x in os.environ.get("ROT_MS", "1").split(",")]
SEED = int(os.environ.get("ROT_SEED", "1"))
THREADS = int(os.environ.get("ROT_THREADS", "1"))
TIMEOUT = float(os.environ.get("ROT_TIMEOUT", "1800"))
OUT = os.environ.get("ROT_OUT", "rotation_orbit_results.json")


def neg_shift(d):
    N = np.zeros((d, d), dtype=np.int64)
    for i in range(d - 1):
        N[i, i + 1] = 1
    N[d - 1, 0] = -1          # x * x^(d-1) = x^d = -1
    return N


def ideal_basis(d, q, seed):
    rng = np.random.RandomState(seed)
    h = rng.randint(0, q, size=d).astype(np.int64)
    N = neg_shift(d)
    H = np.zeros((d, d), dtype=np.int64)
    row = h.copy()
    for i in range(d):
        H[i] = row % q
        row = row @ N          # next row = x * previous
    B = np.zeros((2 * d, 2 * d), dtype=np.int64)
    B[:d, :d] = np.eye(d, dtype=np.int64)
    B[:d, d:] = H
    B[d:, d:] = q * np.eye(d, dtype=np.int64)
    P = np.zeros((2 * d, 2 * d), dtype=np.int64)
    P[:d, :d] = N
    P[d:, d:] = N
    return B, P


def rotation_matrix(B, P):
    """M = B P B^-1 in coefficient space (row vectors). Returns an exact integer matrix or raises."""
    M = np.rint(B @ P @ np.linalg.inv(B.astype(float))).astype(np.int64)
    if not np.array_equal(M @ B, B @ P):
        raise ValueError("rotation is not an integer map in this basis (basis/automorphism mismatch)")
    return M


def canon(c):
    for x in c:
        if x != 0:
            return tuple(c) if x > 0 else tuple(-y for y in c)
    return tuple(c)


def closure(vectors, M, order):
    out = set()
    for c in vectors:
        v = np.array(c, dtype=np.int64)
        for _ in range(order):
            out.add(canon(tuple(int(x) for x in v)))
            v = v @ M
    return out


def image_hit_rate(vectors, M, order, reference):
    """Fraction of NON-trivial rotation images (k = 1..order-1) that lie in `reference`.
    (closure() includes k = 0, i.e. each vector itself, so a closure-based self-check is always 1.)"""
    hits = total = 0
    for c in vectors:
        v = np.array(c, dtype=np.int64)
        for _ in range(order - 1):
            v = v @ M
            total += 1
            hits += canon(tuple(int(x) for x in v)) in reference
    return hits / total if total else float("nan")


def selftest():
    for d, q in [(8, 97), (16, 257), (24, 257)]:
        B, P = ideal_basis(d, q, 3)
        # (1) automorphism preserves the lattice: images of basis rows are integer combinations
        M = rotation_matrix(B, P)
        # (2) order divides 2d
        Mk = np.eye(2 * d, dtype=np.int64)
        for _ in range(2 * d):
            Mk = Mk @ M
        assert np.array_equal(Mk, np.eye(2 * d, dtype=np.int64)), "order does not divide 2d"
        # (3) norm preserving on random lattice vectors
        rng = np.random.RandomState(0)
        for _ in range(50):
            c = rng.randint(-3, 4, size=2 * d)
            assert (c @ B) @ (c @ B) == ((c @ M) @ B) @ ((c @ M) @ B)
        # (4) the same holds in a non-trivial basis B' = U B (what LLL/BKZ would hand to G6K)
        U = np.eye(2 * d, dtype=np.int64)
        for _ in range(6 * d):
            i, j = rng.choice(2 * d, 2, replace=False)
            U[i] += rng.randint(-2, 3) * U[j]
        B2 = U @ B
        M2 = rotation_matrix(B2, P)
        c = rng.randint(-3, 4, size=2 * d)
        assert np.array_equal((c @ M2) @ B2, ((c @ B2) @ P)), "M2 does not implement P"
        # (5) the hit-rate metric separates a rotation-closed db (1.0) from a random one (~0)
        base = [tuple(int(x) for x in rng.randint(-1, 2, 2 * d)) for _ in range(20)]
        closed = closure(base, M, 2 * d)
        assert image_hit_rate(closed, M, 2 * d, closed) == 1.0
        assert image_hit_rate(set(base), M, 2 * d, set(base)) < 0.2
        print(f"selftest d={d} q={q}: integer M, order | 2d, norm-preserving, basis-change, "
              "hit-rate metric OK")
    print("SELFTEST PASSED")


def _sieve_worker(args, qout):
    """Runs in a child process so a non-terminating undersized sieve can be killed."""
    from fpylll import IntegerMatrix, LLL, BKZ
    from g6k import Siever
    try:
        from g6k.siever_params import SieverParams
    except ImportError:
        from g6k.siever import SieverParams
    B, m = args
    n = B.shape[0]
    A = IntegerMatrix.from_matrix(B.tolist())
    A = LLL.reduction(A)
    BKZ.reduction(A, BKZ.Param(block_size=min(20, n)))  # in place; do not rely on its return value
    g = Siever(A, SieverParams(threads=THREADS, default_sieve="bgj1", reserved_n=n,
                               db_size_factor=3.2 / m), seed=SEED)
    g.initialize_local(0, 0, n)
    try:
        from g6k.siever import SaturationError
    except ImportError:
        SaturationError = RuntimeError
    saturated = True
    t0 = time.perf_counter()
    try:
        g(alg="bgj1")
    except SaturationError:   # raised for n >= 50 when the (shrunk) db cannot reach saturation
        saturated = False
    wall = time.perf_counter() - t0
    Bn = np.array([[A[i, j] for j in range(n)] for i in range(n)], dtype=np.int64)
    db = [tuple(int(x) for x in v) for v in g.itervalues()]
    qout.put({"m": m, "wall": wall, "db": db, "B": Bn.tolist(), "saturated": saturated})


def run_sieve(B, m):
    # Read the result BEFORE joining: a child blocked flushing a large result into the queue never
    # exits, so join-then-get would misreport every big database as a timeout.
    import queue as _queue
    q = mp.Queue()
    p = mp.Process(target=_sieve_worker, args=((B, m), q))
    p.start()
    deadline = time.time() + TIMEOUT
    res = None
    while time.time() < deadline:
        try:
            res = q.get(timeout=1.0)
            break
        except _queue.Empty:
            if not p.is_alive():  # child died (crash / G6K exit()) without a result
                try:
                    res = q.get(timeout=1.0)
                except _queue.Empty:
                    pass
                break
    p.join(5)
    if p.is_alive():
        p.terminate()
    return res


def main():
    d, n = D, 2 * D
    B0, P = ideal_basis(d, Q, SEED)
    results = {"d": d, "q": Q, "runs": {}}
    full = run_sieve(B0, 1)
    if full is None:
        sys.exit("full-size run failed or timed out")
    Bf = np.array(full["B"], dtype=np.int64)
    # P acts on ambient coordinates; LLL/BKZ only change the basis of the same lattice, so the
    # coefficient-space map for the sieve basis is M = Bf P Bf^-1 (selftest case 4 checks this).
    Mf = rotation_matrix(Bf, P)
    # Saturation counts only SHORT vectors: those with |v|^2 <= (4/3) GH^2 (G6K's saturation_radius).
    # The rest of a database is long vectors that differ from run to run, so comparing whole
    # databases says nothing (a first real run gave coverage exactly 0.000 for that reason).
    logdet = np.linalg.slogdet(Bf.astype(float))[1]
    from math import lgamma, pi as _pi, exp as _exp
    # exact Gaussian heuristic (the n/(2 pi e) form is ~11% low in squared length at n = 48)
    gh2 = _exp(2.0 * lgamma(n / 2.0 + 1.0) / n) / _pi * float(np.exp(2.0 * logdet / n))
    R2 = (4.0 / 3.0) * gh2
    expected_short = (4.0 / 3.0) ** (n / 2.0) / 2.0   # expected number of +-pairs below the radius

    def short_only(vs, basis):
        out = set()
        for c in vs:
            v = np.array(c, dtype=np.int64) @ basis
            if float(v @ v) <= R2:
                out.add(c)
        return out

    s_full = {canon(v) for v in full["db"]}
    s_short = short_only(s_full, Bf)
    hit = image_hit_rate(s_short, Mf, 2 * d, s_short) if s_short else float("nan")
    print(f"d={d} n={n} q={Q}: full db {len(s_full)} vectors, of which {len(s_short)} short "
          f"(|v|^2 <= 4/3 GH^2; about {expected_short:.0f} such +-pairs exist in total)")
    print(f"  fraction of non-trivial rotation images of the short vectors that are already in the "
          f"short set: {hit:.3f} (near 1: already rotation-closed; near 0: rotations add new short vectors)")
    # What rotation alone would give: close the found short set under the ring rotation.
    cl_full = closure(s_short, Mf, 2 * d)
    orbit_reps = {min(closure([v], Mf, 2 * d)) for v in s_short}
    print(f"  rotation closure of the {len(s_short)} found short vectors: {len(cl_full)} short vectors "
          f"({len(cl_full)/expected_short:.2f} of the ~{expected_short:.0f} expected), "
          f"from only {len(orbit_reps)} distinct rotation orbits")
    results["full"] = {"db": len(s_full), "short": len(s_short), "expected_short_pairs": expected_short,
                       "rotation_image_hit_rate": hit, "wall": full["wall"],
                       "short_after_closure": len(cl_full), "orbits_hit": len(orbit_reps)}
    for m in MS:
        if m == 1:
            continue
        r = run_sieve(B0, m)
        if r is None:
            print(f"m={m}: sieve timed out / failed (undersized database did not saturate)")
            results["runs"][str(m)] = {"status": "timeout"}
            continue
        # the sieve basis may differ between runs: express db in the FULL run's basis via ambient vectors
        Bm = np.array(r["B"], dtype=np.int64)
        amb = [np.array(c, dtype=np.int64) @ Bm for c in r["db"]]
        Binv = np.linalg.inv(Bf.astype(float))
        small = set()
        for a in amb:
            c = np.rint(a @ Binv).astype(np.int64)
            assert np.array_equal(c @ Bf, a)
            small.add(canon(tuple(int(x) for x in c)))
        small_short = short_only(small, Bf)          # rotating long vectors cannot help saturation
        cl = closure(small_short, Mf, 2 * d)         # all images are short too (norm preserved)
        denom = max(len(s_short), 1)
        cov0 = len(small_short & s_short) / denom
        cov1 = len(cl & s_short) / denom
        print(f"m={m}: db {len(small)} (x1/{m}), wall {r['wall']:.2f}s vs {full['wall']:.2f}s | "
              f"short vectors found: {len(small_short)} -> {len(cl)} after rotation closure "
              f"(full run: {len(s_short)}; ~{expected_short:.0f} exist) | "
              f"overlap with full run's short set: raw {cov0:.3f} -> closure {cov1:.3f}"
              f"{'' if r.get('saturated', True) else '  [stopped before requested saturation]'}")
        results["runs"][str(m)] = {"saturated": r.get("saturated", True), "db": len(small),
                                   "wall": r["wall"], "short_found": len(small_short),
                                   "short_after_closure": len(cl), "coverage_raw": cov0,
                                   "coverage_closure": cov1}
    with open(OUT, "w") as f:
        json.dump(results, f, indent=1)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
