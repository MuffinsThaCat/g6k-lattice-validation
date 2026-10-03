"""
Subgroup-aligned tour vs plain HKZ-style reduction: end-to-end cost AND profile quality (numpy toy).

PLAIN   : for l = 0, 1, ...: sieve the projected lattice pi_l(L) with the bucketed sieve (no symmetry), take the
          shortest projected vector, insert it (a row of the prefix basis).
ALIGNED : for j = 0, 1, ...: sieve pi_{M_j}(L) with the ORBIT sieve (subgroup order t, rotation-equivariant sketches);
          among the database orbit representatives choose u minimising the volume of the orbit lattice R_t pi(u)
          (Gram determinant of its t rotations) and append the t rows u, yu, ..., y^(t-1) u to the prefix.
          The prefix span stays y-invariant, so the next projected lattice is again y-invariant.
Both start from the same random lattice, use the same sieve code paths and stop when the remaining dimension
falls below MIN_DIM (small sieves are unreliable); only the head of the profile, which is what a dual attack
uses, is compared.  Counts: xor-popcounts + full scalar products (G6K's units).

Reported: ops for the plain and the aligned reduction to the same prefix length, and the profile quality as the
excess (bits of mean log2 Gram-Schmidt norm per vector) of the aligned prefix over the plain one at each block
boundary, with c = excess / (slope * t).

  python aligned_tour.py     Env: AT_D=8 AT_K=6 AT_T=8 AT_Q=257 AT_SEEDS=2 AT_MINDIM=26 AT_MAXROUNDS=30000
"""
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import orbit_sieve_proto as O
import orbit_bucketed_sieve as B

D = int(os.environ.get("AT_D", "8"))
K = int(os.environ.get("AT_K", "6"))
TT = int(os.environ.get("AT_T", "0"))
Q = int(os.environ.get("AT_Q", "257"))
SEEDS = int(os.environ.get("AT_SEEDS", "2"))
MIN_DIM = int(os.environ.get("AT_MINDIM", "26"))
MAXROUNDS = int(os.environ.get("AT_MAXROUNDS", "30000"))
PAT_PLAIN = int(os.environ.get("AT_PAT_PLAIN", "4000"))     # rounds without a new short vector before giving up
PAT_ORBIT = int(os.environ.get("AT_PAT_ORBIT", "600"))
OUT = os.environ.get("AT_OUT", "aligned_tour_results.json")


class Prefix:
    def __init__(self, n, F):
        self.n = n
        self.F = F
        if len(F):
            Qm, _ = np.linalg.qr(F.T)
            self.Pi = np.eye(n) - Qm @ Qm.T
            self.Fp = np.linalg.pinv(F)
            self.logvol = 0.5 * np.linalg.slogdet(F @ F.T)[1] / math.log(2)
        else:
            self.Pi = np.eye(n)
            self.Fp = None
            self.logvol = 0.0

    def reduce(self, W):
        if self.Fp is None:
            return W
        return W - np.rint(W @ self.Fp) @ self.F


def gh_proj_sq(logvolL, pre, m):
    logv = logvolL - pre.logvol                       # log2 vol of the projected lattice
    return math.exp(2.0 * math.lgamma(m / 2.0 + 1.0) / m) / math.pi * 2.0 ** (2.0 * logv / m)


def sample_projected(Bn, pre, count, rng, m, radius=None):
    """Lattice vectors Babai-rounded around Gaussian targets in the projected space (as orbit_sieve_proto.sample_db
    does for the full lattice); lifts kept, reduced against the prefix."""
    n = Bn.shape[0]
    Binv = np.linalg.inv(Bn)
    out, X, seen = [], [], set()
    guard = 0
    if radius is None:
        radius = 3.0 * math.sqrt(max(sample_projected.gh2, 1e-9))
    while len(out) < count and guard < 50 * count:
        guard += 1
        tg = rng.normal(0, radius / math.sqrt(m), size=n) @ pre.Pi
        w = pre.reduce((np.rint(tg @ Binv) @ Bn)[None, :])[0]
        x = w @ pre.Pi
        k = O.canon_key(w)
        if np.linalg.norm(x) < 1e-6 or k in seen:
            continue
        seen.add(k)
        out.append(w)
        X.append(x)
    return np.array(out), np.array(X)


def reuse_db(Wc, pre, count, Bn, rng, m, P=None, t=1):
    """Project the previous database onto the new complement and keep the `count` shortest distinct entries
    (G6K's pump reuses its database across positions the same way); top up with fresh samples if short."""
    W = pre.reduce(Wc)
    X = W @ pre.Pi
    nm = (X * X).sum(1)
    keep, keys = [], set()
    for i in np.argsort(nm):
        if nm[i] < 1e-6:
            continue
        k = O.orbit_key(W[i], P, t) if t > 1 else O.canon_key(W[i])
        if k in keys:
            continue
        keys.add(k)
        keep.append(i)
        if len(keep) >= count:
            break
    W = W[keep]
    if len(W) < count:
        extra, _ = sample_projected(Bn, pre, count - len(W), rng, m)
        W = np.vstack([W, extra])
    return W


def plain_sieve_proj(Wl, Hm, pre, m, R2, target, alpha, rng, cnt):
    W = Wl.copy()
    X = W @ pre.Pi
    N = len(W)
    S = B.pack(X @ Hm >= 0)
    nm = (X * X).sum(1)
    seen = {O.canon_key(w) for w in W}
    rounds = 0
    best_prog, last_imp = float("inf"), 0
    while rounds < MAXROUNDS:
        sc = int((nm <= R2).sum())
        if sc >= target:
            break
        prog = float(np.sort(nm)[: max(1, N // 4)].mean())     # progress = shrinking of the shortest quartile
        if prog < best_prog * (1 - 2e-4):
            best_prog, last_imp = prog, rounds
        elif rounds - last_imp >= PAT_PLAIN:
            break
        rounds += 1
        maxlen = np.quantile(nm, B.QUANT)
        order = np.argsort(nm)
        ci = int(order[rng.randint(N // 4, max(N // 4 + 1, N // 2))])
        c, nc = X[ci], nm[ci]
        h = B.ham(S, S[ci])
        cnt.pop += N - 1
        cand = np.flatnonzero(B.passes(h, B.T_BUCKET))
        cand = cand[cand != ci]
        new = []
        if len(cand):
            dots = X[cand] @ c
            cnt.dots += len(cand)
            sel = np.abs(dots) >= alpha * math.sqrt(nc) * np.sqrt(nm[cand])
            for j, dj in zip(cand, dots):
                if nc + nm[j] - 2 * abs(dj) < maxlen:
                    new.append(W[ci] - np.sign(dj) * W[j])
            bidx = cand[sel]
            M = len(bidx)
            if M > 1:
                SB = S[bidx]
                iu, ju = np.triu_indices(M, 1)
                hh = B.ham(SB[iu], SB[ju])
                cnt.pop += len(iu)
                ok = B.passes(hh, B.T_INNER)
                iu, ju = iu[ok], ju[ok]
                if len(iu):
                    A_, B_ = X[bidx[iu]], X[bidx[ju]]
                    dd = (A_ * B_).sum(1)
                    cnt.dots += len(iu)
                    nn = nm[bidx[iu]] + nm[bidx[ju]] - 2 * np.abs(dd)
                    for a_, b_, d_ in zip(iu[nn < maxlen], ju[nn < maxlen], dd[nn < maxlen]):
                        new.append(W[bidx[a_]] - np.sign(d_) * W[bidx[b_]])
        if new:
            new = [pre.reduce(u[None, :])[0] for u in new]
            xs = [u @ pre.Pi for u in new]
            idx = np.argsort([float(x @ x) for x in xs])
            slots = list(np.argsort(-nm))
            si = 0
            for ii in idx:
                if si >= len(slots) // 3:
                    break
                u, x = new[ii], xs[ii]
                un = float(x @ x)
                k = O.canon_key(u)
                if un < 1e-9 or k in seen:
                    continue
                t_ = slots[si]
                if un >= nm[t_]:
                    break
                seen.discard(O.canon_key(W[t_]))
                seen.add(k)
                W[t_], X[t_], nm[t_] = u, x, un
                S[t_] = B.pack((x @ Hm >= 0)[None, :])[0]
                si += 1
    return W, nm, rounds, int((nm <= R2).sum())


def orbit_sieve_proj(Wl, Hm, pre, P, t, m, R2, target, alpha, rng, cnt):
    W = Wl.copy()
    X = W @ pre.Pi
    R = len(W)
    N = R * t
    nm = (X * X).sum(1)
    seen = {O.orbit_key(w, P, t) for w in W}
    EX = O.rotations(X, P, t)                       # (t, R, n) projected rotations
    EW = O.rotations(W, P, t)                       # (t, R, n) ambient rotations
    SE = B.pack(EX.reshape(t * R, -1) @ Hm >= 0).reshape(t, R, -1)
    rounds = 0
    best_prog, last_imp = float("inf"), 0

    def n_short():
        keys = set()
        for b in np.flatnonzero(nm <= R2):
            for r in range(t):
                keys.add(O.canon_key(EX[r, b]))
        return len(keys)

    while rounds < MAXROUNDS:
        if rounds % 10 == 0:
            sc = n_short()
            if sc >= target:
                break
            prog = float(np.sort(nm)[: max(1, R // 4)].mean())
            if prog < best_prog * (1 - 2e-4):
                best_prog, last_imp = prog, rounds
            elif rounds - last_imp >= PAT_ORBIT:
                break
        rounds += 1
        maxlen = np.quantile(nm, B.QUANT)
        order = np.argsort(nm)
        ci = int(order[rng.randint(R // 4, max(R // 4 + 1, R // 2))])
        c, nc = X[ci], nm[ci]
        h = B.ham(SE, SE[0, ci])
        cnt.pop += N - 1
        pr, pb = np.nonzero(B.passes(h, B.T_BUCKET))
        keep = ~((pr == 0) & (pb == ci))
        pr, pb = pr[keep], pb[keep]
        new = []
        if len(pr):
            Vc = EX[pr, pb]
            dots = Vc @ c
            cnt.dots += len(pr)
            vn = nm[pb]
            for jr, jb, dj, vj in zip(pr, pb, dots, vn):
                if nc + vj - 2 * abs(dj) < maxlen:
                    new.append(W[ci] - np.sign(dj) * EW[jr, jb])
            sel = np.abs(dots) >= alpha * math.sqrt(nc) * np.sqrt(vn)
            br, bb = pr[sel], pb[sel]
            M = len(br)
            if M > 1:
                SB = SE[br, bb]
                iu, ju = np.triu_indices(M, 1)
                hh = B.ham(SB[iu], SB[ju])
                cnt.pop += len(iu)
                ok = B.passes(hh, B.T_INNER)
                iu, ju = iu[ok], ju[ok]
                if len(iu):
                    A_, B_ = EX[br[iu], bb[iu]], EX[br[ju], bb[ju]]
                    dd = (A_ * B_).sum(1)
                    cnt.dots += len(iu)
                    nn = nm[bb[iu]] + nm[bb[ju]] - 2 * np.abs(dd)
                    for a_, b_, d_ in zip(iu[nn < maxlen], ju[nn < maxlen], dd[nn < maxlen]):
                        new.append(EW[br[a_], bb[a_]] - np.sign(d_) * EW[br[b_], bb[b_]])
        if new:
            new = [pre.reduce(u[None, :])[0] for u in new]
            xs = [u @ pre.Pi for u in new]
            idx = np.argsort([float(x @ x) for x in xs])
            slots = list(np.argsort(-nm))
            si = 0
            for ii in idx:
                if si >= len(slots) // 3:
                    break
                u, x = new[ii], xs[ii]
                un = float(x @ x)
                if un < 1e-9:
                    continue
                k = O.orbit_key(u, P, t)
                if k in seen:
                    continue
                t_ = slots[si]
                if un >= nm[t_]:
                    break
                seen.discard(O.orbit_key(W[t_], P, t))
                seen.add(k)
                W[t_], X[t_], nm[t_] = u, x, un
                cx, cw = x, u
                for r in range(t):
                    EX[r, t_], EW[r, t_] = cx, cw
                    cx, cw = cx @ P, cw @ P
                SE[:, t_] = B.pack(EX[:, t_] @ Hm >= 0)
                si += 1
    return W, X, nm, rounds


def orbit_logvol(Xrep, P, t):
    rots = [Xrep]
    for _ in range(t - 1):
        rots.append(rots[-1] @ P)
    Rs = np.array(rots)
    G = np.einsum("rin,sin->irs", Rs, Rs)
    sign, logdet = np.linalg.slogdet(G)
    out = 0.5 * logdet / math.log(2)
    out[sign <= 0] = np.inf
    return out


def tour(Bn, P, ring_d, t, k, mode, nblocks, logvolL, seed, counters):
    n = Bn.shape[0]
    rng = np.random.RandomState(seed)
    F = np.zeros((0, n))
    profile = []                                         # mean log2 GS norm per position (aligned: per block)
    ops_log = []
    pos = 0
    carry = None
    while pos < nblocks * (t if mode == "aligned" else 1) and n - pos >= MIN_DIM:
        pre = Prefix(n, F)
        m = n - pos
        gh2 = gh_proj_sq(logvolL, pre, m)
        sample_projected.gh2 = gh2
        R2 = 4.0 / 3.0 * gh2
        N = int(3.2 * (4 / 3) ** (m / 2))
        target = 0.5 * (4 / 3) ** (m / 2) / 2
        alpha = B.alpha_for(m, N)
        cnt = B.Counter()
        if mode == "plain":
            if carry is None:
                Wl, _ = sample_projected(Bn, pre, N, rng, m)
            else:
                Wl = reuse_db(carry, pre, N, Bn, rng, m)
            Hm = rng.normal(size=(n, B.BITS))
            W, nm, rounds, short = plain_sieve_proj(Wl, Hm, pre, m, R2, target, alpha, rng, cnt)
            carry = W
            j = int(np.argmin(nm))
            profile.append(math.log2(math.sqrt(nm[j])))
            F = np.vstack([F, W[j][None, :]])
            pos += 1
        else:
            R = max(8, N // t)
            if carry is None:
                Wl, _ = sample_projected(Bn, pre, R, rng, m)
            else:
                Wl = reuse_db(carry, pre, R, Bn, rng, m, P, t)
            msk = max(1, round(B.BITS / t))
            Hm = B.check_equivariance(Bn, P, t, n, msk, rng)
            W, X, nm, rounds = orbit_sieve_proj(Wl, Hm, pre, P, t, m, R2, target, alpha, rng, cnt)
            lv = orbit_logvol(X, P, t)
            b = int(np.argmin(lv))
            carry = W
            u = W[b]
            rows = [u]
            for _ in range(1, t):
                rows.append(rows[-1] @ P)
            F = np.vstack([F, np.array(rows)])
            profile.append(float(lv[b]) / t)
            short = int((nm <= R2).sum()) * t               # expanded: each representative stands for t vectors
            pos += t
        ops_log.append({"pos": pos, "m": m, "pop": cnt.pop, "dots": cnt.dots, "rounds": rounds, "short": short})
        counters.pop += cnt.pop
        counters.dots += cnt.dots
    return profile, ops_log


def main():
    ring_d, k = D, K
    t = TT or ring_d
    n = ring_d * k
    rows = []
    print(f"ring d={ring_d} rank {k} (n={n}), subgroup t={t}, q={Q}; sieve until remaining dim < {MIN_DIM}")
    for seed in range(SEEDS):
        Bn, P0 = O.make_lattice(ring_d, k, Q, 40 + seed)
        P = np.linalg.matrix_power(P0, ring_d // t) if t != ring_d else P0
        logvolL = np.linalg.slogdet(Bn)[1] / math.log(2)
        nblocks = (n - MIN_DIM) // t + (1 if (n - MIN_DIM) % t else 0)
        nblocks = max(1, (n - MIN_DIM) // t)
        t0 = time.time()
        ca = B.Counter()
        prof_a, log_a = tour(Bn, P, ring_d, t, k, "aligned", nblocks, logvolL, seed, ca)
        covered = len(prof_a) * t
        cp = B.Counter()
        prof_p, log_p = tour(Bn, P, ring_d, t, k, "plain", covered, logvolL, seed, cp)
        # compare at each block boundary
        res = {"seed": seed, "t": t, "blocks": len(prof_a), "covered": covered,
               "aligned_ops": ca.pop + ca.dots, "plain_ops": cp.pop + cp.dots, "secs": time.time() - t0}
        pa, pp = np.array(prof_a), np.array(prof_p)
        slope = float((pp[0] - pp[min(len(pp), covered) - 1]) / max(1, min(len(pp), covered) - 1))
        exc = []
        for j in range(len(prof_a)):
            r = (j + 1) * t
            qa = float(np.sum(pa[: j + 1]) * t)               # sum of log2 GS norms over the first r positions
            qp = float(np.sum(pp[:r]))
            exc.append((qa - qp) / r)
        res.update({"slope": slope, "excess_per_vec": exc, "c": [e / max(slope * t, 1e-9) for e in exc]})
        rows.append(res)
        for name, lg in (("aligned", log_a), ("plain", log_p)):
            for e in lg:
                tgt = 0.5 * (4 / 3) ** (e["m"] / 2) / 2
                print(f"    {name:7s} step at pos {e['pos']:3d} (dim {e['m']}): rounds {e['rounds']:6d}, short {e['short']} "
                      f"(target {tgt:.0f}{'' if e['short'] >= tgt else ', NOT reached'}), ops {e['pop'] + e['dots']:.3g}")
        res["log_aligned"], res["log_plain"] = log_a, log_p
        ratio = res["plain_ops"] / max(res["aligned_ops"], 1)
        print(f"seed {seed}: prefix covered {covered}; ops plain {res['plain_ops']:.3g} vs aligned {res['aligned_ops']:.3g} "
              f"-> {ratio:.1f}x ({math.log2(ratio):+.2f} bits); slope {slope:.4f} bits/idx; "
              f"excess/vec at block ends {[round(e, 3) for e in exc]}; c {[round(c, 2) for c in res['c']]}  ({res['secs']:.0f}s)", flush=True)
    with open(OUT, "w") as fh:
        json.dump(rows, fh, indent=1, default=float)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
