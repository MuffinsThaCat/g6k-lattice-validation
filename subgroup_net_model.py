"""
Net attack bits of a SYMMETRIC sieve over a rotation subgroup of order t, dual-attack model (dual_cost_model.py),
re-optimised over beta and m, with the real granularity constraint: an invariant sublattice of the module has rank
a multiple of t, so the sieve dimension beta must be a multiple of t.

  sieve gain      G = 0.95 t  (measured 0.95 t at t = 4, 8, 16 and ~1.0 t at the full group d = 32)
  quality penalty e = c_t * slope(beta) * t  bits of log2 Gram-Schmidt norm per vector, slope = 2 log2 delta(beta);
                  dual vectors longer by 2^e (the whole sieved first-beta sublattice is orbit-closed).
Three assumptions for c_t (measured at d = 32: t = 4/8/16/32 -> 0.33/0.27/0.19/0.12):
  const-0.12 : c_t = 0.12 for every t (the full-pool value at t = d = 32)      -> penalty linear in t
  const-0.17 : c_t = 0.17                                                     -> penalty linear in t
  fit        : c_t = 0.327 (t/4)^-0.47 (power-law fit of the four points)     -> penalty ~ t^0.53

EXTRAPOLATION from t <= 32 (toy rings) to t up to 256; d <= 32 measurements only.
  python subgroup_net_model.py
"""
import math

import dual_cost_model as M

TS = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def cost(beta, n, q, sigma, m, t, c, z=M.Z):
    D = m + n
    ln_delta = math.log(M.delta(beta))
    slope = 2.0 * ln_delta / math.log(2)
    e = c * slope * t if t > 1 else 0.0
    gain = math.log2(0.95 * t) if t > 1 else 0.0
    ln_L = 0.5 * math.log(4.0 / 3.0) + 0.5 * math.log(beta / (2 * math.pi * math.e)) \
        + (D - beta) * ln_delta + (n / D) * math.log(q) + e * math.log(2)
    x = 2 * math.pi ** 2 * sigma ** 2 * math.exp(2 * ln_L) / q ** 2
    if x > 700:
        return float("inf")
    log2_nreq = (math.log(z * z / 2.0) + 2 * x) / math.log(2)
    log2_supply = M.C_SUPPLY * beta - 1.0
    return M.C_SIEVE * beta - gain + max(0.0, log2_nreq - log2_supply)


def best(n, q, sigma, t, c):
    out = (float("inf"), 0, 0)
    step = max(1, t)
    for beta in range(((120 + step - 1) // step) * step, 1300, step):
        for m in range(max(16, n // 8), n + 1, max(1, n // 64)):
            v = cost(beta, n, q, sigma, m, t, c)
            if v < out[0]:
                out = (v, beta, m)
    return out


def c_of(assump, t):
    if assump == "const-0.12":
        return 0.12
    if assump == "const-0.17":
        return 0.17
    return 0.327 * (max(t, 4) / 4.0) ** (-0.47)


def main():
    schemes = [("Kyber-512", 512, 3329, 1.0), ("Kyber-768", 768, 3329, 1.0), ("Kyber-1024", 1024, 3329, 1.0)]
    for name, n, q, s in schemes:
        base = best(n, q, s, 1, 0.0)[0]
        print(f"\n{name}: base dual-attack cost {base:.1f} bits.  Net bits saved by a symmetric sieve over a subgroup of order t")
        print(f"  {'assumption':11s} " + " ".join(f"t={t:<4d}" for t in TS[2:]))
        for a in ("const-0.12", "const-0.17", "fit"):
            row = [base - best(n, q, s, t, c_of(a, t))[0] for t in TS[2:]]
            bt = TS[2:][max(range(len(row)), key=lambda i: row[i])]
            print(f"  {a:11s} " + " ".join(f"{r:+6.1f}" for r in row) + f"   best t = {bt}", flush=True)


if __name__ == "__main__":
    main()
