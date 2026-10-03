"""
Net attack bits of the orbit sieve = (sieve speed-up) - (aligned-prefix quality penalty), in the dual-attack
cost model of dual_cost_model.py, re-optimised over beta and m for every setting.

 * Orbit sieve speed-up: sieve cost divided by G (measured: G ~ 0.9 d in G6K's counting units at toy size;
   d = 256 for Kyber's ring; G = 0.5 d shown as a pessimistic case).
 * Aligned-prefix penalty (rotation_prefix_proxy.py): an orbit-closed prefix has a staircase Gram-Schmidt
   profile. Measured on real G6K databases, the excess of the mean log2 GS norm over an unconstrained
   prefix is  e = c * slope * d  bits per vector, with slope = bits of log2 GS norm lost per index
   (= 2 log2 delta(beta) for a BKZ-beta basis) and c ~ 0.1 - 0.3 (power-of-two rings d = 8/16/32: mean 0.14, 0.11, 0.23).
   The dual vectors come from the sieved sublattice, whose Gaussian-heuristic length is therefore longer
   by 2^e.  This is applied to the whole first-beta block (the sieved sublattice is entirely orbit-closed).

An EXTRAPOLATION in d and beta from toy measurements (d <= 32, beta <= 64) to d = 256, beta ~ 400-900.
Only the dual attack's sublattice sieve is modelled; the primal/projected-block case differs.

  python aligned_penalty_model.py
"""
import math

import dual_cost_model as M


def cost(beta, n, q, sigma, m, g, G_bits, c, d, z=M.Z):
    D = m + n
    ln_delta = math.log(M.delta(beta))
    slope = 2.0 * ln_delta / math.log(2)              # bits of log2 GS norm per index (GSA)
    e = c * slope * d                                  # aligned-prefix excess, bits per vector
    ln_L = 0.5 * math.log(4.0 / 3.0) + 0.5 * math.log(beta / (2 * math.pi * math.e)) \
        + (D - beta) * ln_delta + (n / D) * math.log(q) + e * math.log(2)
    x = 2 * math.pi ** 2 * sigma ** 2 * math.exp(2 * ln_L) / q ** 2
    if x > 700:
        return float("inf")
    log2_nreq = (math.log(z * z / 2.0) + 2 * x) / math.log(2)
    log2_supply = M.C_SUPPLY * beta + math.log2(g) - 1.0
    return M.C_SIEVE * beta - G_bits + max(0.0, log2_nreq - log2_supply)


def best(n, q, sigma, G_bits, c, d):
    out = (float("inf"), 0, 0)
    for beta in range(120, 1300):
        for m in range(max(16, n // 8), n + 1, max(1, n // 64)):
            v = cost(beta, n, q, sigma, m, 1, G_bits, c, d)
            if v < out[0]:
                out = (v, beta, m)
    return out


def main():
    d = 256
    schemes = [("Kyber-512", 512, 3329, 1.0), ("Kyber-768", 768, 3329, 1.0), ("Kyber-1024", 1024, 3329, 1.0)]
    print("net bits saved (negative = the aligned-basis penalty outweighs the sieve speed-up); dual attack model, d = 256")
    print(f"{'scheme':11s} {'sieve gain':>11s}  " + "  ".join(f"c={c:<4}" for c in (0.0, 0.1, 0.17, 0.25, 0.3)))
    for name, n, q, sigma in schemes:
        base = best(n, q, sigma, 0.0, 0.0, d)[0]
        for label, G in (("G=0.9d", 0.9 * d), ("G=0.5d", 0.5 * d)):
            row = []
            for c in (0.0, 0.1, 0.17, 0.25, 0.3):
                v, b, m = best(n, q, sigma, math.log2(G), c, d)
                row.append(base - v)
            print(f"{name:11s} {label:>11s}  " + "  ".join(f"{r:+6.1f}" for r in row))
    print("\nc = 0 is the symmetry speed-up with NO aligned-prefix penalty (the optimistic reading).")


if __name__ == "__main__":
    main()
