"""
Dual-attack cost model with rotation amplification g (how many bits does a supply gain g buy?).

Standard sieve-based dual attack (Kyber-spec / Albrecht-Shen style, no key guessing, no FFT):
  LWE(n, q, sigma), m <= n samples. Dual lattice {(v, w): A^T v = w mod q}: dimension D = m + n,
  determinant q^n (rows (e_i, A^T e_i) and (0, q e_j)), and <v,b> = <(v,w),(e,s)> mod q, so the noise is sigma^2 * L^2 (e and s both of std sigma).
  BKZ-beta basis (GSA): b*_i = delta^(D-2i+1) det^(1/D).  Sieve the first beta vectors (a sublattice):
  output length L = sqrt(4/3) * gh(first beta) = sqrt(4/3) sqrt(beta/(2 pi e)) delta^(D-beta) q^(n/D).
  Bias eps = exp(-2 pi^2 sigma^2 L^2 / q^2); samples needed N_req = z^2 / (2 eps^2) (z = detection z-score).
  Sieve supply S = g * 2^(0.2075 beta) / 2 (g = rotation amplification, 1 = none); if N_req > S the sieve
  is repeated N_req / S times. Cost (bits) = 0.292 beta + log2(max(1, N_req / S)).   (core-SVP style)
Minimised over beta and m <= m_max.

This is a deliberately simple model: no dimensions-for-free, no secret guessing, no FFT, no distinguisher
cost, polynomial factors dropped. It is meant to answer one question: how much does multiplying the
sample supply by g move the optimum, relative to the same model with g = 1.

  python dual_cost_model.py
"""
import math

C_SIEVE = 0.292       # classical sieve exponent (core-SVP)
C_SUPPLY = 0.2075     # log2(sqrt(4/3))
Z = 4.0               # detection z-score


def delta(beta):
    return ((math.pi * beta) ** (1.0 / beta) * beta / (2 * math.pi * math.e)) ** (1.0 / (2 * (beta - 1)))


def cost(beta, n, q, sigma, m, g, z=Z):
    D = m + n
    ln_delta = math.log(delta(beta))
    ln_L = 0.5 * math.log(4.0 / 3.0) + 0.5 * math.log(beta / (2 * math.pi * math.e)) \
        + (D - beta) * ln_delta + (n / D) * math.log(q)
    L2 = math.exp(2 * ln_L)
    x = 2 * math.pi ** 2 * sigma ** 2 * L2 / q ** 2     # eps = exp(-x)
    if x > 700:
        return float("inf"), 0.0, 0.0
    log2_nreq = (math.log(z * z / 2.0) + 2 * x) / math.log(2)
    log2_supply = C_SUPPLY * beta + math.log2(g) - 1.0
    extra = max(0.0, log2_nreq - log2_supply)
    return C_SIEVE * beta + extra, log2_nreq, log2_supply


def best(n, q, sigma, g, m_max=None, z=Z):
    m_max = m_max or n
    out = (float("inf"),)
    for beta in range(120, 1300):
        for m in range(max(16, n // 8), m_max + 1, max(1, n // 64)):
            c, ln, ls = cost(beta, n, q, sigma, m, g, z)
            if c < out[0]:
                out = (c, beta, m, ln, ls)
    return out


def main():
    schemes = [("Kyber-512", 512, 3329, 1.0), ("Kyber-768", 768, 3329, 1.0), ("Kyber-1024", 1024, 3329, 1.0)]
    gs = [1, 2, 4, 7, 16, 64, 256]
    print("model dual attack cost (bits) and the saving from a rotation sample gain g; ring dimension 256 caps g at 256")
    print(f"{'scheme':11s} " + " ".join(f"g={g:<4d}" for g in gs) + "   (beta, m at g=1) -> (beta, m at g=256)")
    for name, n, q, sigma in schemes:
        row = [best(n, q, sigma, g) for g in gs]
        base = row[0][0]
        cells = " ".join(f"{base - r[0]:+5.1f} " for r in row)
        print(f"{name:11s} {cells}  ({row[0][1]},{row[0][2]}) -> ({row[-1][1]},{row[-1][2]})   base cost {base:.1f} bits")
    print("\nbase model costs (g=1) for plausibility: compare with published dual-attack core-SVP figures.")


if __name__ == "__main__":
    main()
