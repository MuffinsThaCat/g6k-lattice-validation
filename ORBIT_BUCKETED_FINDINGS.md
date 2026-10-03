# Orbit-aware BUCKETED sieve (bgj1-style) with rotation-equivariant sketches (2026-10-02, numpy toy)

Script: `orbit_bucketed_sieve.py` (uses `orbit_sieve_proto.py`). Counts are G6K's units: xor-popcount tests and
full scalar products, to the SAME stopping rule as G6K (0.5 * (4/3)^(n/2) / 2 vectors with |v|^2 <= 4/3 GH^2).

Baseline: bgj1-like rounds (256-bit SimHash, thresholds 102 / 96, bucket alpha from G6K's formula, replacement
of the longest entries, 0.65-quantile length bound). Calibration: at n = 48 it reaches saturation with
7.3e7 total operations; real G6K bgj1 at n = 48 measured 1.1e8 (xorpopcnt 1.09e8 + scalar products 5.5e6):
the toy baseline is within 1.5x of the real counters.

Orbit version: R = N/d representatives; hash set {x^s h_i} so that sketch(x^r v) is the negacyclic bit-rotation
of sketch(v) (verified numerically in the script); a pair class (a, x^r b) costs ONE popcount, exactly like a
baseline pair, so no FFT overhead. One round's centre stands for its d rotated centres; a new representative
inserts d vectors at once.

Fixed n = 48, default database N = 3189, two instances each (all runs reached the saturation target):

| ring d (rank) | baseline popcounts + dots | orbit popcounts + dots | ratio | bits | memory (full vectors) | short found (baseline / orbit) |
|---|---|---|---|---|---|---|
| 8 (6)  | 7.4e7 + 5.3e6 | 9.7e6 + 7.2e5 | 7.2 - 7.6  | 2.85 - 2.93 | 8x  | 250 / 264-272 |
| 12 (4) | 7.8e7 + 5.7e6 | 7.1e6 + 4.9e5 | 10.0 - 11.0 | 3.33 - 3.46 | 12x | 250 / 252-260 |
| 16 (3) | 7.1e7 + 5.1e6 | 4.9e6 + 3.6e5 | 14.2 - 14.5 | 3.83 - 3.86 | 16x | 250 / 272-384 |
| 24 (2) | 7.4e7 + 5.3e6 | 3.5e6 + 3.0e5 | 20.8 - 20.9 | 4.38 - 4.39 | 24x | 250 / 312-360 |

The reduction is ~0.87-0.95 x d at every ring size, i.e. log2(d) - 0.1..0.2 bits.

NOT established:
- Toy numpy sieve; one lattice family (NTRU/module q-ary, q = 257); n <= 64.
- Full-lattice sieve. Real attacks sieve projected blocks: rotation invariance then needs a module-aligned
  basis, whose quality cost was measured only as a proxy (rotation_prefix_proxy.py) and is not converted to bits.
- Counts only: no memory-bandwidth effects, no real implementation, equivariant sketches not benchmarked
  inside G6K. Rotating a sketch is a shift plus complement, which is cheap but not free.
- Whether the d x gain holds at Kyber's d = 256 and block dimensions 350+ is extrapolation of a trend in d.
- Sieving with automorphisms is a known idea (ideal-lattice sieving); the measurement here is the new part.


## n = 64, d = 32 (power-of-two ring), 1 instance
baseline 7.35e9 popcounts + 3.5e8 scalar products (74847 rounds); orbit 2.3e8 + 1.0e7 (2330 rounds); both reach
the saturation target (2489 short vectors; orbit finds 2560). Ratio 32.1x = 5.00 bits, memory 32x: the gain equals d.

## The aligned-prefix penalty and the NET result (rotation_prefix_proxy.py + aligned_penalty_model.py)
The orbit sieve needs an orbit-closed (rotation-invariant) prefix/sublattice. Such a prefix has a staircase
Gram-Schmidt profile. Measured on real G6K databases (power-of-two rings d = 8/16/32, degenerate zero-divisor
vectors excluded): excess of the mean log2 GS norm over an unconstrained prefix  e = c * slope * d,
slope = bits of log2 GS norm lost per index, c = 0.14 / 0.11 / 0.23 (means; per-instance range 0.00 - 0.30).

Fed into the dual-attack cost model (dual_cost_model.py; the whole sieved first-beta sublattice is orbit-closed,
dual vectors longer by 2^e, optimum re-searched), d = 256, sieve gain G = 0.9 d / 0.5 d:

| scheme | c = 0 (no penalty) | c = 0.1 | c = 0.17 | c = 0.25 | c = 0.3 |
|---|---|---|---|---|---|
| Kyber-512 (G=0.9d) | +7.8 | +0.6 | -4.4 | -10.1 | -13.8 |
| Kyber-768 (G=0.9d) | +7.8 | +0.3 | -5.0 | -11.0 | -14.6 |
| Kyber-1024 (G=0.9d) | +7.8 | -0.3 | -5.9 | -12.0 | -16.1 |

Net bits are NEGATIVE for the central measured c (about 0.17) and only non-negative at the optimistic end
(c <= 0.1). So with the plainest way of getting an orbit-closed prefix, the quality loss outweighs the
sieve speed-up in the dual attack at real parameters.

Caveats: extrapolation from d <= 32 / slope ~ 0.03 to d = 256 / slope ~ 0.01; c is noisy and may drift with d; the
"unconstrained prefix" is a greedy prefix from a sieve database, not a full BKZ basis; only the dual attack's
sublattice sieve is modelled (the primal/projected-block case enters differently and is NOT modelled); a smarter
choice of the invariant sublattice (minimal-covolume R-submodule, module-HKZ) could lower c and is untested.


## Can a smarter invariant prefix lower c? (aligned_search_proxy.py)
vol(R u) = prod_j ||sigma_j(u)|| (twisted FFT; verified against direct Gram-Schmidt) and is unchanged by units, so
candidate generators can be scored by the million. Best aligned prefix by pool, c = excess / (slope * d):

| ring d (n) | pool = 400 shortest | pool = whole sieve database | database + 2e5 random sums/differences |
|---|---|---|---|
| 16 (48) | 0.06 / 0.02 / 0.18 | 0.01 / 0.02 / 0.18 | same as database |
| 32 (64) | 0.24 / 0.30 / 0.15 / 0.22 (mean 0.23) | 0.13 / 0.08 / 0.12 / 0.16 (mean 0.12) | same as database |
| 8 (48)  | 0.07 / 0.24 / 0.00 / 0.25 | 0.07 / 0.19 / -0.39 / 0.25 | same as database |

- Widening the pool from the 400 shortest to the full database roughly halves c at d = 32 (0.23 -> 0.12).
- Random sums of two database vectors add nothing: skewed-embedding generators are not found that way.
- c is strongly instance-dependent (0.01 - 0.25 at d = 16); the attacker cannot pick the instance.
- Net bits at d = 256 (dual-attack model, G = 0.9 d, c varied): break-even near c = 0.10; Kyber-512/768/1024 =
  +4.3/+4.1/+3.8 at c = 0.05, +2.0/+1.7/+1.4 at 0.08, +0.6/+0.3/-0.3 at 0.10, -0.8/-1.2/-1.8 at 0.12.
- The measured c with the full database (about 0.12 at d = 32) sits just on the negative side of break-even.
  Real databases are astronomically larger (pool 2^80 against 3e4 here), so c may fall further, but a
  Gaussian extreme-value extrapolation is unreliable and c has a structural floor (the invariant prefix
  cannot be denser than ~GH^d).
- NOT modelled, and potentially decisive: the BOOTSTRAP cost. Finding a good invariant generator u needs short
  vectors of the full lattice, i.e. a sieve whose cost is not accelerated by the symmetry (the symmetric sieve
  needs the invariant prefix first). If that pool sieve is of comparable dimension, it can cancel the gain.


## Subgroups, granularity, and the first positive NET result (orbit_bucketed_sieve.py OB_T, subgroup_penalty_proxy.py, subgroup_net_model.py)
The lattice is invariant under every rotation SUBGROUP, y = x^(d/t) of order t. Two consequences:
1. Granularity. A rotation-invariant sublattice has rank a multiple of t. For Kyber's full ring (t = 256) the sieve
   dimension would have to be 256 or 512, while the attack optimum is ~400: rounding to 512 costs ~33 bits. Small t
   (4-32) makes beta a near-free choice.
2. Penalty and gain both depend on t.
Measured sieve gain (ring d = 16, rank 3, n = 48, G6K counting units, 2 instances): t = 4: 3.8-4.1x, t = 8: 7.3-7.9x,
t = 16: 14.2-14.5x (~0.95 t), memory /t, same stopping target and equal/more short vectors.
Measured aligned-prefix penalty (ring d = 32, rank 2, n = 64, full-database pool, 4 instances): mean c_t =
0.33 / 0.27 / 0.19 / 0.12 at t = 4 / 8 / 16 / 32, i.e. excess per vector ~ slope * (0.9 log2 t - 0.5): grows only
logarithmically in t on this data (alternative reading: constant c = 0.12-0.17, linear in t).

Net bits (dual-attack model, beta restricted to multiples of t, optimum re-searched, gain 0.95 t):

| scheme | t=4 | t=8 | t=16 | t=32 | t=64 | t=128 | t=256 |
|---|---|---|---|---|---|---|---|
| Kyber-512 (c=0.12) | +1.3 | +2.3 | +3.3 | -0.3 | -8.7 | -26.4 | -25.4 |
| Kyber-768 (c=0.12) | +1.6 | +2.6 | +1.3 | -2.4 | -10.7 | -26.9 | -27.4 |
| Kyber-1024 (c=0.12) | +1.6 | +2.6 | +1.3 | -2.4 | -10.7 | -28.4 | -27.4 |

Results for c = 0.17 and for the power-law c_t differ by <= 0.5 bits at t <= 16. Large t is dominated by GRANULARITY,
not by the penalty. Best t = 8-16: net ~ +2.3-3.3 bits (Kyber-512), ~ +2.6 (768), ~ +2.6 (1024).

Caveats: toy lattices (n <= 64, d <= 32, q = 257), c_t measured only at t <= 32, penalty assumed proportional to the
slope at real beta; the toy sieve baseline is calibrated within 1.5x of G6K counters only at n = 48; model has no
d4f/guessing/FFT, only the dual attack's sublattice sieve; BKZ preprocessing assumed to be accelerated in the same
way (it is made of sieves in aligned blocks); the primal/projected-block attack is not modelled; sieving with
automorphisms and module-BKZ granularity are known topics, only the measurements are new. A real implementation
inside G6K (equivariant sketches, subgroup-aligned tour) has not been built.


## Full aligned tour vs plain HKZ-style reduction, end to end (aligned_tour.py; numpy toy, n = 48, ring d = 8, rank 6)
PLAIN: for each position, sieve the projected lattice (bucketed sieve), insert the shortest projected vector.
ALIGNED: for each block, sieve the projected lattice with the orbit sieve (subgroup t, equivariant sketches), choose the
orbit representative whose orbit lattice has the smallest volume, insert its t rotations. BOTH reuse the database across
steps (as G6K's pump does), use the same sieve code and the same stopping rule, and reach their saturation target at
every step (checked in the per-step log). Cost = popcounts + scalar products to cover the same prefix.

| setting | prefix covered | ops plain / aligned | ratio | excess per vector at block ends (bits) |
|---|---|---|---|---|
| t = 8, seed 0 | 16 | 7.94e7 / 1.14e7 | 7.0x (2.80 bits) | 0.096, 0.059 |
| t = 8, seed 1 | 16 | 8.32e7 / 1.14e7 | 7.3x (2.87 bits) | 0.084, 0.032 |
| t = 4, seed 0 | 16 | 7.94e7 / 1.98e7 | 4.0x (2.00 bits) | 0.060, 0.064, 0.042, 0.043 |
| t = 4, seed 1 | 16 | 8.32e7 / 2.10e7 | 4.0x (1.98 bits) | 0.007, 0.035, 0.048, 0.005 |
| t = 4, seed 2 | 16 | 8.72e7 / 2.24e7 | 3.9x (1.96 bits) | 0.010, 0.029, 0.034, 0.042 |

- The end-to-end gain equals the standalone sieve gain (~t): the speed-up survives a full reduction with database reuse.
- The profile excess per vector stays roughly flat (0.03-0.06 bits at slope ~0.034 bits/index, c ~ 0.1-0.4) across 2-4
  blocks: it does not accumulate with the prefix length, matching the constant-per-vector penalty used in the cost model.
- Earlier intermediate runs that did NOT converge (171x, then 152x, then 10.7x) were artifacts: a stall rule that fired
  before saturation, a sparse sampler, and a baseline that re-sampled its database for every position. They were
  fixed before the numbers above and are listed so they are not mistaken for results.
Limits: n = 48 and 2-4 blocks only; head of the profile only (remaining dimension >= 30); plain baseline is a toy
bucketed sieve with pump-style reuse, not G6K; still no implementation inside G6K's kernel.


## Literature check (2026-10-02): the technique is known, and its BKZ-level cost is known too
- Schneider, "Sieving for Shortest Vectors in Ideal Lattices" (ePrint 2011/458; AFRICACRYPT 2013): sieve over ideal lattices
  with a practical speed-up LINEAR in the degree of the field polynomial.
- Ducas, Albrecht, Laarhoven, "Sieving for Shortest Vectors in Ideal Lattices: a Practical Perspective" (ePrint 2014/880):
  a symbolic Fourier transform (the FFT-correlation view) in a parallel Gauss sieve; 128-dimensional negacyclic ideal
  lattice solved in < 9 days on 1024 cores. The same literature notes these results are for a naive sieve and that it is
  unclear how they combine with LSH, progressive sieving and dimensions-for-free.
- Becker, Laarhoven, "Efficient (ideal) lattice sieving using cross-polytope LSH" (ePrint 2015/823; AFRICACRYPT 2016): "the
  hash of a shifted vector is a shift of the hash value of the original vector" with rerandomisation matrices that keep the
  property -> linear decrease in space AND linear speed-up of the whole (bucketed) algorithm. This is our
  rotation-equivariant sketch and our ~t x gain, already published.
- "Predicting Module-Lattice Reduction" (ePrint 2025/1904, arXiv 2510.10540): for power-of-two cyclotomic fields,
  module-BKZ needs a blocksize larger than its unstructured counterpart by d - 1 + o(1) (the question was raised as
  "Q8" in Kyber's NIST submission). This is the aligned-prefix quality cost we measured, stated as a blocksize penalty.

Consequence for the net estimate: apply that penalty to a subgroup of order t (blocksize + (t-1), i.e. 0.292 (t-1) bits at
the core-SVP slope) against the sieve gain log2(0.95 t): net = +0.6 / +0.9 / +1.05 / +1.1 / +1.05 / +0.9 / -0.45 bits
at t = 2 / 3 / 4 / 5 / 6 / 8 / 16, i.e. at most about +1 bit (t ~ 4-6). That is lower than the +2-3 bits our
model gave at t = 8-16, whose log-in-t penalty is therefore optimistic against the published blocksize result.
So: no novel bits at the sieve level; at the BKZ level the literature-grounded net is about +1 bit at best.
