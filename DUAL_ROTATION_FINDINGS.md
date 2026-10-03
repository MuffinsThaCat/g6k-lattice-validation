# Rotation amplification of a dual distinguisher, measured on real G6K sieve output (2026-10-02, ARM, toy)

Script: `dual_rotation_amp.py`. Toy module-LWE (rank k=2 secret, m=3 ring samples, dual dimension 3d), dual
lattice sieved with G6K bgj1, distinguisher S = sum cos(2 pi <w,b>/q) over the database vectors vs over their
rotation closure (all x^i w). d' = (mean_LWE - mean_null)/std_null, 1500 trials each. "Sample gain" =
(d'_closure / d'_db)^2, each with its own best length cut-off.

| setup | db vectors | closure | sample gain | null variance / independent value |
|---|---|---|---|---|
| d=8,  q=257 | 102 | x8.0 | 3.8 - 7.0 | 0.95 - 1.01 |
| d=12, q=257 | 568 | x2.5 | 2.3 - 2.4 | 0.96 - 1.05 |
| d=16, q=257 | 3190 | x7.9 | 6.5 - 6.9 | 0.95 - 1.06 |
| d=20, q=257 | 17919 | x6.3 | 4.4 - 5.4 | 0.98 - 1.03 |
| d=16, q=769 | 3190 | x4.1 | 3.4 - 4.1 | 0.97 - 1.04 |

What is established (at these toy sizes):
- The d rotations of a dual vector behave as independent samples: null variance equals the
  independent-samples value (0.95-1.06), and measured d' matches the independent model within a few %.
- Effective sample gain is 2.3-7x (1.2-2.8 bits of sample count), equal to the closure multiplier weighted by
  bias. It is below d because a saturated sieve database already holds a large share of each short orbit.
- It needs NO module-aligned basis: rotation acts on full dual vectors (an isometry of the whole lattice),
  so the aligned-prefix penalty of the in-sieve idea does not apply. Plain LLL+BKZ bases were used.

What is NOT established:
- Attack bits. Sample-count bits are not cost bits: a real dual attack trades sieve dimension, vector
  length (bias), key guessing and the distinguisher phase against each other. The cost model must be
  re-optimised with the sample supply multiplied by the measured gain. Not done.
- Large dimension. The closure multiplier should rise toward the ring dimension as the database becomes a
  sparser sample of longer vectors, but that is unmeasured (largest here: dual dimension 60).
- The regime where Ducas-Pulles found the independence heuristic failing (large q, many vectors, real
  parameters). The mean-bias formula held here (q = 257 and 769), but this is a small test.
- Distinguisher cost: using d x more vectors raises evaluation work (one ring product per w gives all d
  samples, so roughly a log d factor per sample).
- Real Kyber/Dilithium: ring dimension 256, rank 2-4, centred-binomial error, FFT/guessing structure.


## Converting sample gain to attack bits: `dual_cost_model.py` (simple model, re-optimised for each g)
Standard sieve-based dual attack, GSA profile, no dimensions-for-free, no key guessing, no FFT, polynomial
factors dropped. Baselines (g = 1): Kyber-512 116.2 bits (beta 398), Kyber-768 188.9 (beta 647),
Kyber-1024 263.7 (beta 903), in the plausible range of published dual core-SVP figures.

Bits saved when the sieve's supply of dual vectors is multiplied by g (optimum re-searched over beta, m):

| scheme | g=2 | g=4 | g=7 | g=16 | g=64 | g=256 |
|---|---|---|---|---|---|---|
| Kyber-512 | 0.2 | 0.3 | 0.6 | 0.9 | 1.2 | 1.8 |
| Kyber-768 | 0.3 | 0.6 | 0.6 | 0.9 | 1.2 | 1.8 |
| Kyber-1024 | 0.0 | 0.3 | 0.5 | 0.6 | 1.2 | 1.5 |

Reading:
- g = 256 is the theoretical maximum (ring dimension 256 caps the orbit size). It gives only 1.5-1.8 bits,
  in the same range as the paper's Round 47 credit (1.39-1.84 bits): that credit is the full-d case.
- The measured toy gains (g about 2-7) correspond to about 0.3-0.6 bits.
- Why so little: the sample requirement N_req = z^2/(2 eps^2) rises ~1.3 bits per sieve dimension removed
  (the bias eps collapses as vectors get longer) on top of the 0.2075 bits/dim of supply, so each bit of
  supply gain buys only ~0.2 bits of attack cost.
- Not in this model: dimensions-for-free, secret guessing and FFT (MATZOV-style), the distinguisher's own
  cost, memory. These could change the sensitivity in either direction. Unverified: whether the closure
  multiplier actually approaches 256 at real sieve dimensions (toy: 2.5-8x of a possible 8-20x).
