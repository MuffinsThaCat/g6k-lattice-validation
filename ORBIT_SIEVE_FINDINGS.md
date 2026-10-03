# Orbit (symmetric) sieve prototype: operation-count measurements (2026-10-02, numpy toy, not G6K)

Script: `orbit_sieve_proto.py`. Same naive pass rule in both sieves (best reduction per vector, replace the
longer). plain: N vectors, N(N-1)/2 pair tests per pass. orbit: R = N/d representatives, each standing for its
d rotations; a rep pair is tested through the d inner products <u_a, x^r u_b> (a cyclic cross-correlation,
an FFT in a real implementation; computed directly here, only the NUMBER of pair classes is counted).

Fixed n = 48, default database size N ~ 3.2 (4/3)^(n/2) = 3189, 1 instance each:

| ring d (rank) | stored plain / orbit | pair tests plain / orbit | gain | shortest (both) | short vectors plain / orbit |
|---|---|---|---|---|---|
| 8 (6)  | 3189 / 398 | 6.66e8 / 6.27e7 | 10.6x = 3.41 bits | 1.050 GH | 323 / 368 |
| 12 (4) | 3189 / 265 | 6.30e8 / 4.72e7 | 13.4x = 3.74 bits | 1.079 GH | 356 / 376 |
| 16 (3) | 3189 / 199 | 6.25e8 / 3.20e7 | 19.5x = 4.29 bits | 1.050 GH | 447 / 448 |
| 24 (2) | 3189 / 132 | 6.51e8 / 2.13e7 | 30.5x = 4.93 bits | 1.066 GH | 485 / 528 |

Plain sieve with the SAME small memory (R vectors) does not converge (2.6-3.0 GH, 0 short vectors).
n = 64, d = 32 (plain baseline too large to run): 995 reps converge to 1.019 GH with 4608 of ~4977 expected
short pairs; estimated gain ~32x (plain tests = passes x N(N-1)/2).

Findings:
- A database of N/d orbit representatives converges as well as the full N-vector database (same shortest
  vector, as many or more short vectors), with d x less memory and ~1.2 d x fewer pair tests. The gain
  tracks the ring dimension: 3.4 -> 4.9 bits for d = 8 -> 24, i.e. ~log2(1.2 d).
- If it persists, the ceiling is ~log2(d) + 0.3 bits in pair classes and log2(d) in memory: about 8 bits at
  d = 256, from the symmetry alone.

NOT established (read before quoting):
- Baseline is a naive all-pairs sieve. G6K's bgj1 buckets and filters (xor-popcount sketches), so its pair
  count is far below N^2; an orbit version needs an orbit-aware bucketing/filter that does not exist here.
- Per-class cost: an FFT correlation costs ~ n log d / d flops per class (~140 at d = 256, n = 768) versus
  ~25 ops for a popcount-filtered test, so against an optimised baseline the net is nearer 5-6 bits at d = 256.
- Full-lattice sieve only. BKZ sieves projected blocks: rotation invariance needs a module-aligned basis,
  whose quality penalty grew with d (rotation_prefix_proxy.py) and is not converted to attack bits.
- Toy dimensions (n <= 64), 1 instance per setting, ideal/module q-ary lattices, q = 257.
- This TYPE of speedup (sieving with automorphisms in ideal lattices) is known in the literature; what is
  new here is only the measurement. Check prior work before claiming novelty.
