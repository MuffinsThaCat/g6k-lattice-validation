# Pre-registered criteria, ROUND 4: does rotation survive dimensions-for-free? (written 2026-10-03, BEFORE any R4 run)

Structural point that motivates this round (an argument, not a measurement). A rotation-augmented sieve needs the sieve
context [l, r) to be invariant under the rotation, which needs the Q-rational prefix span(b_0..b_{l-1}) to be invariant.
For R = Z[x]/(x^d+1) with d a power of two, Q[x]/(x^d+1) is a field, so an invariant rational prefix has dimension a
multiple of d. Dimensions-for-free (d4f) sieves [l+f, r) with f about 36-63 at Kyber-size blocks, not a multiple of d = 256.
So full-ring rotation and d4f do not stack: using the full ring costs the d4f gain (0.292 bits per free dimension, roughly
10-18 bits at beta 400-850), more than the ring's gross ceiling log2(d) = 8 bits.
Escape route: a SUBGROUP <x^(2d/g)> of order g. Its field Q(zeta_g) has degree g/2 (g a power of two), so the prefix only
has to be a multiple of G = g/2 dimensions, and d4f can be kept up to rounding f down to a multiple of G.

Quantities (all stated now):
- Gross gain G(g) = log2(TTS_base / TTS_rotall,g) in wall clock, d = 32 (n = 64), lattice seed 1, holdout seeds 200-215,
  `rotall` with the subgroup of order g in {2, 4, 8, 16, 32}; g = 64 (the full group) and base are taken from Round 2
  (same lattice, same seeds 200-215). Same success rule and L_ref rule as R2 (shortest database vector == best known).
- Efficiency eps(g) = G(g) / log2(g/2)  (the symmetry group acts on +- classes with g/2 elements; ceiling log2(g/2)).
- d4f alignment loss A(g) = 0.292 x (g/2) / 2 bits = 0.146 (g/2): f is rounded DOWN to a multiple of G = g/2 (mean
  rounding error G/2 dims at 0.292 bits per dimension). This is a MODEL for the cost of keeping a valid aligned prefix.
- Net before prefix quality  N(g) = G(g) - A(g).
- g = 2 is a NULL CONTROL: the group is {+-1}, which the +- canonicalisation already identifies, so G(2) must be <= 0
  (clone overhead only). If G(2) has a CI lower bound > 0 the harness is giving gains it should not and nothing from
  this round is trusted.
Decision rules
- A subgroup gain is claimed only where the CI lower bound of G(g) is > 0 (95% bootstrap over runs, as in R2).
- Reported: G(g), eps(g), A(g), N(g) for every g, and g* = argmax N(g). If max N(g) <= 0 the rotation credit under d4f is 0.
- N(g*) is a TOY-SCALE (n = 64) result: it is not extrapolated to n ~ 350 or to d = 256. It also does NOT include the
  aligned-prefix quality cost (ROTATION_KERNEL_SPEC.md proxy: +0.01 to +0.06 log-volume bits per prefix vector at small
  d; conversion to cost bits unresolved). Any credit carried to the paper is labelled "before prefix-quality cost".
- No other group order, dimension, lattice or seed may be added after results are seen.
