# Evidence inventory: what the lattice paper claims, what has been checked against a real sieve, what has not
(2026-10-03. Scope: auditing the paper's EXISTING credits. This document does not propose new attacks.)

## 1. What the paper claims
Module-LWE total 2^9.77 = 8.17 (real timed eight-axis composition, capped at beta <= 24, numpy baseline)
                        + 1.6 (ring rotation, formula n^(2/3)/log2 d)
                        + 0.0 (MATZOV credit, retracted in Round 42).
The paper itself states that ~84% of the total rests on the beta <= 24 test while the attacks target beta 420-856, and
reports Kyber-1024 / Dilithium-2 / Dilithium-5 as "undetermined", not "Survives".

## 2. What has been checked against a REAL sieve (G6K master, built on ARM 2026-10-02; counts, not x86 times)
| Item | Result | Supports the paper? |
|---|---|---|
| G6K builds and runs (ARM, patched); counters need -DENABLE_STATS in CXXFLAGS | yes; E1 slope ~0.42 ops-bits/beta (beta 40-58) | infrastructure |
| Termination via saturation ratio (0.5 -> 0.1) | 0.03-0.1 bit | not the same knob as the paper's credit |
| Approx-SVP plateau termination, oracle stop, G6K-like bucketed sieve (calibrated within 1.5x of G6K counters) | 0.2-0.5 bit (alpha 1.02-1.2), best norm appears at 70-88% of the ops | SUPPORTS the original 0.5-bit credit; does NOT support the 3.6-4.1 figure |
| Ring-rotation structure in G6K output | closure of the found short set completes saturation (249 -> 504 vs ~498 expected) | supports the mechanism |
| In-sieve rotation cloning | ~0.1 bit (same output), 9-22x counted ops but ~half the orbits | small |
| Dual-distinguisher rotation amplification | rotation samples independent (null variance 0.95-1.06); effective sample gain 2.3-7x | mechanism supported; Round 47 credit (1.39-1.84 bits) corresponds to the theoretical maximum g = 256 in a simple model; measured gain gives ~0.3-0.6 |
| Orbit (symmetric) sieve, toy | ops gain ~t at subgroup order t; quality cost of an aligned prefix measured | known technique (below); BKZ-level net <= ~1 bit, already inside the 1.6 |

## 3. Published work found (literature check)
- Schneider 2011/458: sieve over ideal lattices, practical speed-up linear in the degree.
- Ducas-Albrecht-Laarhoven 2014/880: symbolic Fourier transform sieve; 128-dim negacyclic record.
- Becker-Laarhoven 2015/823: rotation-equivariant LSH, linear space and time speed-up (same as the equivariant sketch used here).
- ePrint 2025/1904 (arXiv 2510.10540): for power-of-two cyclotomics module-BKZ needs blocksize larger by d-1+o(1).
Not yet searched: literature on the dual-decoding rotation amplification (Round 47).

## 4. Gaps: what is NOT yet verified (each is a measurement or a check, not a new technique)
1. Composition of the 8.17: which axes it contains and how much of it each carries (termination, streaming, batching, kernel fusion,
   precision, compression, ...). Only termination and the rotation items were tested against a real sieve. The throughput-type
   items (batching, fusion, precision, compression) were never compared with a compiled, AVX2 sieve.
2. x86 / AVX2 timings: every real-G6K number above is an operation count from an ARM build. Wall-clock constants are unmeasured.
   The staged GitHub workflow (smoke mode first) is the prepared way to get them.
3. Scale: largest real sieve here is dimension ~64; attacks are at beta 400+. All extrapolations in d and beta are untested.
4. Model fidelity: the dual-attack cost model has no dimensions-for-free, key guessing, FFT or distinguisher cost; the primal
   variant of rotation amplification is not modelled.
5. The Ducas-Pulles regime (large q, real parameters) for the independence heuristic: tested only at q = 257 and 769.
6. Whether the 1.6-bit ring-rotation formula matches measurement: the formula gives n^(2/3)/log2 d, earlier sieve-symmetry tests
   measured list shrink ~n; the BKZ-level net with the known module-BKZ penalty is <= ~1 bit.
7. No implementation inside G6K's kernel exists for any of the symmetry variants.

## 5. Sensitivity of the paper's verdict margins (constants from round47_propagate.py, ADD headline)
margin = BASELINE - (9.77 + ISO + R); negative = credits exceed the baseline.
| case | K-512 | K-768 | K-1024 | D-2 | D-3 | D-5 |
|---|---|---|---|---|---|---|
| paper as is | -5.66 | -6.09 | -0.51 | +1.29 | -3.60 | -0.77 |
| ring rotation 1.6 -> 1.0 | -5.06 | -5.49 | +0.09 | +1.89 | -3.00 | -0.17 |
| ring rotation -> 0 | -4.06 | -4.49 | +1.09 | +2.89 | -2.00 | +0.83 |
| + termination 3.94 -> 0.1 (ASSUMES that share sits inside the 9.77; unverified) | -1.22 | -1.65 | +3.93 | +5.73 | +0.84 | +3.67 |

## 6. Suggested verification order (cheapest first)
1. Trace the 8.17 composition: list its axes and map each to a measurement or a "not measured" tag.
2. Run the staged workflow in smoke mode on x86 (needs `gh` login fixed), then full mode, to get real timings.
3. Add a short "real-sieve reconciliation" section to the paper with sections 2-5 above.
4. Search the literature for the Round 47 rotation-amplified decoding mechanism.
