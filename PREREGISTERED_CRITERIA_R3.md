# Pre-registered criteria, ROUND 3 (written 2026-10-03, BEFORE any Round-3 run exists)

Follows `PREREGISTERED_CRITERIA_R2.md`. Round 2 selected `rotall` from tuning (d = 28) and measured gross wall gains of
+1.20 / +2.09 / +3.02 bits at d = 28 / 32 / 36 on ONE lattice per d (`ideal_basis(d, 257, 1)`); at d = 40 base, `rot` and
`rotmid` all hit the 3000 s limit on 8/8 seeds while `rotall` solved 8/8 (median 346 s). No Round-2 verdict changes.
Round 3 asks three narrow questions. Variant is FIXED to `rotall` (the Round-2 selection); nothing is re-tuned.
Same success rule as R2: a run succeeds iff its shortest database vector has squared length <= L_ref^2 (1 + 1e-9), with
L_ref the minimum over all non-timed-out runs for that (d, lattice). TTS = total cost / successes.

R3a. Turn the d = 40 censored point into a measurement
- d = 40, lattice 1, base only, holdout seeds 200-203 (a subset of Round 2's seeds), 4 in parallel, timeout 17000 s.
- If >= 2 runs succeed: gain(d=40) = log2(TTS_base / TTS_rotall), TTS_rotall taken from the Round-2 d = 40 runs (seeds
  200-207). If < 2 succeed: the gain is reported only as a lower bound log2(17000 / median rotall wall), never as a credit.

R3b. Does the growth bend? (rotall only, lattice 1)
- d = 44 and d = 48, rotall, seeds 200-207, timeout 7000 s. A timed-out run is a failure charged the timeout.
- Report log2(wall of rotall) against d for d = 24..48 and the base trend fitted on d = 24..36 (and 40 if R3a measured it).
  The fitted base trend extrapolated to d = 44, 48 is an EXTRAPOLATION and is labelled so: no credit is read from it.
  The question answered is whether rotall's own wall-clock slope per d rises, stays or falls across d = 36 -> 48.

R3c. Lattice-to-lattice variance
- d in {28, 32, 36}, lattices (seed of the random ring element h) 2, 3, 4, modes base and rotall, seeds 200-207 each.
- Per (d, lattice): gain wall and gain ops_incl_clones_dense with 95% bootstrap CI over runs.
- Per d across lattices 1 (from Round 2, seeds 200-207 only), 2, 3, 4: mean and standard deviation of the point gains.
  Claim "robust across lattices" only if the gain's CI lower bound is > 0 on EVERY lattice at that d; otherwise report the
  lattices where it fails.

Reporting: GROSS only, as in R2; the aligned-basis quality cost is still not netted. No other dimension, lattice, variant,
seed set or time limit may be added after results are seen.
