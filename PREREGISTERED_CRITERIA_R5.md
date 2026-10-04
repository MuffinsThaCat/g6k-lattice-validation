# Round 5: c (baseline chances) and the rotation saving under G6K's real schedule

Written 2026-10-03, before any Round-5 Actions run. Workflow: `.github/workflows/g6k-round5-c.yml`.

## Question
Round 47 of the paper credits rotation-amplified decoding using c = 16 baseline chances, never measured.
A local fpylll measurement at d = 128 (100 instances per set) gave c_eff = 2.07 (95% CI 1.59-2.70, n64k1) and ~2.5
(n32k2). Does c stay small (a) under G6K's pump-and-jump schedule instead of fpylll BKZ, and (b) at larger
dimension and block size?

## Method (fixed)
Per instance (fresh A, fresh secret; module-LWE over Z_q[x]/(x^n+1), rounded-Gaussian s, e), on the same A:
- E: Kannan-embedded target, progressive G6K pump-and-jump BKZ (jump 1, one tour per block size, every block size
  from 10, all tours through G6K: G6K_FPLLL_BELOW=0); success = +-(e, -s, M) is a basis row. Stops at success.
- T1 / Tany: target-free lattice, same schedule; after each block size, norm event (projection of the displacement
  onto the last beta Gram-Schmidt directions <= ||b*_{d-beta}||) for view 0 (T1) and for any of the n negacyclic
  rotations (Tany). Stops once view 0 has hit and the block size has reached E's success block size.
- Readout: cumulative success curves, beta at 50% for E, T1, Tany; c_hat(E) by maximum likelihood under
  P(found by beta_j) = E[1 - prod_{i<=j}(1-F_i)^c], F_i = Beta(beta_i/2, (d-beta_i)/2) law on the target-free profile;
  the same fit on T1 (true c = 1 event) is the control, and c_eff = c_hat(E) / c_hat(T1), bootstrap 95% CI (200).
  Script: `aggregate_c_jsonl.py`.

## Sets
| tag | n | k | m | d | q | sigma | rotations | seeds | float |
|---|---|---|---|---|---|---|---|---|---|
| d128a | 32 | 2 | 64 | 128 | 3329 | 4.0 | 32 | 5000-5031 | double |
| d128b | 64 | 1 | 64 | 128 | 3329 | 4.2 | 64 | 5000-5031 | double |
| d160 | 32 | 3 | 64 | 160 | 3329 | 2.0 | 32 | 7000-7031 | double |
| d256 | 128 | 1 | 128 | 256 | 3329 | 2.0 | 128 | 9000-9031 | dd |
d128a/b use the same instances as the local fpylll runs (same generator and seeds), so tool is the only difference.

## Decision rules (recorded; the paper owner decides what goes in the paper)
- A set counts only with >= 20 finished instances and E found in >= 80% of them.
- If c_eff's 95% CI upper bound < 16 at every counted set: the adopted c = 16 is conservative; the credit is
  recomputed at the largest counted c_eff (not the smallest), via `joint_ticket_estimator.py`.
- If any counted set's CI contains 16 or lies above: keep c = 16.
- Trend: report c_eff against d; if it rises with d, say so and do not extrapolate a small c to NIST sizes.
- All sets reported, including failed or timed-out ones.

## Caveats that go with any number
Block sizes stay far below the attack range (400-900); G6K here is single-threaded, default parameters; the norm
event, not exact decoding, is used for T1/Tany (Round 47 measured exact decoding 6-15% below the norm event).
