# Pre-registered criteria, ROUND 2 (written 2026-10-03, BEFORE any Round-2 run exists)

Round 1 (`PREREGISTERED_CRITERIA.md`) is unchanged and its outcomes stand: termination credit measured ~0
(replaces 3.943 b), rotation `rot-short` op-count credit 0.15-0.19 b (< 0.5 b), item D unmeasured (script crashed).
Round 2 does NOT reopen those verdicts. It tests three NEW, separately defined questions. Same general rules as
Round 1 (a measured value replaces a paper credit, missing = "unmeasured", all seeds reported, results that lower a
credit get the same prominence).

## C'. Rotation, clone-everything family: time to solution including cloning work

Why a new test: Round 1 measured `rot-short`, which by construction cannot gain much. The clone-everything variants
showed a wall-clock gain that grows with n (spec table, ARM), but (i) the `ops` counter left out clone work,
(ii) it reaches the saturation COUNT with fewer orbits, and (iii) "same output" was not checked. C' fixes all three.

Design (fixed in advance)
- Instance: NTRU-type ideal lattice from `rotation_orbit_check.ideal_basis(d, 257, 1)`, d in {24, 28, 32, 36, 40}
  (n = 2d = 48..80). bgj1, db_size_factor 3.2, saturation 0.5 (G6K default), 1 thread per run, 4 runs in parallel.
- Variants: `base`; `rot` (clone vectors shorter than the saturation radius); `rotmid` (clone vectors shorter than
  1.5 x the saturation radius); `rotall` (clone every new vector).
- Seeds: TUNING seeds 100-107 at d = 28 only. HOLDOUT seeds 200-215 at d = 24, 28, 32, 36; 200-207 at d = 40.
  Tuning and holdout are different seeds; the chosen variant is picked from tuning only.
- Variant selection (done by `rotation_tts_analyze.py`, not by hand): among {rot, rotmid, rotall}, the variant with
  the smallest tuning time-to-solution (wall) at d = 28; ties -> the one cloning less (rot < rotmid < rotall).
  No other variant, parameter or seed set may be added after seeing holdout data.
- A run SUCCEEDS iff the shortest vector in its final database has squared length <= L_ref^2 (1 + 1e-9), where L_ref(d)
  is the minimum over every non-timed-out run at that d (all variants, all seeds). Rationale: an attack needs the
  shortest orbit, not all orbits; a variant that skips the shortest orbit fails. A run that exceeds 3000 s is a
  failure and is charged 3000 s.
- Time to solution (TTS) of an arm = (sum of cost over all HOLDOUT runs of the arm) / (number of successes);
  infinite if none (the standard restart estimator). Costs: `wall`; `ops_incl_clones_dense` (counted ops + every
  clone built x (n/2 + n): the patch as implemented, dense rotation matrix); `ops_incl_clones_opt` (counted ops +
  clones x (n/2 + 1): a cyclic-shift kernel, a MODEL of a kernel nobody has built).
- Gross credit at d: g(d) = log2( TTS_base / TTS_chosen ), for each cost; 95% CI by bootstrap over runs (2000
  resamples per arm). A credit is claimed at d only if the CI lower bound is > 0.
- Reported credit: wall-clock gain = point estimate at the LARGEST d where the wall CI lower bound > 0 (no
  extrapolation past d = 40). Gate-count gain = the same using `ops_incl_clones_dense` (measured as implemented);
  `ops_incl_clones_opt` is reported as "potential with an optimised kernel", never as a measured credit.
- Growth: least-squares slope of g(d) against d over d >= 28 (holdout). Slope <= 0 -> labelled "non-growing" and the
  credit is capped at its d = 28 value.
- Gate-count credit needs >= 0.5 b in the dense metric; below that it is 0 (as in Round 1, rule C).
- NET vs GROSS: this round measures the sieve only. The cost of keeping a module-aligned basis (ROTATION_KERNEL_SPEC.md,
  prefix proxy: +0.12 to +0.18 b per vector, growing with d) is NOT netted. Any credit here is GROSS. Gross credit
  enters the paper only as an upper-bound column, not in the headline totals, until a module-BKZ comparison exists.
  (Recorded as the author's default; the paper owner may change this rule BEFORE results are read.)
- Not measured, not to be claimed: BKZ projected blocks of Kyber/Dilithium, d = 256 rings, d > 40, other q,
  multi-thread timing.

## E'. The unknown c behind the Round 47 lift-amplification credit

Question: how many independent chances does the standard progressive-BKZ baseline already get? Round 47's estimate is
tabulated for c = 1 (credit 2.39/2.72/3.02 b Kyber, 1.90/2.20/2.41 Dilithium) and c = 16 (1.54/1.80/2.04, 1.18/1.37/
1.52); the paper adopts c = 16 x 0.90 without a measurement.

Design (`measure_c_baseline.py`, fpylll only)
- Planted LWE (Kannan embedding), q = 97, CBD(3) secret and error, n = 40, m in {28, 32, 36, 40, 44}, seeds
  11..58 per m. Progressive BKZ: block size 10, 12, ..., 40, two tours per block size, `BKZ.DEFAULT_STRATEGY`.
- AMENDMENT (2026-10-03, after a 4-seed calibration scan, before the real run): the first draft said CBD(1),
  m in {28..54}, up to block 44, seeds 1..32. A scan (eta in {2, 3}, m in {24, 32, 40}, seeds 1-4, block <= 34) showed
  CBD(1) is solved at block size 10-12 (no information) and eta = 3, m = 28-44 puts the transition at block 14-30.
  Parameters above are the amended ones; calibration seeds 1-4 are excluded from the real run.
- After each block size beta: recovered_i(beta) = the planted vector (+-) is a row of the reduced basis (cumulative over
  all earlier stages: this is the whole baseline, every position, every tour). p1_i(beta) = the model probability of
  ONE view at beta from the REAL Gram-Schmidt profile: P[V^2 X <= r_{d-beta}], X ~ Beta(beta/2, (d-beta)/2),
  V = ||planted||.
- P_full(beta) = mean recovered_i; P_1(beta) = mean p1_i.  c_eff(beta, m) = ln(1 - P_full) / ln(1 - P_1), computed
  only where both are in (0.03, 0.97). Bootstrap 95% CI over instances.
- Reported: c_eff for every (m, beta) that qualifies, and the median. The adopted c for the credit is the median
  c_eff over qualifying cells; if fewer than 3 cells qualify, c is "unmeasured" and the paper's c = 16 stays labelled
  unmeasured.
- Mapping to credit: log-linear interpolation of the Round 47 credit between the c = 1 and c = 16 rows
  (clamped; c outside [1, 16] uses the nearest row, with the CI shown).
- Caveat that must accompany any number: toy block sizes (beta <= 44), plain LWE (the rotations are not exercised);
  whether c at beta ~ 400-900 equals c here is not measured.

## D'. Planted-vector dimensions for free (re-run of item D)

Round 1's run crashed in fpylll ("infinite loop in babai") during pre-reduction. Fix = higher floating-point
precision (`long double`, falling back to `mpfr`) in pre-reduction. Criteria are exactly Round 1 item D. If the run
fails again the credit is 0 and reported "unmeasured". This is plain LWE, so rotation closure is NOT applicable;
a rotation-closure variant of D needs a ring-LWE planted instance and is out of scope for this round.
