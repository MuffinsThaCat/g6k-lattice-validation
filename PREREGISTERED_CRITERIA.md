# Pre-registered pass/fail criteria for the G6K validation

Written BEFORE any G6K result exists (2026-10-02). Purpose: decide in advance how each measurement
changes the paper's credits, so the outcome cannot be fitted to a preferred answer afterwards.

General rules
- A credit survives only as G6K vs G6K+technique. A credit measured only against this paper's own
  numpy sieve is a numpy-baseline result and is not carried over.
- Operation counts (G6K's `xorpopcnt_total`, `fullscprods_total`, ...) are the primary metric; they
  are hardware independent. Wall-clock on shared CI runners is reported but never used to set a credit.
- A measured value REPLACES the paper's credit (it is not added to it). If a result is missing,
  inconclusive, or a run failed, the credit is reported as "unmeasured", not kept silently.
- Seeds and block sizes are reported in full; results with fewer than 3 seeds per setting are marked
  low-power. Multi-thread runs with a data-race fraction above `G6K_RACE_WARN` are discarded for
  saturation/termination questions.
- The smoke step must print non-zero `xorpopcnt_total` and `fullscprods_total`; otherwise nothing
  below is interpretable.

A. Termination credit (paper: 3.943 b, Table `tab:components`; E2 in `g6k_opcount_vs_walltime.py`)
- Metric: log2(ops at saturation 0.5 / ops at saturation s) for s in {0.3, 0.7, 0.9}, same lattices
  and seeds, beta = 48 (and 52/56 if time allows).
- Valid only if "same best vector" is >= 90% of runs; otherwise the early stop changed the answer
  and the credit for that s is void.
- Decision: measured credit = largest valid gain over default. If it is < 0.5 b, the paper's
  3.943 b credit is replaced by the measured value (possibly ~0).
- G6K default stopping (saturation 0.5, radius sqrt(4/3)) is the baseline, not run-to-completion.

B. Throughput credits (kernel fusion 0.79, tensor core 0.35, streaming BKZ 2.02, speculative BKZ 0.6)
- Each credit is classified as operation-count or time-only from the paper's own definition.
- A time-only credit counts 0 b toward gate-count floors unless G6K shows an operation-count
  reduction from the same technique. E1 (ns per op) is used to state how much of the numpy-vs-G6K
  gap is per-operation speed (a time effect) and not an operation-count effect.

C. Sieve symmetry / ring rotation (paper: 1.6 b Module-LWE, 2.2 b Falcon-512)
- `rotation_orbit_check.py`: counts of short vectors found vs closure under rotation. Pass = the
  orbit-closed count matches the expected short-vector count within 15%.
- `rotation_sieve_ab.py` (needs `rotation_kernel.patch`): credit = log2(op-count ratio) of
  base vs rotate-short at equal saturation. Memory-only gains are labelled memory-only and are not
  gate-count credit. If no op-count reduction >= 0.5 b is seen at any d, the gate-count credit stays at
  the paper's existing figure only if the existing derivation is independently supported; otherwise 0.

D. Dimensions for free on a planted vector (`g6k_d4f_planted.py`; paper uses Ducas's formula)
- Metric: f_obs = largest f with the planted vector found, at the block size where
  rho = ||pi(v)||/gh <= 0.97 (threshold regime). Excess = f_obs - f_Ducas(beta_block).
- Credit rule: a d4f credit is considered ONLY IF mean excess >= 2 dims at beta_block >= 55 AND the
  excess slope against beta is not negative across the measured range. Then credit = 0.292 b per
  dimension x mean excess, capped at the measured value (no extrapolation past the largest beta).
  Otherwise the credit is 0.
- Instance sizes are toy (beta_block ~ 50-80). The paper must say the credit is toy-scale and
  not extrapolated to beta 400-900.

E. Reporting
- Each result states: what was measured, seeds, beta, machine, G6K commit, and what was NOT
  measured. Results that lower a credit are reported with the same prominence as results that confirm.
- If E2 shows the termination credit collapses, the abstract's totals and verdicts are recomputed
  from the measured credits (`round47_propagate.py` holds the credit constants) before anything is
  claimed about Kyber/Dilithium margins.
