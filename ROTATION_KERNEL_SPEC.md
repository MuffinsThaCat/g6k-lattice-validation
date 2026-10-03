# Rotation database inflation inside G6K: IMPLEMENTED as rotation_kernel.patch (tested on ARM)

`rotation_kernel.patch` (applies to G6K master with `patch -p1`) adds `Siever.set_rotation(M, order,
clone_below)`: whenever bgj1 creates a new database vector shorter than `clone_below` it also queues the
images x*M^k (k < order). Clones go through G6K's own uid dedup and transaction replacement, and short
clones count toward the saturation target. `rotation_sieve_ab.py` is the A/B harness.

## Measured A/B (ARM, bgj1, NTRU-type ideal lattice, db_size_factor 3.2, 3 seeds; counts, not x86 times)
Ops = xorpopcnt + fullscprods; cloning work is NOT in `ops`, only in wall time.

| d (n) | requested sat | mode | reached target | ops | wall | orbits found (of all) |
|---|---|---|---|---|---|---|
| 24 (48) | 0.5 | base | 3/3 | 1.09e8 | 0.42 s | 21 (of 21) |
| 24 | 0.5 | rotate short only | 3/3 | 0.96e8 | 0.38 s | 11 |
| 24 | 0.5 | rotate every new vector | 3/3 | 5.5e6 | 0.25 s | 11 |
| 24 | 1.0 | base | 0/3 (397/498) | 1.25e8 | 0.49 s | 21 |
| 24 | 1.0 | rotate short only | 3/3 (504) | 0.97e8 | 0.38 s | 21 |
| 24 | 1.0 | rotate every new vector | 0/3 (408) | 1.4e7 | 0.28 s | 17 |
| 26 (52) | 0.5 | base / short / every | 3/3 each | 3.5e8 / 3.2e8 / 1.6e7 | 1.25 / 1.16 / 0.61 s | 30 / 17 / 17 |
| 20 (40) | 0.5 | base / short / every | 3/3 each | 1.0e7 / 0.94e7 / 0.75e6 | 0.056 / 0.051 / 0.051 s | 9 / 4 / 4 |

Findings, stated plainly:
- Rotating only short vectors is the safe variant: same orbit coverage as base, ~10-22% fewer ops and
  wall time, and it can complete saturation that base cannot.
- Rotating every new vector cuts counted operations 9-22x and wall time ~1.7-2x, but the saturation
  COUNT is then reached with fewer independent orbits (about half at 0.5). Per orbit found it is
  ~7-10x cheaper; it is NOT a like-for-like speedup of a full run. Cloning cost is in wall time only.
- A smaller database (db_size_factor 1.6 / 1.2) still fails for every mode: no short vector is ever
  produced, so the hook never fires. The hook does not let the sieve survive on a smaller database.
- Applicability limit: the automorphism holds for the WHOLE ideal lattice (full-rank sieve, l = 0). BKZ
  sieves projected local blocks, where the rotation is generally not an automorphism. Kyber/Dilithium
  are module lattices and need the per-block rotation. Nothing here is measured on those.

## Scaling with dimension (ARM, NTRU-type ideal lattice, q=257, 2-3 seeds, matched = same orbits as base)
Gain = log2(base cost / variant cost). Wall includes cloning; ops does not. Coverage = orbits found / base orbits.

| n | safe variant (clone short only): wall bits | clone-everything: wall bits | ops bits | coverage | coverage-adjusted wall bits |
|---|---|---|---|---|---|
| 40 | -0.04 (best matched) | +0.25 | +2.6 | 40% | -1.07 |
| 48 | +0.15 | +0.59 | +3.0 | 81% | +0.29 |
| 56 | +0.03 | +1.47 | +3.7 | 78% | +1.11 |
| 60 | +0.11 | +2.00 | +3.85 | 66% | +1.40 |
| 64 | (not run) | +2.15 | +4.0 | 76% | +1.75 |

- Safe variant: ~0 to 0.15 bit, NOT growing with n. This is the only variant with the same output as the base sieve.
- Clone-everything: wall gain rises 0.25 -> 2.15 bits over n = 40 -> 64 (flattening between 60 and 64), but it finds
  only 66-81% of the orbits. "Coverage-adjusted" = wall bits - log2(1/coverage), assuming independent restarts
  recover the missing orbits (UNVERIFIED assumption).
- Ceiling: the symmetry group has order d on the +- classes, so the gain is bounded by about log2(d) bits
  (5 bits at d = 32; ~8 for a d = 256 ring). Observed 2.15 of 5 at n = 64. A polynomial factor, not a slope change.
- Not measured: BKZ projected blocks (rotation is not an automorphism there), module lattices, any n above 64,
  other q or ring shapes, x86 timings. Do not extrapolate these to Kyber/Dilithium block sizes.

## Projected block (what BKZ sieves): rotation_block_test.py, rank-3 module lattice, ARM, 2-3 seeds
Basis rows x^i*g1 (exact signed rotations, NOT reduced mod q) then q-vectors; the prefix R*g1 is exactly
rotation-invariant, so the block [d, 3d) projected orthogonally to it is a rotation-invariant lattice.
Checked numerically first: rotation preserves projected norms (relative error 2e-10 to 2e-9).

| block dim | safe variant wall bits | clone-everything wall bits | ops bits | orbits found (of base) |
|---|---|---|---|---|
| 40 | +0.08 | -0.08 | +2.18 | 6.0/9.0 |
| 48 | +0.04 | +0.66 | +3.07 | 12.0/19.0 |
| 56 | +0.01 | +1.46 | +3.58 | 41.0/53.0 |

Same numbers as the full-rank case at equal dimension: the symmetry and the gain survive projection when
the context starts at a ring-block boundary of a module-aligned basis.
NOT measured: the reduction-quality cost of keeping a module-aligned basis (only a local LLL is used here),
Kyber/Dilithium's real parameters (rank 2-4, d = 256), and any block above 56.

## Quality cost of an aligned prefix (rotation_prefix_proxy.py; PROXY, real G6K databases, ARM, 2 seeds)
An aligned basis starts with the d rotations of one vector v; a normal basis starts with d independently
chosen short vectors. From the same sieve database: best aligned prefix vs greedy unconstrained prefix.

| ring d, rank K (n) | aligned minus unconstrained, per vector | total over the first ring block |
|---|---|---|
| 12, 4 (48) | +0.03 to +0.06 bits | 0.3 - 0.8 bits |
| 16, 3 (48) | +0.01 to +0.03 | 0.1 - 0.5 |
| 20, 3 (60) | +0.12 to +0.15 | 2.4 - 3.0 |
| 24, 2 (48) | +0.12 to +0.18 | 3.0 - 4.3 |

The penalty is small for small d and GROWS with the ring dimension (and with d/n). The symmetry gain is
bounded by ~log2(d) bits, so the two scale differently; at Kyber's d = 256 the penalty could outgrow the
gain. This is log-volume of the basis prefix, not attack-cost bits: converting it needs a full module-BKZ
comparison, which does not exist yet. Net attack bits remain UNPROVEN and the new evidence leans negative.

(Original design notes follow.)

## What real G6K runs showed (ARM build of G6K master, bgj1, d = 20/24/26, 2026-10-02)
- The mechanism is real. G6K's sieve at saturation 0.5 finds about half of the ~(4/3)^(n/2)/2 short
  +-pairs, with no rotation bias, and closing them under the ring rotation recovers the whole short
  set: d=24: 249 -> 504 (~498 expected), d=20: 80 -> 180 (~158), d=26: 442 -> 780 (~886).
  Only ~n/2 distinct rotation orbits are hit.
- A POST-HOC test with a smaller database is impossible: with db_size_factor / 2 the sieve finds 0
  short vectors (it needs about (4/3)^(n/2) points to find reducing pairs at all). So the saving can
  only come from rotated images entering the database DURING the sieve, which is the change below.
  The measured result says that change has something to work with; it does not measure the speedup.
  Implementing it is still unwritten kernel C++ and is the next real step, to be built and tested
  on a machine where G6K compiles (it now builds on ARM with the patches in the validation notes).

## Where the hook goes (verified against G6K master source)
- `Siever::sample(large)` in `kernel/db.inl:114` produces new database entries. `grow_db_task`
  (`kernel/control.cpp:702`) calls it and rejects duplicates through `uid_hash_table.insert_uid`.
  Duplicates are already handled, so rotation images need no new dedup logic.
- Python cannot write database vectors (only `grow_db`, `shrink_db`, `insert`), so this must be a
  kernel change plus Cython plumbing (`siever.pxd`, `decl.pxd`, `siever_params.pyx`).

## Change
1. Parameter `rotation_order` (int, 0 = off) and a Python-supplied integer matrix `M`
   (coefficient-space rotation, `M = B P B^-1`, shape n x n, exact integer; compute it as in
   `rotation_orbit_check.rotation_matrix`). Valid only for full-rank sieves (`l = 0`, `r = n`), since the
   rotation is an automorphism of the whole ideal lattice, not of a projected sublattice.
2. In `sample()` when `rotation_order > 0` and `fullS >= 200 + 10*n`: with probability p pick a random
   existing `db` entry, apply `M^k` (k uniform in 1..rotation_order-1) to `e.x`, then
   `recompute_data_for_entry` (as the existing code does at `db.inl:134`). The norm is preserved, so
   no length filter is needed.
3. Thread safety: read `db[j].x` of an existing entry while other threads write other slots; bgj1
   already tolerates this class of race (see `dataraces_*` counters). Count rotation-sourced entries
   in a new statistic so the effect can be separated from ordinary sampling.
4. Run E1/E2-style sweeps with `db_size_factor / m` and compare time and op counts with the
   unmodified siever at the same m. Expected if the credit is real: runs at small m saturate, the
   baseline at the same m does not.

## Termination must count rotation images (otherwise the credit cannot appear)
bgj1 stops when an ABSOLUTE count of database vectors below `saturation_radius` is reached
(`GBL_saturation_count = saturation_radius^(n/2) * saturation_ratio / 2`, `bgj1_sieve.cpp:104`),
independent of database size. With default settings the database holds about 3.2*(4/3)^(n/2)
entries and the target count is about 0.25*(4/3)^(n/2), so a database shrunk by more than ~12x can
never reach it and the sieve will not terminate. A smaller database therefore only saves work if the
saturation count also credits the rotation images of the vectors found (each short vector
stands for up to 2d). The kernel change must update `GBL_saturation_count` accordingly, and the
coverage check in `rotation_orbit_check.py` is the evidence that doing so is sound.
At m = 8 the default settings are already close to this limit: expect timeouts at large m.

## Risks to check first
- Undersized databases may never reach the saturation count and loop forever; always run under a
  timeout (the orbit script does).
- Kyber/Dilithium lattices are module lattices: the automorphism group is the ring's, acting on each
  ring block; use the matching `P` per block structure rather than the single-ring `P` used here.
- Rotations change correlations between entries; check `dataraces_replaced_was_saturated` and the
  "same best vector" rate before quoting any gain.
