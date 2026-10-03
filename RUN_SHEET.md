# G6K validation run sheet

Goal: measure the paper's credits against compiled G6K instead of the numpy baseline.
Neither script has been executed yet; expect to fix a build step on first run.

## 1. Machine
- Linux x86-64 with AVX2, 32+ cores, 64 GB RAM (e.g. AWS c7i.8xlarge/16xlarge, GCP c3-standard-44).
- Ubuntu 22.04. Check: `grep -m1 avx2 /proc/cpuinfo`.
- Budget roughly 4-8 hours of instance time for the full sweep.

## 2. Copy files
    scp colab_g6k_setup.sh g6k_opcount_vs_walltime.py user@HOST:~/

## 3. Build (stops at first failing step and names it)
    bash colab_g6k_setup.sh 2>&1 | tee setup.log

## 4. Smoke run (minutes) -- confirm counters are read, not NaN
    cd g6k && source ./activate
    G6K_BETAS="40,42,44" G6K_SEEDS=2 G6K_SATS="0.5,0.7" G6K_E2_BETA=44 \
      python ../g6k_opcount_vs_walltime.py
    # If xorpopcnt/fullscprod print nan, fix read_counters() for this G6K version first.

## 5. Full run
    # single-thread: clean time-per-op and slope
    G6K_THREADS=1 G6K_BETAS="40,46,52,58,64,70" G6K_SEEDS=5 G6K_MAX_SECONDS=14400 \
      G6K_OUT=res_t1.json python ../g6k_opcount_vs_walltime.py | tee run_t1.log
    # multi-thread: reach higher beta (op counts should match t1; wall should drop)
    G6K_THREADS=$(nproc) G6K_BETAS="60,66,72,78" G6K_SEEDS=3 G6K_MAX_SECONDS=21600 \
      G6K_OUT=res_tN.json python ../g6k_opcount_vs_walltime.py | tee run_tN.log

## 6. Bring back
res_t1.json, res_tN.json, run_*.log, setup.log.

## 7. How to read it
- Fit slope of log2(ops) vs beta: compare with 0.292 (core-SVP) and with the paper's fitted slope.
- ns/op: how fast a G6K operation is; compare with the numpy harness's ns/op.
  The ratio is the overhead any throughput credit (batching, fusion, precision) must be discounted by.
- E2 "op bits" is the termination credit vs G6K's default saturation 0.5. Compare with 3.6-4.1.
  "same best vector" must be high, otherwise the early stop changed the answer and the credit is void.
- `datarace_frac` per beta (E1) and the WARNING lines: with >1 thread, a fraction above G6K_RACE_WARN
  (default 1e-3, untuned) means saturation/termination results are unreliable; rerun that beta with
  fewer threads. With 1 thread any nonzero count is a bug to investigate.
- E1 stops by itself when the next beta is projected to exceed G6K_MAX_SECONDS; a single run
  can still overshoot if the projection is off, so watch the largest beta.
- Smoke test must print both counters. Setup asserts -DENABLE_STATS is in g6k.pc before building
  (a real build showed `./configure --with-stats=1` alone does NOT enable the counters).
- Claims allowed afterward: only credits re-measured as G6K vs G6K+technique.

## 7b. Rotation check (after step 5, same activated env, cwd = the g6k checkout)
    python ../rotation_orbit_check.py --selftest          # math, runs anywhere
    for cfg in "20 193" "24 257" "26 257"; do set -- $cfg; \
      ROT_D=$1 ROT_Q=$2 python ../rotation_orbit_check.py | tee run_rot_d$1.log; done
Reads: "rotation closure of the N found short vectors: M (x of the ~E expected)". ARM runs gave
249 -> 504 (~498 expected) at d=24; an x86 run should agree, since these are counts, not timings.
Do not use ROT_MS>1: a database that small cannot sieve (finds 0 short vectors), see the docstring.

## 7c. In-sieve rotation A/B (needs rotation_kernel.patch applied before the build; setup does it)
    for cfg in "20 193" "24 257" "26 257"; do set -- $cfg; for sat in 0.5 1.0; do \
      AB_D=$1 AB_Q=$2 AB_SAT=$sat AB_FACTORS="3.2" AB_SEEDS=3 python ../rotation_sieve_ab.py; done; done
Compare with the ARM table in ROTATION_KERNEL_SPEC.md. Op counts should agree closely; wall times will not.

## 8. Next (needs code, not just a run)
Ring/module rotation database inflation and rotation-amplified decoding inside Siever,
then the four arms: baseline, +DRAWDOWN, +extras, combined.
