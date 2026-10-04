# Pre-registered criteria, ROUND 4B (written 2026-10-04, BEFORE any R4B run)

Why this round exists. Round 4 (run 37179192729) failed its own null control: g = 2 showed a gain with 95% CI
[+0.01, +0.07] bits, where it must show none, so by PREREGISTERED_CRITERIA_R4.md nothing from Round 4 is trusted.
The likely cause (not verified) is that Round 4 took its baseline time-to-solution from Round 2, a different
workflow run on a different day and possibly different runner hardware, so a machine-speed difference shifts every
gain. Round 4's rules forbid changing it after seeing results, so this is a new round with one design change.

The one change: every job measures the baseline itself, on its own runner, interleaved seed by seed with the
rotation runs (TTS_MODES="base,rotall"; rotation_tts.py orders tasks seed by seed, base then rotall). Each gain is
computed against the baseline of the SAME job. g = 64 (the full group, TTS_GROUP = 0) is run as its own job instead
of being taken from Round 2.

Everything else is unchanged from Round 4: d = 32 (n = 64), lattice seed 1, holdout seeds 200-215, the same
rotation_tts.py, the same G6K build (colab_g6k_setup.sh), the same success rule and L_ref rule (shortest database
vector equals the best known over all rows of this round), timeout 3000 s per run, 4 parallel runs per job.

Quantities (as Round 4): G(g) = log2(TTS_base,job(g) / TTS_rotall,job(g)), with a 95% bootstrap CI that resamples the
base and rotall runs of that job (seed 1, 2000 resamples); eps(g) = G(g) / log2(g/2); A(g) = 0.146 (g/2) bits;
N(g) = G(g) - A(g).

Decision rules (as Round 4):
- NULL CONTROL g = 2: if the CI lower bound of G(2) is > 0, nothing from this round is trusted.
- A subgroup gain is claimed only where the CI lower bound of G(g) is > 0.
- Reported: G, eps, A, N for every g, and g* = argmax N(g) among trusted gains; if max N <= 0 the credit is 0.
- Toy scale (n = 64): not extrapolated, and before the aligned-prefix quality cost, as in Round 4.
- No group order, dimension, lattice, seed or rule may be added or changed after results are seen.
- Also reported, not used for decisions: the six per-job baseline TTS values, to show how much runners differ.
