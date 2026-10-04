# Round 5b: completing the d256 set of Round 5

Written 2026-10-04, after Round 5's run (37169226764) and before this one.

Round 5's d256 set finished 8 of 32 instances (seeds 9003, 9016-9019, 9026, 9028, 9029); the other 24 did
not finish inside the 6 h job limit (about 2-4.5 h per instance on a runner, 4 per job). Round 5b runs exactly
those 24 seeds with identical code, parameters (n=128, k=1, m=128, q=3329, sigma=2.0, bmax 90, G6K pnj for every
block size, dd precision) and analysis (`aggregate_c_jsonl.py`), one instance per process, timeout 19800 s.

The d256 set is then the union of both runs' finished instances. The Round 5 decision rules apply unchanged:
it counts only with >= 20 finished instances and E found in >= 80% of them; unfinished seeds are reported
as such. The interim 8-instance figure (c_eff 1.02, 95% CI 0.27-11.15) was seen before this run was planned,
which is why no rule, parameter or seed choice is changed here.
