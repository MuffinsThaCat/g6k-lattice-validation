"""
Run g6k_measure_c.worker on a shard of seeds without multiprocessing.Pool (a Pool worker
died silently at d=256 and hung the run). One JSON line per finished instance, appended,
so a killed run keeps what it finished.

  ~/latticefail_env/venv312/bin/python g6k_measure_c_shard.py <n> <k> <m> <q> <sigma> <bmax> <seed0> <count> <out.jsonl>
"""
import json
import sys

import g6k_measure_c as G


def main():
    n, k, m, q = map(int, sys.argv[1:5])
    sigma = float(sys.argv[5])
    bmax, seed0, count = int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8])
    out = sys.argv[9]
    betas = list(range(10, bmax + 1))
    for seed in range(seed0, seed0 + count):
        r = G.worker((n, k, m, q, sigma, betas, seed))
        r["per_beta"] = {str(b): v for b, v in r["per_beta"].items()}
        with open(out, "a") as f:
            f.write(json.dumps(r) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
