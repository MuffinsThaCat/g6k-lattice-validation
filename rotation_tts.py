"""
Round-2 item C' (see PREREGISTERED_CRITERIA_R2.md): time-to-solution of clone-everything rotation variants.

Runs rotation_sieve_ab.run_one for every (variant, seed) at one ring dimension, several runs in parallel, each in its
own process with a hard timeout (a hung sieve is a failed run charged the timeout).  Tuning seeds are run only at the
tuning dimension; holdout seeds at every dimension.  Selection and credits are computed by rotation_tts_analyze.py.

  TTS_D=28 TTS_TUNE=1 python rotation_tts.py          (cwd = patched g6k checkout, like the other scripts)
Env: TTS_D, TTS_Q=257, TTS_MODES="base,rot,rotmid,rotall", TTS_TUNE=0/1, TTS_HOLDOUT_SEEDS=16,
     TTS_NPROC=4, TTS_TIMEOUT=3000, TTS_OUT=res_tts_d<D>.json
"""
import json
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.environ.get("G6K_DIR", os.getcwd()))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

D = int(os.environ["TTS_D"])
Q = int(os.environ.get("TTS_Q", "257"))
GROUP = int(os.environ.get("TTS_GROUP", "0"))     # 0 = the full rotation group; else the subgroup of this order
LAT = int(os.environ.get("TTS_LATTICE", "1"))   # seed of the random ring element h (which lattice)
MODES = os.environ.get("TTS_MODES", "base,rot,rotmid,rotall").split(",")
TUNE = os.environ.get("TTS_TUNE", "0") == "1"
HOLDOUT = int(os.environ.get("TTS_HOLDOUT_SEEDS", "16"))
NPROC = int(os.environ.get("TTS_NPROC", "4"))
TIMEOUT = float(os.environ.get("TTS_TIMEOUT", "3000"))
OUT = os.environ.get("TTS_OUT", f"res_tts_d{D}_lat{LAT}" + (f"_g{GROUP}" if GROUP else "") + ".json")
SAT = 0.5
FACTOR = 3.2


def child(q, d, mode, seed, lat, group):
    try:
        import rotation_sieve_ab as AB
        import rotation_orbit_check as R
        B0, P = R.ideal_basis(d, Q, lat)
        q.put(AB.run_one(B0, P, FACTOR, mode, seed, sat=SAT, group=group))
    except BaseException as exc:  # report, never hide
        q.put({"error": f"{type(exc).__name__}: {exc}"[:300]})


def main():
    tasks = []
    if TUNE:
        tasks += [("tune", m, s) for s in range(100, 108) for m in MODES]
    tasks += [("holdout", m, s) for s in range(200, 200 + HOLDOUT) for m in MODES]
    print(f"d={D} lattice={LAT} n={2 * D}: {len(tasks)} runs, {NPROC} parallel, timeout {TIMEOUT:.0f}s", flush=True)
    ctx = mp.get_context("fork")
    running, rows, pending = [], [], list(tasks)
    t0 = time.time()
    while pending or running:
        while pending and len(running) < NPROC:
            sset, mode, seed = pending.pop(0)
            q = ctx.Queue()
            p = ctx.Process(target=child, args=(q, D, mode, seed, LAT, GROUP))
            p.start()
            running.append((p, q, sset, mode, seed, time.time()))
        time.sleep(0.5)
        still = []
        for p, q, sset, mode, seed, ts in running:
            el = time.time() - ts
            if not q.empty():
                r = q.get()
                p.join()
                done = True
            elif not p.is_alive():
                r = {"error": "child exited without a result"}
                done = True
            elif el > TIMEOUT:
                p.terminate()
                p.join()
                r = {"timeout": True, "wall": TIMEOUT}
                done = True
            else:
                done = False
            if not done:
                still.append((p, q, sset, mode, seed, ts))
                continue
            r.update(set=sset, mode=mode, seed=seed, d=D, lat=LAT, group=GROUP)
            rows.append(r)
            tag = "TIMEOUT" if r.get("timeout") else (r.get("error") or f"wall {r['wall']:.1f}s ops {r['ops']:.3g} clones {r['clones_attempted']}")
            print(f"[{time.time() - t0:7.0f}s] {sset:7} {mode:7} seed {seed}: {tag}", flush=True)
            with open(OUT, "w") as fh:
                json.dump({"d": D, "lat": LAT, "q": Q, "sat": SAT, "factor": FACTOR, "rows": rows}, fh)
        running = still
    print(f"Wrote {OUT} ({len(rows)} runs)")


if __name__ == "__main__":
    main()
