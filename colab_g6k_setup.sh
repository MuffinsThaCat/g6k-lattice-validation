#!/usr/bin/env bash
# Build G6K on an x86-64 Linux machine with AVX2 (Google Colab free CPU runtime, a cloud VM, ...).
# NOT runnable on the ARM64 Mac this paper was developed on (G6K has a hard AVX2 dependency).
#
# STATUS: written from G6K's README and bootstrap description; it has NOT been executed by the author.
# G6K/fpylll build against moving upstream masters, so a step may need adjusting on a new Python.
# The script stops at the first failure and says which step failed.
#
# Colab usage (each in its own cell, prefix with ! or use %%bash):
#   !bash colab_g6k_setup.sh
#   !bash -c "cd g6k && source ./activate && python /content/g6k_opcount_vs_walltime.py"
set -euo pipefail

step() { echo; echo "=== $* ==="; }

step "1/5 CPU check (AVX2 required)"
if ! grep -q -m1 avx2 /proc/cpuinfo; then
  echo "FAIL: this CPU does not report avx2; G6K will not build/run here." >&2
  exit 1
fi
grep -m1 "model name" /proc/cpuinfo || true
nproc

step "2/5 system packages"
sudo_cmd=""; [ "$(id -u)" -ne 0 ] && sudo_cmd="sudo"
$sudo_cmd apt-get update -qq
$sudo_cmd apt-get install -y -qq autoconf automake libtool pkg-config g++ make git \
  libgmp-dev libmpfr-dev libqd-dev python3-dev python3-venv python3-pip patch

step "3/5 clone G6K"
[ -d g6k ] || git clone https://github.com/fplll/g6k.git
# Optional rotation-augmentation kernel patch (Siever.set_rotation). It is inert unless set_rotation
# is called, so the baseline E1/E2 runs are unaffected. Verified to apply to G6K master (2026-10-02).
if [ -f rotation_kernel.patch ] && ! grep -q set_rotation g6k/kernel/siever.h; then
  (cd g6k && patch -p1 < ../rotation_kernel.patch) && echo "applied rotation_kernel.patch"
fi

step "4/5 bootstrap (builds fplll + fpylll + G6K inside ./g6k, creates ./g6k/activate)"
cd g6k
# bootstrap.sh runs `python -m virtualenv` (python3-venv is not enough).
python3 -m pip install -q virtualenv \
  || python3 -m pip install --user -q virtualenv \
  || python3 -m pip install --user -q --break-system-packages virtualenv \
  || $sudo_cmd apt-get install -y -qq python3-virtualenv
export PATH="$HOME/.local/bin:$PATH"
# G6K compiles its operation counters (xorpopcnt_total, fullscprods_total, ...) OUT by default.
# VERIFIED BY A REAL BUILD: `./configure --with-stats=1` alone does NOT enable them, because
# kernel/siever.h includes statistics.hpp (line 43) BEFORE g6k_config.h (line 44), so the define
# arrives too late and every counter stays "not collected". ENABLE_STATS has to be a compiler flag.
# configure records CXXFLAGS in the kernel Makefile and in g6k.pc, and G6K's setup.py builds the
# Python extension with g6k.pc's Cflags, so one -DENABLE_STATS reaches both. Giving CXXFLAGS
# replaces G6K's defaults, so they are repeated here (configure adds the SIMD/native flags itself).
# G6K's setup.py runs `./configure` only if no Makefile exists, so we run it first, as G6K's CI does.
autoreconf -i
CXXFLAGS="-Ofast -ftree-vectorize -funroll-loops -Wall -Wextra -DENABLE_STATS" ./configure
grep -q "^Cflags:.*-DENABLE_STATS" g6k.pc \
  || { echo "FAIL: -DENABLE_STATS missing from g6k.pc Cflags after configure" >&2; exit 1; }
echo "configured G6K with -DENABLE_STATS"
PYTHON=python3 ./bootstrap.sh -j "$(nproc)"

step "5/5 smoke test (README example)"
# G6K's activate script reads unset variables (e.g. CFLAGS) and can end on a non-zero status,
# so relax nounset/errexit just while sourcing it
set +eu
# shellcheck disable=SC1091
source ./activate
set -eu
echo "activate sourced; python is $(command -v python)"
python - <<'PY'
from fpylll import IntegerMatrix, LLL, FPLLL
from g6k import Siever
FPLLL.set_random_seed(0x1337)
A = LLL.reduction(IntegerMatrix.random(50, "qary", k=25, bits=20))
g6k = Siever(A)
g6k.initialize_local(0, 0, 50)
g6k(alg="gauss")
print("G6K smoke test OK")
# Counters must be compiled in, otherwise the whole experiment is meaningless.
from g6k.siever_params import SieverParams
from g6k.siever import SaturationError
g6k = Siever(A, SieverParams(threads=1, default_sieve="bgj1", reserved_n=50))
g6k.initialize_local(0, 0, 50)
try:
    g6k(alg="bgj1")
except SaturationError:   # raised for dim >= 50 if saturation is not reached; the counters are still valid
    print("note: smoke sieve stopped before the requested saturation (not an error here)")
for key in ("xorpopcnt_total", "fullscprods_total"):
    v = g6k.get_stat(key)
    assert v is not None and v > 0, f"counter {key} missing/zero: rebuild with -DENABLE_STATS"
    print(f"counter {key} = {v}")
PY
echo; echo "Setup complete. Now run g6k_opcount_vs_walltime.py inside the activated environment."
