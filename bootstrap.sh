#!/usr/bin/env bash
# Bootstrap on a fresh Linux/macOS machine.
#
#   bash bootstrap.sh
#
# Only core dependency installation is fatal. GGUF support, tests and the
# preflight are reported but never abort the script.
set -uo pipefail
cd "$(dirname "$0")"

echo "== GPU Starvation bootstrap =="

command -v python3 >/dev/null || { echo "python3 not found"; exit 1; }
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' \
  || { echo "Python 3.10+ required"; exit 1; }
echo "Python $(python3 -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')"

[ -d .venv ] || { echo "Creating .venv ..."; python3 -m venv .venv || exit 1; }
VPY=.venv/bin/python

echo "Installing core dependencies ..."
"$VPY" -m pip install --upgrade pip --quiet
"$VPY" -m pip install -r requirements.txt || { echo "Core install failed."; exit 1; }
"$VPY" -m pip install pytest ruff --quiet

# The deprecated `pynvml` distribution shadows nvidia-ml-py; both provide
# `import pynvml`, but the old one makes torch emit a FutureWarning.
"$VPY" -m pip uninstall -y pynvml --quiet 2>/dev/null || true
"$VPY" -m pip install nvidia-ml-py --quiet
"$VPY" -m pip install -e . --quiet

echo
echo "Installing optional GGUF support (llama-cpp-python) ..."
if ! "$VPY" -m pip install -r requirements-gguf.txt --quiet; then
  echo "  llama-cpp-python did not install; GGUF will be skipped."
  echo "  Prebuilt CUDA wheels: --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124"
fi

echo
echo "Running tests ..."
"$VPY" -m pytest tests/ -q || echo "Tests failed -- see above."

echo
"$VPY" scripts/preflight.py || true

echo
echo "Activate the environment with:  source .venv/bin/activate"
exit 0
