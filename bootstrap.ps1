# Bootstrap on a fresh Windows machine.
#
#   powershell -ExecutionPolicy Bypass -File bootstrap.ps1
#
# Creates a virtual environment, installs dependencies, and runs the preflight.
# Does not download model weights -- point configs/ at wherever yours live.
#
# Only core dependency installation is fatal. GGUF support, tests and the
# preflight are reported but never abort: a partial environment you can
# diagnose is more useful than an aborted one you cannot.
#
# NOTE ON ERROR HANDLING: ErrorActionPreference stays "Continue" throughout.
# Under "Stop", any native command that writes to stderr -- pip routinely does,
# even for harmless notices -- raises NativeCommandError and kills the script.
# Success is determined by $LASTEXITCODE, which is what actually means failure.

param(
    # CUDA wheel index for torch, newest first. The default PyPI wheel is
    # CPU-only on Windows, which imports fine and silently sees no GPU.
    # cu128 first: it is the first CUDA line with Blackwell (sm_120) kernels,
    # which the RTX 50-series needs, and it matches this machine's driver.
    [string[]]$TorchIndex = @(
        "https://download.pytorch.org/whl/cu128",
        "https://download.pytorch.org/whl/cu130",
        "https://download.pytorch.org/whl/cu126"
    ),
    # Skip the CUDA torch install (e.g. a machine with no NVIDIA GPU).
    [switch]$CpuOnly
)

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

function Fail($msg) {
    Write-Host ""
    Write-Host "BOOTSTRAP FAILED: $msg" -ForegroundColor Red
    exit 1
}

Write-Host "== GPU Starvation bootstrap ==" -ForegroundColor Cyan

# --- 1. Interpreter ---------------------------------------------------------
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Fail "python not found on PATH. Install Python 3.10 or newer."
}
$ver = (& python -c "import sys;print(f'{sys.version_info.major}.{sys.version_info.minor}')").Trim()
$okVer = (& python -c "import sys;print(1 if sys.version_info>=(3,10) else 0)").Trim()
if ($okVer -ne "1") { Fail "Python 3.10+ required, found $ver." }
Write-Host "Python $ver"

# --- 2. Virtual environment -------------------------------------------------
if (-not (Test-Path ".venv")) {
    Write-Host "Creating .venv ..."
    & python -m venv .venv
    if ($LASTEXITCODE -ne 0) { Fail "venv creation failed." }
}
$vpy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $vpy)) { Fail "venv incomplete: $vpy missing." }

# --- 3. Core dependencies (fatal) -------------------------------------------
Write-Host "Installing core dependencies ..."
& $vpy -m pip install --upgrade pip --quiet

# torch FIRST, from a CUDA index. Installing it via requirements.txt would pull
# the default PyPI wheel, which on Windows is CPU-only -- it imports without
# error and reports no CUDA device, so the failure is silent until a sweep
# produces nothing. Try indexes newest-first and stop at the first that works.
$torchOk = & $vpy -c "import torch,sys; sys.stdout.write('1' if torch.cuda.is_available() else '0')" 2>$null
if ($torchOk -ne "1" -and -not $CpuOnly) {
    Write-Host "Installing CUDA-enabled torch ..."
    foreach ($idx in $TorchIndex) {
        Write-Host "  trying $idx"
        & $vpy -m pip install torch --index-url $idx --quiet
        if ($LASTEXITCODE -eq 0) {
            $ok = & $vpy -c "import torch,sys; sys.stdout.write('1' if torch.cuda.is_available() else '0')" 2>$null
            if ($ok -eq "1") {
                $v = (& $vpy -c "import torch;print(torch.__version__)").Trim()
                Write-Host "  torch $v with CUDA -- OK" -ForegroundColor Green
                break
            }
            Write-Host "  installed but still no CUDA device; trying next index" -ForegroundColor Yellow
        }
    }
}

& $vpy -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { Fail "Core dependency install failed (see pip output above)." }
& $vpy -m pip install pytest ruff --quiet

# The deprecated `pynvml` distribution shadows nvidia-ml-py and makes torch emit
# a FutureWarning on every import. Both provide `import pynvml`. Check before
# uninstalling: pip warns on stderr when the package is absent, which is noise.
$hasOld = (& $vpy -c "import importlib.metadata as m; print(1 if any((d.metadata['Name'] or '').lower()=='pynvml' for d in m.distributions()) else 0)").Trim()
if ($hasOld -eq "1") {
    Write-Host "Replacing deprecated pynvml with nvidia-ml-py ..."
    & $vpy -m pip uninstall -y pynvml --quiet
}
& $vpy -m pip install nvidia-ml-py --quiet
& $vpy -m pip install -e . --quiet

# --- 4. Optional: GGUF (non-fatal) ------------------------------------------
Write-Host ""
Write-Host "Installing optional GGUF support (llama-cpp-python) ..."
& $vpy -m pip install -r requirements-gguf.txt --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "  llama-cpp-python did not install." -ForegroundColor Yellow
    Write-Host "  pickle vs safetensors still works; GGUF will be skipped." -ForegroundColor Yellow
    Write-Host "  For a prebuilt CUDA wheel:" -ForegroundColor Yellow
    Write-Host "    .\.venv\Scripts\python -m pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124" -ForegroundColor Yellow
}

# --- 5. Tests (non-fatal) ---------------------------------------------------
Write-Host ""
Write-Host "Running tests ..."
& $vpy -m pytest tests/ -q
if ($LASTEXITCODE -ne 0) { Write-Host "Tests failed -- see above." -ForegroundColor Red }

# --- 6. Preflight (informational) -------------------------------------------
Write-Host ""
& $vpy scripts/preflight.py
$preflight = $LASTEXITCODE

Write-Host ""
Write-Host "Bootstrap complete. Activate with:  .\.venv\Scripts\activate" -ForegroundColor Cyan
if ($preflight -ne 0) {
    Write-Host "Preflight reported problems -- each lists its remedy above." -ForegroundColor Yellow
}
exit 0
