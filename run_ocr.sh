#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_ocr_pipeline.sh /workspace/inputs/YourBook.pdf "1-300"
# Page range is optional; omit for all pages.
PDF_PATH="${1:-}"
PAGES="${2:-}"

if [[ -z "${PDF_PATH}" ]]; then
  echo "Usage: $0 /path/to/book.pdf [page_range]"
  echo "Example: $0 /workspace/inputs/YourBook.pdf \"1-300\""
  exit 1
fi

# ---- Paths (edit if you want) ----
WORKDIR="${WORKDIR:-/workspace/work}"
VENV_DIR="${VENV_DIR:-$WORKDIR/venvs}"
PIP_QUIET="${PIP_QUIET:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PADDLE_USE_SYSTEM_SITE_PACKAGES="${PADDLE_USE_SYSTEM_SITE_PACKAGES:-0}"
DEEPSEEK_USE_SYSTEM_SITE_PACKAGES="${DEEPSEEK_USE_SYSTEM_SITE_PACKAGES:-1}"
MERGE_USE_SYSTEM_SITE_PACKAGES="${MERGE_USE_SYSTEM_SITE_PACKAGES:-1}"
PADDLE_TORCH_CPU="${PADDLE_TORCH_CPU:-1}"
TORCH_CPU_INDEX_URL="${TORCH_CPU_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-$WORKDIR/.cache/pip}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-$WORKDIR/.cache}"
SKIP_PIP_UPGRADE="${SKIP_PIP_UPGRADE:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PDF_OCR="${SCRIPT_DIR}/pdf_ocr.py"
UNIFY="${SCRIPT_DIR}/unify_results.py"

# ---- Render settings (tuned for two-column TTRPG scans) ----
DPI="${DPI:-350}"
MAX_SIDE="${MAX_SIDE:-3600}"
JPEG_QUALITY="${JPEG_QUALITY:-90}"
PREFER_EMBEDDED="${PREFER_EMBEDDED:-1}"   # 1 = attempt embedded extraction first

# ---- DeepSeek settings ----
DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-deepseek-ai/DeepSeek-OCR-2}"
DEEPSEEK_ATTN="${DEEPSEEK_ATTN:-sdpa}"   # sdpa is safest; flash_attention_2 if installed
DEEPSEEK_CROP_MODE="${DEEPSEEK_CROP_MODE:-1}" # 1 = crop_mode=True

# ---- Merger model ----
MERGE_MODEL="${MERGE_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
MERGE_MAX_NEW_TOKENS="${MERGE_MAX_NEW_TOKENS:-4096}"
MERGE_FAST="${MERGE_FAST:-1}"            # 1 = skip merger when candidates are near-identical

# ---- Cache locations ----
export HF_HOME="${HF_HOME:-$WORKDIR/.hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TOKENIZERS_PARALLELISM=false
export PIP_CACHE_DIR
export XDG_CACHE_HOME

ts() {
  date "+%Y-%m-%d %H:%M:%S"
}

log() {
  echo "[$(ts)] $*"
}

pip_install() {
  local pip="$1"
  shift
  if [[ "${PIP_QUIET}" == "1" ]]; then
    "${pip}" install "$@" >/dev/null
  else
    "${pip}" install "$@"
  fi
}

mkdir -p "${WORKDIR}" "${VENV_DIR}" "${PIP_CACHE_DIR}" "${XDG_CACHE_HOME}"

if [[ ! -f "${PDF_OCR}" ]]; then
  echo "ERROR: missing ${PDF_OCR}"
  exit 2
fi
if [[ ! -f "${UNIFY}" ]]; then
  echo "ERROR: missing ${UNIFY}"
  exit 2
fi

log "[info] WORKDIR=${WORKDIR}"
log "[info] HF_HOME=${HF_HOME}"
log "[info] PDF=${PDF_PATH}"
log "[info] PAGES=${PAGES:-<all>}"
log "[info] PIP_QUIET=${PIP_QUIET}"
log "[info] PYTHON_BIN=${PYTHON_BIN}"
log "[info] PADDLE_USE_SYSTEM_SITE_PACKAGES=${PADDLE_USE_SYSTEM_SITE_PACKAGES}"
log "[info] DEEPSEEK_USE_SYSTEM_SITE_PACKAGES=${DEEPSEEK_USE_SYSTEM_SITE_PACKAGES}"
log "[info] MERGE_USE_SYSTEM_SITE_PACKAGES=${MERGE_USE_SYSTEM_SITE_PACKAGES}"
log "[info] PADDLE_TORCH_CPU=${PADDLE_TORCH_CPU}"
log "[info] PIP_CACHE_DIR=${PIP_CACHE_DIR}"
log "[info] XDG_CACHE_HOME=${XDG_CACHE_HOME}"
log "[info] SKIP_PIP_UPGRADE=${SKIP_PIP_UPGRADE}"

# ---- Helpers ----
ensure_venv() {
  local venv_path="$1"
  local use_system="${2:-1}"
  if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "[error] Python not found: ${PYTHON_BIN}. Set PYTHON_BIN to a valid interpreter (e.g. python3.11)."
    exit 2
  fi
  if [[ ! -d "${venv_path}" ]]; then
    if [[ "${use_system}" == "1" ]]; then
      "${PYTHON_BIN}" -m venv --system-site-packages "${venv_path}"
    else
      "${PYTHON_BIN}" -m venv "${venv_path}"
    fi
  fi
  if [[ ! -x "${venv_path}/bin/pip" ]]; then
    log "[venv] pip missing; bootstrapping with ensurepip"
    "${venv_path}/bin/python" -m ensurepip --upgrade || true
  fi
  if ! "${venv_path}/bin/python" - <<'PY' >/dev/null 2>&1
import pip  # noqa: F401
PY
  then
    log "[venv] pip import failed; re-running ensurepip"
    "${venv_path}/bin/python" -m ensurepip --upgrade
  fi
  if [[ "${SKIP_PIP_UPGRADE}" != "1" ]]; then
    pip_install "${venv_path}/bin/pip" -U pip wheel setuptools
  fi
  log "[venv] ${venv_path} -> $("${venv_path}/bin/python" -V 2>&1) (system_site_packages=${use_system})"
}

install_paddle_stack() {
  local py="$1"
  local pip="$2"

  # Rendering deps
  log "[paddle] Installing render deps (pymupdf, pillow)"
  pip_install "${pip}" -U pymupdf pillow

  # If paddle isn't installed, try a few CUDA wheel indexes.
  if ! "${py}" -c "import paddle" >/dev/null 2>&1; then
    echo "[paddle] Installing paddlepaddle-gpu (will try a few CUDA indexes)..."
    PADDLE_VER="${PADDLE_VER:-3.2.1}"
    set +e
    log "[paddle] Attempting paddlepaddle-gpu==${PADDLE_VER} (cu126)"
    "${pip}" install "paddlepaddle-gpu==${PADDLE_VER}" -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
    install_status=$?
    if [[ "${install_status}" -ne 0 ]]; then
      log "[paddle] Attempting paddlepaddle-gpu==${PADDLE_VER} (cu121)"
      "${pip}" install "paddlepaddle-gpu==${PADDLE_VER}" -i https://www.paddlepaddle.org.cn/packages/stable/cu121/
      install_status=$?
    fi
    if [[ "${install_status}" -ne 0 ]]; then
      log "[paddle] Attempting paddlepaddle-gpu==${PADDLE_VER} (cu118)"
      "${pip}" install "paddlepaddle-gpu==${PADDLE_VER}" -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
      install_status=$?
    fi
    set -e
    if [[ "${install_status}" -ne 0 ]]; then
      echo "[paddle][WARN] Failed to install paddlepaddle-gpu. Falling back to CPU paddlepaddle."
      pip_install "${pip}" "paddlepaddle==${PADDLE_VER}"
    fi
  fi

  # PaddleOCR doc parser
  log "[paddle] Installing PaddleOCR doc parser"
  pip_install "${pip}" -U "paddleocr[doc-parser]"

  if [[ "${PADDLE_TORCH_CPU}" == "1" ]]; then
    if ! "${py}" -c "import torch" >/dev/null 2>&1; then
      log "[paddle] Installing CPU torch (for PaddleOCR deps)"
      pip_install "${pip}" --index-url "${TORCH_CPU_INDEX_URL}" torch
    fi
  fi

  if ! "${py}" - <<'PY'
import paddleocr
print("paddleocr import OK", getattr(paddleocr, "__version__", "unknown"))
PY
  then
    echo "[paddle][ERROR] paddleocr import failed after install. Check Python version and PaddleOCR compatibility."
    exit 3
  fi

  echo "[paddle] Versions:"
  "${py}" -c "import paddle; print('  paddle:', paddle.__version__, 'cuda:', paddle.is_compiled_with_cuda())" || true
}

install_deepseek_stack() {
  local py="$1"
  local pip="$2"

  # Rely on system torch (common in Runpod images). If torch isn't CUDA-enabled, you'll need to install a CUDA torch build.
  if ! "${py}" -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -qi "true"; then
    echo "[deepseek][WARN] torch.cuda.is_available() is not True in this venv."
    echo "               If you're on a GPU pod, consider using a base image that already includes CUDA torch,"
    echo "               or install torch from the PyTorch CUDA index URL."
  fi

  # DeepSeek-OCR-2 known-good deps from their model card (pinning helps stability)
  log "[deepseek] Installing DeepSeek deps (transformers/tokenizers/accelerate/pillow)"
  pip_install "${pip}" -U "transformers==4.47.1" "tokenizers==0.21.0" "accelerate>=0.30.0" pillow
}

install_merge_stack() {
  local py="$1"
  local pip="$2"

  if ! "${py}" -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -qi "true"; then
    echo "[merge][WARN] torch.cuda.is_available() is not True in this venv."
  fi

  # Qwen2.5-VL requires newer Transformers.
  log "[merge] Installing merge deps (transformers/accelerate/pillow/qwen-vl-utils)"
  pip_install "${pip}" -U "transformers>=4.48.0" accelerate pillow qwen-vl-utils
}

# ---- 1) Paddle env: render + paddle OCR ----
VENV_PADDLE="${VENV_DIR}/paddle"
ensure_venv "${VENV_PADDLE}" "${PADDLE_USE_SYSTEM_SITE_PACKAGES}"
PY_PADDLE="${VENV_PADDLE}/bin/python"
PIP_PADDLE="${VENV_PADDLE}/bin/pip"
install_paddle_stack "${PY_PADDLE}" "${PIP_PADDLE}"

RENDER_ARGS=(
  --engine render
  --input "${PDF_PATH}"
  --workdir "${WORKDIR}"
  --dpi "${DPI}"
  --max-side "${MAX_SIDE}"
  --img-format jpg
  --jpeg-quality "${JPEG_QUALITY}"
)
if [[ "${PREFER_EMBEDDED}" == "1" ]]; then
  RENDER_ARGS+=( --prefer-embedded )
fi
if [[ -n "${PAGES}" ]]; then
  RENDER_ARGS+=( --pages "${PAGES}" )
fi

log "[step] Render pages"
"${PY_PADDLE}" "${PDF_OCR}" "${RENDER_ARGS[@]}"

PADDLE_ARGS=(
  --engine paddle
  --workdir "${WORKDIR}"
)
if [[ -n "${PAGES}" ]]; then
  PADDLE_ARGS+=( --pages "${PAGES}" )
fi

log "[step] PaddleOCR-VL candidates"
"${PY_PADDLE}" "${PDF_OCR}" "${PADDLE_ARGS[@]}"

# ---- 2) DeepSeek env: deepseek OCR ----
VENV_DEEPSEEK="${VENV_DIR}/deepseek"
ensure_venv "${VENV_DEEPSEEK}" "${DEEPSEEK_USE_SYSTEM_SITE_PACKAGES}"
PY_DEEPSEEK="${VENV_DEEPSEEK}/bin/python"
PIP_DEEPSEEK="${VENV_DEEPSEEK}/bin/pip"
install_deepseek_stack "${PY_DEEPSEEK}" "${PIP_DEEPSEEK}"

DEEPSEEK_ARGS=(
  --engine deepseek
  --workdir "${WORKDIR}"
  --deepseek-model "${DEEPSEEK_MODEL}"
  --deepseek-attn "${DEEPSEEK_ATTN}"
)
if [[ "${DEEPSEEK_CROP_MODE}" == "1" ]]; then
  DEEPSEEK_ARGS+=( --deepseek-crop-mode )
fi
if [[ -n "${PAGES}" ]]; then
  DEEPSEEK_ARGS+=( --pages "${PAGES}" )
fi

log "[step] DeepSeek-OCR-2 candidates"
"${PY_DEEPSEEK}" "${PDF_OCR}" "${DEEPSEEK_ARGS[@]}"

# ---- 3) Merge env: Qwen2.5-VL unify ----
VENV_MERGE="${VENV_DIR}/merge"
ensure_venv "${VENV_MERGE}" "${MERGE_USE_SYSTEM_SITE_PACKAGES}"
PY_MERGE="${VENV_MERGE}/bin/python"
PIP_MERGE="${VENV_MERGE}/bin/pip"
install_merge_stack "${PY_MERGE}" "${PIP_MERGE}"

MERGE_ARGS=(
  --pages-dir "${WORKDIR}/pages"
  --paddle-root "${WORKDIR}/ocr_paddle"
  --deepseek-root "${WORKDIR}/ocr_deepseek"
  --out-root "${WORKDIR}/final"
  --model "${MERGE_MODEL}"
  --max-new-tokens "${MERGE_MAX_NEW_TOKENS}"
)
if [[ -n "${PAGES}" ]]; then
  MERGE_ARGS+=( --page-range "${PAGES}" )
fi
if [[ "${MERGE_FAST}" == "1" ]]; then
  MERGE_ARGS+=( --only-run-merger-when-different )
fi

log "[step] Merge candidates -> final markdown"
"${PY_MERGE}" "${UNIFY}" "${MERGE_ARGS[@]}"

echo ""
log "[done] Final outputs:"
log "  - Combined markdown: ${WORKDIR}/final/book.md"
log "  - Per-page markdown: ${WORKDIR}/final/pages/"
log "  - Unified assets:    ${WORKDIR}/final/assets/"
