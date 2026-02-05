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

mkdir -p "${WORKDIR}" "${VENV_DIR}"

if [[ ! -f "${PDF_OCR}" ]]; then
  echo "ERROR: missing ${PDF_OCR}"
  exit 2
fi
if [[ ! -f "${UNIFY}" ]]; then
  echo "ERROR: missing ${UNIFY}"
  exit 2
fi

echo "[info] WORKDIR=${WORKDIR}"
echo "[info] HF_HOME=${HF_HOME}"
echo "[info] PDF=${PDF_PATH}"
echo "[info] PAGES=${PAGES:-<all>}"

# ---- Helpers ----
ensure_venv() {
  local venv_path="$1"
  if [[ ! -d "${venv_path}" ]]; then
    python3 -m venv --system-site-packages "${venv_path}"
  fi
  "${venv_path}/bin/python" -m pip install -U pip wheel setuptools >/dev/null
}

install_paddle_stack() {
  local py="$1"
  local pip="$2"

  # Rendering deps
  "${pip}" install -U pymupdf pillow >/dev/null

  # If paddle isn't installed, try a few CUDA wheel indexes.
  if ! "${py}" -c "import paddle" >/dev/null 2>&1; then
    echo "[paddle] Installing paddlepaddle-gpu (will try a few CUDA indexes)..."
    PADDLE_VER="${PADDLE_VER:-3.2.1}"
    set +e
    "${pip}" install "paddlepaddle-gpu==${PADDLE_VER}" -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
    install_status=$?
    if [[ "${install_status}" -ne 0 ]]; then
      "${pip}" install "paddlepaddle-gpu==${PADDLE_VER}" -i https://www.paddlepaddle.org.cn/packages/stable/cu121/
      install_status=$?
    fi
    if [[ "${install_status}" -ne 0 ]]; then
      "${pip}" install "paddlepaddle-gpu==${PADDLE_VER}" -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
      install_status=$?
    fi
    set -e
    if [[ "${install_status}" -ne 0 ]]; then
      echo "[paddle][WARN] Failed to install paddlepaddle-gpu. Falling back to CPU paddlepaddle."
      "${pip}" install "paddlepaddle==${PADDLE_VER}" >/dev/null
    fi
  fi

  # PaddleOCR doc parser
  "${pip}" install -U "paddleocr[doc-parser]" >/dev/null

  if ! "${py}" -c "import paddleocr" >/dev/null 2>&1; then
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
  "${pip}" install -U "transformers==4.47.1" "tokenizers==0.21.0" "accelerate>=0.30.0" pillow >/dev/null
}

install_merge_stack() {
  local py="$1"
  local pip="$2"

  if ! "${py}" -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -qi "true"; then
    echo "[merge][WARN] torch.cuda.is_available() is not True in this venv."
  fi

  # Qwen2.5-VL requires newer Transformers.
  "${pip}" install -U "transformers>=4.48.0" accelerate pillow qwen-vl-utils >/dev/null
}

# ---- 1) Paddle env: render + paddle OCR ----
VENV_PADDLE="${VENV_DIR}/paddle"
ensure_venv "${VENV_PADDLE}"
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

echo "[step] Render pages"
"${PY_PADDLE}" "${PDF_OCR}" "${RENDER_ARGS[@]}"

PADDLE_ARGS=(
  --engine paddle
  --workdir "${WORKDIR}"
)
if [[ -n "${PAGES}" ]]; then
  PADDLE_ARGS+=( --pages "${PAGES}" )
fi

echo "[step] PaddleOCR-VL candidates"
"${PY_PADDLE}" "${PDF_OCR}" "${PADDLE_ARGS[@]}"

# ---- 2) DeepSeek env: deepseek OCR ----
VENV_DEEPSEEK="${VENV_DIR}/deepseek"
ensure_venv "${VENV_DEEPSEEK}"
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

echo "[step] DeepSeek-OCR-2 candidates"
"${PY_DEEPSEEK}" "${PDF_OCR}" "${DEEPSEEK_ARGS[@]}"

# ---- 3) Merge env: Qwen2.5-VL unify ----
VENV_MERGE="${VENV_DIR}/merge"
ensure_venv "${VENV_MERGE}"
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

echo "[step] Merge candidates -> final markdown"
"${PY_MERGE}" "${UNIFY}" "${MERGE_ARGS[@]}"

echo ""
echo "[done] Final outputs:"
echo "  - Combined markdown: ${WORKDIR}/final/book.md"
echo "  - Per-page markdown: ${WORKDIR}/final/pages/"
echo "  - Unified assets:    ${WORKDIR}/final/assets/"
