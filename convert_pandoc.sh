#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./convert_pandoc.sh [input_md] [out_dir] [base_name]

Defaults:
  input_md  = final/book.md
  out_dir   = directory of input_md
  base_name = input_md filename without .md

Environment variables:
  TOC=1              Include table of contents (default: 1)
  TOC_DEPTH=2        TOC depth (default: 2)
  EMBED_HTML=1       Embed images into HTML (default: 1)
  DOLLAR_MATH=0      Enable $...$ TeX math parsing (default: 0)

Examples:
  ./convert_pandoc.sh
  ./convert_pandoc.sh final/book.md final shadowrun_2e
  EMBED_HTML=0 ./convert_pandoc.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

INPUT_MD="${1:-final/book.md}"
OUT_DIR="${2:-$(dirname "$INPUT_MD")}"
BASE_NAME="${3:-$(basename "$INPUT_MD" .md)}"

if [[ ! -f "$INPUT_MD" ]]; then
  echo "Input not found: $INPUT_MD" >&2
  exit 1
fi

if ! command -v pandoc >/dev/null 2>&1; then
  echo "pandoc not found in PATH. Install pandoc and retry." >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

RESOURCE_PATH="$(dirname "$INPUT_MD")"
FROM="markdown+raw_html"
if [[ "${DOLLAR_MATH:-0}" != "1" ]]; then
  FROM="${FROM}-tex_math_dollars"
fi

PANDOC_COMMON_ARGS=(
  --from="$FROM"
  --resource-path="$RESOURCE_PATH"
)

if [[ "${TOC:-1}" == "1" ]]; then
  PANDOC_COMMON_ARGS+=(--toc --toc-depth="${TOC_DEPTH:-2}")
fi

EPUB_OUT="${OUT_DIR}/${BASE_NAME}.epub"
HTML_OUT="${OUT_DIR}/${BASE_NAME}.html"

pandoc "$INPUT_MD" \
  "${PANDOC_COMMON_ARGS[@]}" \
  --to=epub \
  --output="$EPUB_OUT"

HTML_ARGS=(--to=html --standalone)
if [[ "${EMBED_HTML:-1}" == "1" ]]; then
  HTML_ARGS+=(--embed-resources)
fi

pandoc "$INPUT_MD" \
  "${PANDOC_COMMON_ARGS[@]}" \
  "${HTML_ARGS[@]}" \
  --output="$HTML_OUT"

echo "Wrote: $EPUB_OUT"
echo "Wrote: $HTML_OUT"
