# pdf-ocr-bot

OCR pipeline for large scanned PDFs. It renders pages to images, runs two OCR engines (PaddleOCR-VL and DeepSeek-OCR-2), then merges candidates into clean per-page and combined Markdown using a multimodal model (default: Qwen2.5-VL-7B-Instruct).

Project layout:
- `run_ocr.sh` orchestrates the full pipeline and venv setup.
- `pdf_ocr.py` renders pages and runs PaddleOCR-VL or DeepSeek-OCR-2.
- `unify_results.py` merges the candidates into final Markdown.

**Quick Start**
1. Make the script executable.
1. Run the pipeline, optionally with a page range.

```bash
chmod +x run_ocr.sh
./run_ocr.sh /path/to/YourBook.pdf "1-300"
```

Outputs (default):
- Combined Markdown: `/workspace/work/final/book.md`
- Per-page Markdown: `/workspace/work/final/pages/`
- Unified assets: `/workspace/work/final/assets/`

Set `WORKDIR` to change the output location.

**Runpod Setup**
This pipeline is designed for Runpod and large PDFs. It expects a CUDA-capable environment for DeepSeek and the merge model.

GPU guidance (from the original plan):
- Prefer a **24GB VRAM** GPU for good throughput and headroom.
- Common choices: **RTX 4090**, **NVIDIA L4**, or **RTX 3090**.
- If you plan to use vLLM acceleration, favor compute capability ≥ 8.0 and CUDA ≥ 12.6.
- Pricing and inventory change frequently, so verify availability in the Runpod UI.

Storage guidance:
- Use a **Network Volume** so your PDF, model caches, and outputs persist across pod restarts.
- Keep large assets under `/workspace` (typical Runpod mount point).

Data transfer:
- For large PDFs, use **SSH/rsync** rather than the web terminal.
- Runpod’s “basic terminal access” is not true SSH and does not support SCP/rsync.
- Use `rsync -avzP --inplace` for resumable transfers when you have SSH access.

**Runpod Quickstart**
1. Start a GPU pod with enough disk to store model caches and rendered pages.
1. Attach a network volume (recommended).
1. Open a terminal in the pod.
1. Clone this repo into `/workspace` and create an inputs folder.
1. Upload or copy your PDF into `/workspace/inputs/`.
1. Run the pipeline.

```bash
cd /workspace
git clone <your-repo-url> pdf-ocr-bot
mkdir -p /workspace/inputs
cd /workspace/pdf-ocr-bot
chmod +x run_ocr.sh
./run_ocr.sh /workspace/inputs/YourBook.pdf "1-300"
```

Outputs will be written to `/workspace/work/final/` unless `WORKDIR` is set. You can download `book.md`, `pages/`, and `assets/` from there.

If you see warnings that `torch.cuda.is_available()` is false, use a CUDA-enabled base image or install a CUDA build of PyTorch in the venvs created by the script.

**Configuration**
Environment variables supported by `run_ocr.sh`:
- `WORKDIR` (default `/workspace/work`)
- `PYTHON_BIN` (default `python3`, set to `python3.11` if you need a stable ABI across pods)
- `DPI` (default `350`)
- `MAX_SIDE` (default `3600`)
- `JPEG_QUALITY` (default `90`)
- `PREFER_EMBEDDED` (default `1`)
- `PIP_QUIET` (default `0`, set to `1` to silence pip output)
- `SKIP_PIP_UPGRADE` (default `0`, set to `1` to avoid upgrading pip/wheel/setuptools each run)
- `PIP_CACHE_DIR` (default `${WORKDIR}/.cache/pip`)
- `XDG_CACHE_HOME` (default `${WORKDIR}/.cache`)
- `PADDLEX_HOME` (default `${WORKDIR}/.paddlex`)
- `PADDLE_PDX_HOME` (default `${PADDLEX_HOME}`)
- `PADDLE_PDX_CACHE_HOME` (default `${PADDLEX_HOME}`)
- `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK` (default `1`)
- `DEEPSEEK_MODEL` (default `deepseek-ai/DeepSeek-OCR-2`)
- `DEEPSEEK_REVISION` (default empty; set to a commit hash/tag to pin model code)
- `DEEPSEEK_ATTN` (default `eager`)
- `DEEPSEEK_CROP_MODE` (default `1`)
- `MERGE_MODEL` (default `Qwen/Qwen2.5-VL-7B-Instruct`)
- `MERGE_MAX_NEW_TOKENS` (default `4096`)
- `MERGE_FAST` (default `1`)
- `MERGE_REPORT` (default `0`, set to `1` to write per-page merge reports + diffs)
- `MERGE_REPORT_DIR` (default `${WORKDIR}/final/merge_reports`)
- `MERGE_DIFF_CONTEXT` (default `3`, unified diff context lines)
- `MERGE_DIFF_MAX_LINES` (default `400`, per-page diff line cap; `0` = no limit)
- `PADDLE_VER` (default `3.2.1`)
- `HF_HOME` and `TRANSFORMERS_CACHE` for model caching
- `PADDLE_USE_SYSTEM_SITE_PACKAGES` (default `0`)
- `DEEPSEEK_USE_SYSTEM_SITE_PACKAGES` (default `1`)
- `MERGE_USE_SYSTEM_SITE_PACKAGES` (default `1`)
- `PADDLE_TORCH_CPU` (default `1`)
- `PADDLE_NUMPY_PIN` (default `1`, pins `numpy<2` in the Paddle venv)

**Manual Stages**
You can run stages independently with `pdf_ocr.py` and `unify_results.py`.

Render pages:
```bash
python3 pdf_ocr.py --engine render --input /path/to/book.pdf --workdir /path/to/work --pages "1-300"
```

Paddle OCR (run in the Paddle venv):
```bash
python3 pdf_ocr.py --engine paddle --workdir /path/to/work --pages "1-300"
```

DeepSeek OCR (run in the DeepSeek venv):
```bash
python3 pdf_ocr.py --engine deepseek --workdir /path/to/work --pages "1-300"
```

Merge candidates (run in the merge venv):
```bash
python3 unify_results.py \
  --pages-dir /path/to/work/pages \
  --paddle-root /path/to/work/ocr_paddle \
  --deepseek-root /path/to/work/ocr_deepseek \
  --out-root /path/to/work/final \
  --page-range "1-300" \
  --report-dir /path/to/work/final/merge_reports
```

**Notes**
- The pipeline is resume-safe. It skips existing page renders and OCR outputs unless overwrite flags are set.
- Each run creates three venvs under `${WORKDIR}/venvs` (`paddle`, `deepseek`, `merge`).
- `run_ocr.sh` assumes a CUDA-capable PyTorch is present in the base image; it does not install `torch` itself.
- If `MERGE_REPORT=1`, per-page JSON reports and candidate diffs are written under `${WORKDIR}/final/merge_reports`.
