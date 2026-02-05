# Agent Notes

This repo is a Runpod-friendly OCR pipeline for large scanned PDFs. It renders pages, runs two OCR engines, then merges their outputs into per-page and combined Markdown with images.

Origins / rationale (from `init-project-plan.txt`):
- Built for **large PDFs** and **resume-safe** processing (page-by-page, skip existing outputs).
- Designed to run on **Runpod** with GPU, using a **network volume** for persistence.
- Uses **separate venvs** to avoid Paddle vs Torch dependency conflicts.
- Combines **PaddleOCR-VL** and **DeepSeek-OCR-2** and merges with **Qwen2.5-VL** for higher-quality Markdown.

Key entrypoints:
- `run_ocr.sh` orchestrates the full pipeline and venv setup.
- `pdf_ocr.py` renders pages and runs PaddleOCR-VL or DeepSeek-OCR-2.
- `unify_results.py` merges candidates with a multimodal model.

Operational expectations:
- GPU is strongly recommended for DeepSeek and the merge model.
- The pipeline is resume-safe; it skips work when outputs already exist unless overwrite flags are set.
- Large model downloads happen on first run. Keep disk space in mind.

When changing behavior:
- Preserve CLI flags and defaults in `pdf_ocr.py` and `unify_results.py` unless there is a clear reason to break compatibility.
- If you add or rename an option, update `run_ocr.sh` and `README.md` together.
- Keep new dependencies minimal and justified.
- Do not reintroduce PP-StructureV3 without an explicit request; the current pipeline intentionally uses PaddleOCR-VL + DeepSeek + merge.

Running locally:
- Use `./run_ocr.sh /path/to/file.pdf "1-300"` (page range optional).
- Outputs land under `WORKDIR` (default `/workspace/work`).

Tests:
- No automated test suite is defined in this repo.
