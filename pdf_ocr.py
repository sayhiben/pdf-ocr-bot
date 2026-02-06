#!/usr/bin/env python3
"""
pdf_ocr_candidates.py

Render/extract pages from a large scanned PDF into per-page images, then run
OCR with either:
  - PaddleOCRVL (PaddleOCR-VL / PaddleOCR-VL-1.5 doc parser), or
  - DeepSeek-OCR-2 (HF model.infer API)

Output layout (default):
  workdir/
    pages/
      page_0001.jpg
      page_0002.jpg
      ...
    ocr_paddle/
      page_0001/   # markdown/json/assets written by res.save_to_markdown/save_to_json
      page_0002/
      ...
    ocr_deepseek/
      page_0001/
        result.mmd
        images/0.jpg ...
      page_0002/
      ...

This is designed to be resume-safe and friendly to large PDFs:
- Page images are cached on disk.
- OCR outputs are cached per page and skipped unless overwrite flags are set.

Recommended usage:
  1) Render pages once:
     python pdf_ocr_candidates.py --engine render --input book.pdf --workdir work

  2) Run Paddle OCR (in a Paddle env):
     python pdf_ocr_candidates.py --engine paddle --workdir work

  3) Run DeepSeek OCR (in a Torch env):
     python pdf_ocr_candidates.py --engine deepseek --workdir work
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


LOG = logging.getLogger("pdf_ocr")


# ----------------------------
# Small utils
# ----------------------------

def eprint(*args) -> None:
    LOG.error(" ".join(str(a) for a in args))


def setup_logging() -> None:
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_write_json(path: Path, obj) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_page_range(spec: str, max_page_1based: int) -> List[int]:
    """
    Parse "1-3,8,10-" into sorted unique 1-based page numbers.
    If spec is empty -> 1..max_page
    """
    spec = (spec or "").strip()
    if not spec:
        return list(range(1, max_page_1based + 1))

    out: set[int] = set()
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    for part in parts:
        if "-" in part:
            left, right = part.split("-", 1)
            left = left.strip()
            right = right.strip()
            start = int(left) if left else 1
            end = int(right) if right else max_page_1based
            if start < 1:
                start = 1
            if end > max_page_1based:
                end = max_page_1based
            if end >= start:
                for p in range(start, end + 1):
                    out.add(p)
        else:
            p = int(part)
            if 1 <= p <= max_page_1based:
                out.add(p)
    return sorted(out)


def discover_pages_dir_max(pages_dir: Path) -> int:
    """
    If we don't have the PDF handy, infer max page number from existing images.
    """
    best = 0
    for p in pages_dir.glob("page_*.jpg"):
        m = re.match(r"page_(\d+)\.jpg$", p.name)
        if m:
            best = max(best, int(m.group(1)))
    for p in pages_dir.glob("page_*.png"):
        m = re.match(r"page_(\d+)\.png$", p.name)
        if m:
            best = max(best, int(m.group(1)))
    return best


def get_page_image_path(pages_dir: Path, page_num_1based: int) -> Optional[Path]:
    """
    Find the page image path for a given page number among common extensions.
    """
    for ext in ("jpg", "jpeg", "png", "webp"):
        cand = pages_dir / f"page_{page_num_1based:04d}.{ext}"
        if cand.exists():
            return cand
    return None


# ----------------------------
# Rendering / extraction
# ----------------------------

def extract_largest_embedded_image(doc, page) -> Optional[Tuple[bytes, str]]:
    """
    Try to extract the largest embedded raster image on a page.
    Returns (bytes, ext) or None.
    """
    try:
        infos = page.get_images(full=True)
    except Exception:
        return None
    if not infos:
        return None

    best: Optional[Tuple[bytes, str]] = None
    best_pixels = 0
    for info in infos:
        xref = info[0]
        try:
            data = doc.extract_image(xref)
        except Exception:
            continue
        if not data:
            continue
        img_bytes = data.get("image")
        ext = (data.get("ext") or "").lower()
        w = int(data.get("width") or 0)
        h = int(data.get("height") or 0)
        if not img_bytes or not ext or w <= 0 or h <= 0:
            continue
        pixels = w * h
        if pixels > best_pixels:
            best_pixels = pixels
            best = (img_bytes, ext)
    return best


def render_pages(
    pdf_path: Path,
    pages_dir: Path,
    page_nums_1based: Sequence[int],
    *,
    dpi: int,
    max_side: int,
    img_format: str,
    jpeg_quality: int,
    prefer_embedded: bool,
    overwrite_pages: bool,
) -> Dict[str, object]:
    """
    Render/extract each selected page to pages_dir/page_XXXX.<img_format>
    """
    ensure_dir(pages_dir)
    total_pages = len(page_nums_1based)
    LOG.info(
        "Render: %d pages (dpi=%s max_side=%s format=%s prefer_embedded=%s)",
        total_pages,
        dpi,
        max_side,
        img_format,
        prefer_embedded,
    )

    # Lazy imports so OCR-only runs can work in different envs
    import fitz  # PyMuPDF
    from PIL import Image

    doc = fitz.open(pdf_path)
    try:
        page_count = doc.page_count
        for idx, p1 in enumerate(page_nums_1based, start=1):
            LOG.info("Render page %04d (%d/%d)", p1, idx, total_pages)
            if not (1 <= p1 <= page_count):
                continue

            out_img = pages_dir / f"page_{p1:04d}.{img_format}"
            if out_img.exists() and not overwrite_pages:
                LOG.debug("Render skip page %04d (exists)", p1)
                continue

            page = doc.load_page(p1 - 1)

            # Try embedded image extraction (fast for scanned PDFs)
            if prefer_embedded:
                extracted = extract_largest_embedded_image(doc, page)
                if extracted is not None:
                    img_bytes, ext = extracted
                    tmp = pages_dir / f"page_{p1:04d}.embedded.{ext}"
                    tmp.write_bytes(img_bytes)
                    try:
                        im = Image.open(tmp).convert("RGB")
                        if max_side > 0 and max(im.size) > max_side:
                            scale = max_side / float(max(im.size))
                            new_size = (
                                max(1, int(im.size[0] * scale)),
                                max(1, int(im.size[1] * scale)),
                            )
                            im = im.resize(new_size, Image.LANCZOS)
                        if img_format in ("jpg", "jpeg"):
                            im.save(out_img, "JPEG", quality=jpeg_quality, optimize=True, progressive=True)
                        else:
                            im.save(out_img, "PNG")
                        tmp.unlink(missing_ok=True)
                        LOG.debug("Render page %04d via embedded image", p1)
                        continue
                    except Exception:
                        # fallback to render below
                        tmp.unlink(missing_ok=True)

            # Render page
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

            if max_side > 0 and max(im.size) > max_side:
                scale = max_side / float(max(im.size))
                new_size = (max(1, int(im.size[0] * scale)), max(1, int(im.size[1] * scale)))
                im = im.resize(new_size, Image.LANCZOS)

            if img_format in ("jpg", "jpeg"):
                im.save(out_img, "JPEG", quality=jpeg_quality, optimize=True, progressive=True)
            else:
                im.save(out_img, "PNG")

            LOG.info("Render page %04d -> %s", p1, out_img.name)

        manifest = {
            "pdf": str(pdf_path),
            "generated_at": now_iso(),
            "dpi": dpi,
            "max_side": max_side,
            "img_format": img_format,
            "jpeg_quality": jpeg_quality,
            "prefer_embedded": prefer_embedded,
            "pages_rendered": list(page_nums_1based),
        }
        return manifest
    finally:
        doc.close()


# ----------------------------
# OCR: PaddleOCR-VL / PaddleOCR-VL-1.5
# ----------------------------

def run_paddleocr_vl(
    pages_dir: Path,
    out_root: Path,
    page_nums_1based: Sequence[int],
    *,
    overwrite_ocr: bool,
    vl_rec_backend: Optional[str],
    vl_rec_server_url: Optional[str],
) -> Dict[str, object]:
    """
    Runs PaddleOCRVL doc parser on each page image.
    Writes per-page markdown/json/assets to out_root/page_XXXX/.
    """
    ensure_dir(out_root)
    total_pages = len(page_nums_1based)
    LOG.info("PaddleOCR-VL: %d pages -> %s", total_pages, out_root)
    LOG.info("Loading PaddleOCR-VL pipeline")

    # Lazy import so deepseek env doesn't need paddle installed
    from paddleocr import PaddleOCRVL  # type: ignore

    kwargs = {}
    if vl_rec_backend:
        kwargs["vl_rec_backend"] = vl_rec_backend
    if vl_rec_server_url:
        kwargs["vl_rec_server_url"] = vl_rec_server_url

    # PaddleOCR-VL-1.5 README shows this usage pattern (predict + save_to_markdown/save_to_json).  [oai_citation:1‡Hugging Face](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5/blob/main/README.md)
    pipeline = PaddleOCRVL(**kwargs)

    done: List[int] = []
    skipped: List[int] = []

    for idx, p1 in enumerate(page_nums_1based, start=1):
        LOG.info("Paddle page %04d (%d/%d)", p1, idx, total_pages)
        img_path = get_page_image_path(pages_dir, p1)
        if img_path is None:
            eprint(f"[paddle] missing page image for {p1:04d} in {pages_dir}")
            continue

        page_out = out_root / f"page_{p1:04d}"
        ensure_dir(page_out)

        # Skip if already has a markdown file
        existing_md = list(page_out.glob("*.md"))
        if existing_md and not overwrite_ocr:
            skipped.append(p1)
            LOG.debug("Paddle skip page %04d (exists)", p1)
            continue

        try:
            output = pipeline.predict(str(img_path))
            for res in output:
                res.save_to_json(save_path=str(page_out))
                res.save_to_markdown(save_path=str(page_out))
            safe_write_json(page_out / "meta.json", {
                "engine": "paddleocr_vl",
                "generated_at": now_iso(),
                "input_image": str(img_path),
                "vl_rec_backend": vl_rec_backend,
                "vl_rec_server_url": vl_rec_server_url,
            })
            done.append(p1)
            LOG.info("Paddle page %04d -> %s", p1, page_out)
        except Exception as ex:
            eprint(f"[paddle][ERROR] page {p1:04d}: {ex}")

    LOG.info("PaddleOCR-VL done=%d skipped=%d", len(done), len(skipped))
    manifest = {
        "engine": "paddleocr_vl",
        "generated_at": now_iso(),
        "pages_dir": str(pages_dir),
        "out_root": str(out_root),
        "done": done,
        "skipped": skipped,
        "vl_rec_backend": vl_rec_backend,
        "vl_rec_server_url": vl_rec_server_url,
    }
    return manifest


# ----------------------------
# OCR: DeepSeek-OCR-2
# ----------------------------

def load_deepseek_model(
    model_id: str,
    attn_implementation: str,
    revision: str,
):
    import torch
    from transformers import AutoModel, AutoTokenizer  # type: ignore

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # HF usage recommends AutoTokenizer + AutoModel with trust_remote_code.  [oai_citation:2‡Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        revision=revision or None,
    )

    attn_candidates: List[str] = []
    if attn_implementation:
        attn_candidates.append(attn_implementation)
    for cand in ("sdpa", "eager"):
        if cand not in attn_candidates:
            attn_candidates.append(cand)

    model = None
    last_err: Optional[Exception] = None
    for impl in attn_candidates:
        try:
            model = AutoModel.from_pretrained(
                model_id,
                trust_remote_code=True,
                use_safetensors=True,
                _attn_implementation=impl,
                revision=revision or None,
            )
            if impl != attn_implementation:
                LOG.warning("DeepSeek attn_implementation fallback: %s -> %s", attn_implementation, impl)
            break
        except Exception as ex:
            last_err = ex
            LOG.warning("DeepSeek attn_implementation=%s failed: %s", impl, ex)
            continue
    if model is None:
        raise RuntimeError(f"Failed to load DeepSeek model with attn implementations: {attn_candidates}") from last_err

    model = model.eval()
    if device == "cuda":
        model = model.cuda().to(dtype)

    return tokenizer, model, device, dtype


def _deepseek_single_page_worker(
    queue,
    *,
    img_path: str,
    page_out: str,
    model_id: str,
    prompt: str,
    base_size: int,
    image_size: int,
    crop_mode: bool,
    attn_implementation: str,
    revision: str,
) -> None:
    try:
        tokenizer, model, device, dtype = load_deepseek_model(
            model_id=model_id,
            attn_implementation=attn_implementation,
            revision=revision,
        )
        _ = model.infer(
            tokenizer,
            prompt=prompt,
            image_file=img_path,
            output_path=page_out,
            base_size=base_size,
            image_size=image_size,
            crop_mode=crop_mode,
            save_results=True,
        )
        safe_write_json(Path(page_out) / "meta.json", {
            "engine": "deepseek_ocr2",
            "generated_at": now_iso(),
            "input_image": img_path,
            "model_id": model_id,
            "prompt": prompt,
            "base_size": base_size,
            "image_size": image_size,
            "crop_mode": crop_mode,
            "attn_implementation": attn_implementation,
            "device": device,
            "dtype": str(dtype),
        })
        queue.put({"ok": True})
    except Exception as ex:
        queue.put({"ok": False, "error": str(ex)})

def run_deepseek_ocr2(
    pages_dir: Path,
    out_root: Path,
    page_nums_1based: Sequence[int],
    *,
    overwrite_ocr: bool,
    model_id: str,
    prompt: str,
    base_size: int,
    image_size: int,
    crop_mode: bool,
    attn_implementation: str,
    revision: str,
    page_timeout_sec: int,
) -> Dict[str, object]:
    """
    Runs deepseek-ai/DeepSeek-OCR-2 on each page image.
    Uses model.infer(tokenizer, ..., save_results=True) which writes:
      - result.mmd
      - images/{idx}.jpg
    """
    ensure_dir(out_root)
    total_pages = len(page_nums_1based)
    LOG.info(
        "DeepSeek-OCR-2: %d pages -> %s (attn=%s timeout=%s)",
        total_pages,
        out_root,
        attn_implementation,
        f"{page_timeout_sec}s" if page_timeout_sec and page_timeout_sec > 0 else "none",
    )
    done: List[int] = []
    skipped: List[int] = []
    timed_out: List[int] = []
    device = "unknown"
    dtype = "unknown"

    if page_timeout_sec and page_timeout_sec > 0:
        import multiprocessing as mp
        import queue as queue_mod

        LOG.warning("DeepSeek per-page timeout enabled. Model will be loaded in a subprocess for each page.")
        ctx = mp.get_context("spawn")

        for idx, p1 in enumerate(page_nums_1based, start=1):
            LOG.info("DeepSeek page %04d (%d/%d)", p1, idx, total_pages)
            img_path = get_page_image_path(pages_dir, p1)
            if img_path is None:
                eprint(f"[deepseek] missing page image for {p1:04d} in {pages_dir}")
                continue

            page_out = out_root / f"page_{p1:04d}"
            ensure_dir(page_out)

            result_mmd = page_out / "result.mmd"
            if result_mmd.exists() and not overwrite_ocr:
                skipped.append(p1)
                LOG.debug("DeepSeek skip page %04d (exists)", p1)
                continue

            q = ctx.Queue()
            proc = ctx.Process(
                target=_deepseek_single_page_worker,
                kwargs={
                    "queue": q,
                    "img_path": str(img_path),
                    "page_out": str(page_out),
                    "model_id": model_id,
                    "prompt": prompt,
                    "base_size": base_size,
                    "image_size": image_size,
                    "crop_mode": crop_mode,
                    "attn_implementation": attn_implementation,
                    "revision": revision,
                },
            )
            proc.start()
            proc.join(timeout=page_timeout_sec)
            if proc.is_alive():
                LOG.warning("DeepSeek page %04d timed out after %ss; terminating.", p1, page_timeout_sec)
                proc.terminate()
                proc.join(timeout=5)
                timed_out.append(p1)
                continue

            try:
                result = q.get_nowait()
            except queue_mod.Empty:
                eprint(f"[deepseek][ERROR] page {p1:04d}: no result returned")
                continue

            if result.get("ok"):
                done.append(p1)
                LOG.info("DeepSeek page %04d -> %s", p1, page_out)
            else:
                err = result.get("error") or "unknown error"
                eprint(f"[deepseek][ERROR] page {p1:04d}: {err}")
    else:
        LOG.info("Loading DeepSeek model %s", model_id)
        tokenizer, model, device, dtype = load_deepseek_model(
            model_id=model_id,
            attn_implementation=attn_implementation,
            revision=revision,
        )

        for idx, p1 in enumerate(page_nums_1based, start=1):
            LOG.info("DeepSeek page %04d (%d/%d)", p1, idx, total_pages)
            img_path = get_page_image_path(pages_dir, p1)
            if img_path is None:
                eprint(f"[deepseek] missing page image for {p1:04d} in {pages_dir}")
                continue

            page_out = out_root / f"page_{p1:04d}"
            ensure_dir(page_out)

            # DeepSeek code writes {output_path}/result.mmd when save_results=True.  [oai_citation:3‡Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2/blob/main/modeling_deepseekocr2.py)
            result_mmd = page_out / "result.mmd"
            if result_mmd.exists() and not overwrite_ocr:
                skipped.append(p1)
                LOG.debug("DeepSeek skip page %04d (exists)", p1)
                continue

            try:
                # Prompt & call signature per model card.  [oai_citation:4‡Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2)
                _ = model.infer(
                    tokenizer,
                    prompt=prompt,
                    image_file=str(img_path),
                    output_path=str(page_out),
                    base_size=base_size,
                    image_size=image_size,
                    crop_mode=crop_mode,
                    save_results=True,
                )
                safe_write_json(page_out / "meta.json", {
                    "engine": "deepseek_ocr2",
                    "generated_at": now_iso(),
                    "input_image": str(img_path),
                    "model_id": model_id,
                    "prompt": prompt,
                    "base_size": base_size,
                    "image_size": image_size,
                    "crop_mode": crop_mode,
                    "attn_implementation": attn_implementation,
                    "device": device,
                    "dtype": str(dtype),
                })
                done.append(p1)
                LOG.info("DeepSeek page %04d -> %s", p1, page_out)
            except Exception as ex:
                eprint(f"[deepseek][ERROR] page {p1:04d}: {ex}")

    LOG.info("DeepSeek-OCR-2 done=%d skipped=%d timed_out=%d", len(done), len(skipped), len(timed_out))
    manifest = {
        "engine": "deepseek_ocr2",
        "generated_at": now_iso(),
        "pages_dir": str(pages_dir),
        "out_root": str(out_root),
        "done": done,
        "skipped": skipped,
        "timed_out": timed_out,
        "model_id": model_id,
        "prompt": prompt,
        "base_size": base_size,
        "image_size": image_size,
        "crop_mode": crop_mode,
        "attn_implementation": attn_implementation,
        "page_timeout_sec": page_timeout_sec,
        "device": device,
        "dtype": str(dtype),
    }
    return manifest


# ----------------------------
# Main
# ----------------------------

def main() -> int:
    setup_logging()

    ap = argparse.ArgumentParser(description="PDF -> per-page images -> OCR candidates (PaddleOCR-VL + DeepSeek-OCR-2).")
    ap.add_argument("--engine", required=True, choices=["render", "paddle", "deepseek", "all"],
                    help="Which stage to run.")
    ap.add_argument("--workdir", required=True, help="Working directory root.")
    ap.add_argument("--input", default="", help="Input PDF path (required for render or auto-render).")
    ap.add_argument("--pages-dir", default="", help="Override pages image dir (default: workdir/pages).")
    ap.add_argument("--pages", default="", help="Page range like '1-10,25,40-'. Default: all pages.")

    # Rendering options
    ap.add_argument("--dpi", type=int, default=350)
    ap.add_argument("--max-side", type=int, default=3600)
    ap.add_argument("--img-format", choices=["jpg", "png"], default="jpg")
    ap.add_argument("--jpeg-quality", type=int, default=90)
    ap.add_argument("--prefer-embedded", action="store_true", help="Prefer extracting embedded images from PDF pages (fast for scanned PDFs).")
    ap.add_argument("--overwrite-pages", action="store_true")
    ap.add_argument("--auto-render-missing", action="store_true",
                    help="If OCR is requested and a page image is missing, render it (requires --input).")

    # OCR overwrite
    ap.add_argument("--overwrite-ocr", action="store_true")

    # Paddle options (optional)
    ap.add_argument("--paddle-out", default="", help="Override paddle output root (default: workdir/ocr_paddle).")
    ap.add_argument("--vl-rec-backend", default="", help="PaddleOCRVL vl_rec_backend (e.g. vllm-server).")
    ap.add_argument("--vl-rec-server-url", default="", help="PaddleOCRVL vl_rec_server_url (e.g. http://127.0.0.1:8080/v1).")

    # DeepSeek options
    ap.add_argument("--deepseek-out", default="", help="Override deepseek output root (default: workdir/ocr_deepseek).")
    ap.add_argument("--deepseek-model", default="deepseek-ai/DeepSeek-OCR-2")
    ap.add_argument("--deepseek-prompt", default="<image>\n<|grounding|>Convert the document to markdown. ")
    ap.add_argument("--deepseek-base-size", type=int, default=1024)
    ap.add_argument("--deepseek-image-size", type=int, default=768)
    ap.add_argument("--deepseek-crop-mode", action="store_true", help="Enable crop_mode=True in infer()")
    ap.add_argument("--deepseek-attn", default="eager", help="Attention impl: flash_attention_2, sdpa, or eager")
    ap.add_argument("--deepseek-revision", default="", help="Pin a specific model revision (commit hash/tag) to avoid code updates.")
    ap.add_argument("--deepseek-page-timeout", type=int, default=0,
                    help="If >0, run each page in a subprocess and terminate if it exceeds this many seconds.")

    args = ap.parse_args()

    workdir = Path(args.workdir).expanduser().resolve()
    ensure_dir(workdir)

    pages_dir = Path(args.pages_dir).expanduser().resolve() if args.pages_dir else (workdir / "pages")
    paddle_out = Path(args.paddle_out).expanduser().resolve() if args.paddle_out else (workdir / "ocr_paddle")
    deepseek_out = Path(args.deepseek_out).expanduser().resolve() if args.deepseek_out else (workdir / "ocr_deepseek")
    LOG.info("Workdir=%s pages_dir=%s", workdir, pages_dir)
    LOG.info("Paddle out=%s DeepSeek out=%s", paddle_out, deepseek_out)

    # Determine page count (prefer PDF if available)
    pdf_path = Path(args.input).expanduser().resolve() if args.input else None

    max_pages: int = 0
    if pdf_path and pdf_path.exists() and args.engine in ("render", "all"):
        # We'll open PDF inside render; but we still need page_count for parsing ranges.
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        try:
            max_pages = doc.page_count
        finally:
            doc.close()
    else:
        # Infer from existing pages_dir if possible
        if pages_dir.exists():
            max_pages = discover_pages_dir_max(pages_dir)

    if max_pages <= 0 and args.engine in ("render", "all"):
        if not pdf_path or not pdf_path.exists():
            eprint("ERROR: --input PDF is required for --engine render/all.")
            return 2

        # We can still compute max_pages by opening now
        import fitz
        doc = fitz.open(pdf_path)
        try:
            max_pages = doc.page_count
        finally:
            doc.close()

    if max_pages <= 0 and args.pages.strip():
        eprint("ERROR: cannot parse --pages without knowing max pages. Provide --input or render pages first.")
        return 2

    # Choose pages
    if max_pages > 0:
        page_nums = parse_page_range(args.pages, max_pages)
    else:
        # No range spec and no pdf; just use discovered pages
        if not pages_dir.exists():
            eprint("ERROR: no pages dir exists and no PDF provided.")
            return 2
        # Use all discovered page numbers
        nums = []
        for p in sorted(list(pages_dir.glob("page_*.jpg")) + list(pages_dir.glob("page_*.png"))):
            m = re.match(r"page_(\d+)\.(jpg|png)$", p.name)
            if m:
                nums.append(int(m.group(1)))
        page_nums = sorted(set(nums))

    if not page_nums:
        eprint("No pages selected.")
        return 2
    LOG.info("Selected pages: %d", len(page_nums))

    # Helper: auto-render missing page images if requested
    def ensure_page_images_exist() -> None:
        if not pages_dir.exists():
            ensure_dir(pages_dir)
        missing = [p for p in page_nums if get_page_image_path(pages_dir, p) is None]
        if not missing:
            return
        if not args.auto_render_missing:
            raise RuntimeError(f"Missing {len(missing)} page images. Run --engine render or use --auto-render-missing with --input.")
        if not pdf_path or not pdf_path.exists():
            raise RuntimeError("auto-render-missing requested but --input PDF not provided.")
        render_manifest = render_pages(
            pdf_path,
            pages_dir,
            missing,
            dpi=args.dpi,
            max_side=args.max_side,
            img_format=args.img_format,
            jpeg_quality=args.jpeg_quality,
            prefer_embedded=args.prefer_embedded,
            overwrite_pages=args.overwrite_pages,
        )
        safe_write_json(workdir / "render_manifest.json", render_manifest)

    # Run requested engine(s)
    if args.engine == "render":
        if not pdf_path or not pdf_path.exists():
            eprint("ERROR: --input PDF is required for --engine render.")
            return 2
        manifest = render_pages(
            pdf_path,
            pages_dir,
            page_nums,
            dpi=args.dpi,
            max_side=args.max_side,
            img_format=args.img_format,
            jpeg_quality=args.jpeg_quality,
            prefer_embedded=args.prefer_embedded,
            overwrite_pages=args.overwrite_pages,
        )
        safe_write_json(workdir / "render_manifest.json", manifest)
        return 0

    if args.engine in ("paddle", "all"):
        ensure_page_images_exist()
        manifest = run_paddleocr_vl(
            pages_dir,
            paddle_out,
            page_nums,
            overwrite_ocr=args.overwrite_ocr,
            vl_rec_backend=(args.vl_rec_backend or None),
            vl_rec_server_url=(args.vl_rec_server_url or None),
        )
        safe_write_json(paddle_out / "manifest.json", manifest)

    if args.engine in ("deepseek", "all"):
        ensure_page_images_exist()
        manifest = run_deepseek_ocr2(
            pages_dir,
            deepseek_out,
            page_nums,
            overwrite_ocr=args.overwrite_ocr,
            model_id=args.deepseek_model,
            prompt=args.deepseek_prompt,
            base_size=args.deepseek_base_size,
            image_size=args.deepseek_image_size,
            crop_mode=bool(args.deepseek_crop_mode),
            attn_implementation=args.deepseek_attn,
            revision=args.deepseek_revision,
            page_timeout_sec=args.deepseek_page_timeout,
        )
        safe_write_json(deepseek_out / "manifest.json", manifest)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
