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
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


# ----------------------------
# Small utils
# ----------------------------

def eprint(*args) -> None:
    print(*args, file=sys.stderr)


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

    # Lazy imports so OCR-only runs can work in different envs
    import fitz  # PyMuPDF
    from PIL import Image

    doc = fitz.open(pdf_path)
    try:
        page_count = doc.page_count
        for p1 in page_nums_1based:
            if not (1 <= p1 <= page_count):
                continue

            out_img = pages_dir / f"page_{p1:04d}.{img_format}"
            if out_img.exists() and not overwrite_pages:
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

            print(f"[render] page {p1:04d} -> {out_img.name}")

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

    for p1 in page_nums_1based:
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
            print(f"[paddle] page {p1:04d} -> {page_out}")
        except Exception as ex:
            eprint(f"[paddle][ERROR] page {p1:04d}: {ex}")

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
) -> Dict[str, object]:
    """
    Runs deepseek-ai/DeepSeek-OCR-2 on each page image.
    Uses model.infer(tokenizer, ..., save_results=True) which writes:
      - result.mmd
      - images/{idx}.jpg
    """
    ensure_dir(out_root)

    import torch
    from transformers import AutoModel, AutoTokenizer  # type: ignore

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # HF usage recommends AutoTokenizer + AutoModel with trust_remote_code.  [oai_citation:2‡Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    try:
        model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_safetensors=True,
            _attn_implementation=attn_implementation,
        )
    except Exception:
        # Fallback attention implementation if flash-attn isn't installed/available
        model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_safetensors=True,
            _attn_implementation="sdpa",
        )

    model = model.eval()
    if device == "cuda":
        model = model.cuda().to(dtype)

    done: List[int] = []
    skipped: List[int] = []

    for p1 in page_nums_1based:
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
            print(f"[deepseek] page {p1:04d} -> {page_out}")
        except Exception as ex:
            eprint(f"[deepseek][ERROR] page {p1:04d}: {ex}")

    manifest = {
        "engine": "deepseek_ocr2",
        "generated_at": now_iso(),
        "pages_dir": str(pages_dir),
        "out_root": str(out_root),
        "done": done,
        "skipped": skipped,
        "model_id": model_id,
        "prompt": prompt,
        "base_size": base_size,
        "image_size": image_size,
        "crop_mode": crop_mode,
        "attn_implementation": attn_implementation,
        "device": device,
        "dtype": str(dtype),
    }
    return manifest


# ----------------------------
# Main
# ----------------------------

def main() -> int:
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
    ap.add_argument("--deepseek-attn", default="flash_attention_2", help="Attention impl: flash_attention_2 or sdpa")

    args = ap.parse_args()

    workdir = Path(args.workdir).expanduser().resolve()
    ensure_dir(workdir)

    pages_dir = Path(args.pages_dir).expanduser().resolve() if args.pages_dir else (workdir / "pages")
    paddle_out = Path(args.paddle_out).expanduser().resolve() if args.paddle_out else (workdir / "ocr_paddle")
    deepseek_out = Path(args.deepseek_out).expanduser().resolve() if args.deepseek_out else (workdir / "ocr_deepseek")

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
        )
        safe_write_json(deepseek_out / "manifest.json", manifest)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())