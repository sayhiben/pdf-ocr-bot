#!/usr/bin/env python3
"""
Unify two OCR markdown candidates (PaddleOCR-VL + DeepSeek-OCR-2) into a final per-page
Markdown using a multimodal "merger" model (default: Qwen2.5-VL-7B-Instruct).

Key behaviors:
- Copies/renames image assets produced by each OCR engine into final/assets/page_XXXX/
- Rewrites markdown image links to point at the unified assets paths
- Calls the merger model with (page_image + candidateA + candidateB) and writes final markdown
- Safe to resume: skips pages that already have final output unless --overwrite is set
"""

from __future__ import annotations

import argparse
import difflib
import os
import re
import shutil
from pathlib import Path
from typing import Optional, Tuple, List

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


IMG_MD_RE = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')
IMG_HTML_RE = re.compile(r'<img[^>]*src=["\']([^"\']+)["\'][^>]*>', re.IGNORECASE)


def parse_page_range(expr: str) -> List[int]:
    """
    Accepts: "1-5", "1,2,10-12"
    Returns: sorted unique page numbers
    """
    pages: set[int] = set()
    expr = expr.strip()
    if not expr:
        return []
    for part in expr.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            start = int(a)
            end = int(b)
            step = 1 if end >= start else -1
            for p in range(start, end + step, step):
                pages.add(p)
        else:
            pages.add(int(part))
    return sorted(pages)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def looks_trivial(md: str) -> bool:
    stripped = re.sub(r"\s+", "", md)
    return len(stripped) < 80


def similarity(a: str, b: str, limit_chars: int = 20000) -> float:
    a0 = a[:limit_chars]
    b0 = b[:limit_chars]
    return difflib.SequenceMatcher(a=a0, b=b0).ratio()


def find_candidate_markdown_paddle(page_dir: Path) -> Optional[Path]:
    # PaddleOCR save_to_markdown() commonly writes .md somewhere in the directory.
    mds = sorted(page_dir.rglob("*.md"))
    return mds[0] if mds else None


def find_candidate_markdown_deepseek(page_dir: Path) -> Optional[Path]:
    # DeepSeek-OCR-2 (save_results=True) writes result.mmd in output_path (per their code).
    mmd = page_dir / "result.mmd"
    if mmd.exists():
        return mmd
    mds = sorted(page_dir.rglob("*.md"))
    return mds[0] if mds else None


def copy_page_image(page_img: Path, out_asset_dir: Path) -> str:
    out_asset_dir.mkdir(parents=True, exist_ok=True)
    dst = out_asset_dir / "page.jpg"
    if not dst.exists():
        shutil.copy2(page_img, dst)
    return "page.jpg"


def normalize_and_copy_images(
    md_text: str,
    md_base_dir: Path,
    out_asset_dir: Path,
    engine_prefix: str,
    rel_from_page_md_to_assets: str,
) -> str:
    """
    Copies local image assets referenced in md_text into out_asset_dir with deterministic names,
    and rewrites links to point at rel_from_page_md_to_assets/<copied_name>.

    rel_from_page_md_to_assets example: "../assets/page_0001"
    """
    out_asset_dir.mkdir(parents=True, exist_ok=True)
    seen: dict[str, str] = {}
    counter = 0

    def _copy_one(path_str: str) -> Optional[str]:
        nonlocal counter
        path_str = path_str.strip()
        # Handle markdown image syntax with optional title: (path "title")
        # We'll only copy the first token as path.
        tokens = path_str.split()
        img_path = tokens[0]

        # Skip URLs/data URIs
        if re.match(r"^(https?://|data:)", img_path, re.IGNORECASE):
            return None

        # Already rewritten?
        if img_path.startswith(rel_from_page_md_to_assets):
            return img_path

        if img_path in seen:
            return seen[img_path]

        src = (md_base_dir / img_path).resolve()
        if not src.exists() or not src.is_file():
            return None

        ext = src.suffix.lower() or ".jpg"
        dst_name = f"{engine_prefix}_{counter:03d}{ext}"
        counter += 1

        dst = out_asset_dir / dst_name
        if not dst.exists():
            shutil.copy2(src, dst)

        new_rel = f"{rel_from_page_md_to_assets}/{dst_name}"
        seen[img_path] = new_rel
        return new_rel

    # Rewrite Markdown image links
    def md_repl(m: re.Match) -> str:
        alt = m.group(1)
        inside = m.group(2).strip()

        tokens = inside.split()
        img_path = tokens[0]
        title = " ".join(tokens[1:]) if len(tokens) > 1 else ""

        new_rel = _copy_one(inside)
        if new_rel is None:
            return m.group(0)

        # Preserve title if present
        if title:
            return f"![{alt}]({new_rel} {title})"
        return f"![{alt}]({new_rel})"

    md_text = IMG_MD_RE.sub(md_repl, md_text)

    # Rewrite HTML <img src="...">
    def html_repl(m: re.Match) -> str:
        src_attr = m.group(1).strip()
        new_rel = _copy_one(src_attr)
        if new_rel is None:
            return m.group(0)
        return m.group(0).replace(src_attr, new_rel)

    md_text = IMG_HTML_RE.sub(html_repl, md_text)

    return md_text


def extract_markdown_from_fenced(text: str) -> str:
    # If the model returns ```markdown ... ```, keep only inside.
    m = re.search(r"```(?:markdown)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text.strip()


def build_merger_prompt(paddle_md: str, deepseek_md: str) -> str:
    # Strong constraints help reduce hallucination and keep image links valid.
    return f"""
You are an expert OCR post-editor for scanned tabletop RPG books.

You will be shown:
- The PAGE IMAGE (ground truth)
- Candidate A: Markdown OCR output from PaddleOCR-VL
- Candidate B: Markdown OCR output from DeepSeek-OCR-2

Task:
Produce the best possible FINAL Markdown transcription for this page.

Rules (must follow):
1) Output ONLY Markdown (no explanations, no preface, no JSON).
2) Fix reading order for TWO-COLUMN layouts: read top-to-bottom in the left column, then top-to-bottom in the right column.
3) Preserve formatting: headings, bold/italics, lists, stat blocks, tables, blockquotes.
4) Do NOT invent content. If uncertain, choose the most plausible text supported by either candidate and/or the page image.
5) Preserve exact numbers, dice notation (e.g. 2d6+3), symbols, and proper nouns.
6) You may keep or drop low-value decorative artifacts, but keep meaningful sidebars, callouts, tables, charts.
7) CRITICAL: Use ONLY image links that already appear in Candidate A or Candidate B. Do not create new filenames.

Candidate A (PaddleOCR-VL):
---BEGIN A---
{paddle_md}
---END A---

Candidate B (DeepSeek-OCR-2):
---BEGIN B---
{deepseek_md}
---END B---

Return the FINAL Markdown now.
""".strip()


def load_merger(model_id: str, prefer_flash_attn: bool, min_pixels: int, max_pixels: int):
    attn_impl = "flash_attention_2" if prefer_flash_attn else "sdpa"
    # If flash_attention_2 isn't available, we'll fall back to sdpa.
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation=attn_impl,
        )
    except Exception:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
        )

    processor = AutoProcessor.from_pretrained(model_id, min_pixels=min_pixels, max_pixels=max_pixels)
    model.eval()
    return model, processor


@torch.no_grad()
def run_merger_on_page(
    model,
    processor,
    page_image_path: Path,
    prompt_text: str,
    max_new_tokens: int,
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "path": str(page_image_path)},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )
    # Move tensors to model device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
    )

    # Strip prompt tokens
    prompt_len = inputs["input_ids"].shape[-1]
    gen_ids = output_ids[0][prompt_len:]
    out = processor.batch_decode([gen_ids], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages-dir", required=True, type=str, help="Rendered page images: page_0001.jpg, ...")
    ap.add_argument("--paddle-root", required=True, type=str, help="Root folder containing Paddle outputs per page dir")
    ap.add_argument("--deepseek-root", required=True, type=str, help="Root folder containing DeepSeek outputs per page dir")
    ap.add_argument("--out-root", required=True, type=str, help="Output root, will create out-root/pages and out-root/assets")

    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct", type=str)
    ap.add_argument("--max-new-tokens", default=4096, type=int)
    ap.add_argument("--min-pixels", default=256 * 28 * 28, type=int)
    ap.add_argument("--max-pixels", default=1280 * 28 * 28, type=int)
    ap.add_argument("--prefer-flash-attn", action="store_true")

    ap.add_argument("--page-range", default="", type=str, help='e.g. "1-300" or "1,2,10-12". Default: all pages found.')
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--only-run-merger-when-different", action="store_true",
                    help="If candidates are nearly identical, skip merger and choose best candidate directly.")
    ap.add_argument("--similarity-threshold", type=float, default=0.975)

    args = ap.parse_args()

    pages_dir = Path(args.pages_dir)
    paddle_root = Path(args.paddle_root)
    deepseek_root = Path(args.deepseek_root)
    out_root = Path(args.out_root)

    out_pages_dir = out_root / "pages"
    out_assets_dir = out_root / "assets"
    out_book = out_root / "book.md"

    # Determine pages to process
    if args.page_range.strip():
        pages = parse_page_range(args.page_range)
        page_imgs = [pages_dir / f"page_{p:04d}.jpg" for p in pages]
    else:
        page_imgs = sorted(pages_dir.glob("page_*.jpg"))

    if not page_imgs:
        raise SystemExit(f"No page images found in {pages_dir}")

    # Load merger model once
    model, processor = load_merger(args.model, args.prefer_flash_attn, args.min_pixels, args.max_pixels)

    combined_chunks: List[str] = []

    for page_img in page_imgs:
        m = re.search(r"page_(\d+)\.", page_img.name)
        if not m:
            continue
        pnum = int(m.group(1))

        paddle_page_dir = paddle_root / f"page_{pnum:04d}"
        deepseek_page_dir = deepseek_root / f"page_{pnum:04d}"

        out_page_md = out_pages_dir / f"page_{pnum:04d}.md"
        out_asset_page_dir = out_assets_dir / f"page_{pnum:04d}"

        if out_page_md.exists() and not args.overwrite:
            # Still include in combined book
            page_md_text = read_text(out_page_md)
            combined_chunks.append(f"\n\n<!-- PAGE {pnum:04d} -->\n\n" + page_md_text.replace("(../assets/", "(assets/"))
            continue

        # Load candidates
        paddle_md_path = find_candidate_markdown_paddle(paddle_page_dir) if paddle_page_dir.exists() else None
        deepseek_md_path = find_candidate_markdown_deepseek(deepseek_page_dir) if deepseek_page_dir.exists() else None

        paddle_md_raw = read_text(paddle_md_path) if paddle_md_path else ""
        deepseek_md_raw = read_text(deepseek_md_path) if deepseek_md_path else ""

        # Ensure we at least have something
        if not paddle_md_raw and not deepseek_md_raw:
            print(f"[WARN] Page {pnum:04d}: no candidates found, skipping.")
            continue

        # Copy the full page image into assets
        copy_page_image(page_img, out_asset_page_dir)

        # Normalize assets + rewrite paths so both candidates point into ../assets/page_XXXX
        rel_assets = f"../assets/page_{pnum:04d}"

        paddle_md = paddle_md_raw
        if paddle_md_path:
            paddle_md = normalize_and_copy_images(
                paddle_md_raw,
                md_base_dir=paddle_md_path.parent,
                out_asset_dir=out_asset_page_dir,
                engine_prefix="paddle",
                rel_from_page_md_to_assets=rel_assets,
            )

        deepseek_md = deepseek_md_raw
        if deepseek_md_path:
            deepseek_md = normalize_and_copy_images(
                deepseek_md_raw,
                md_base_dir=deepseek_md_path.parent,
                out_asset_dir=out_asset_page_dir,
                engine_prefix="deepseek",
                rel_from_page_md_to_assets=rel_assets,
            )

        # Cheap fast-paths (no merger call)
        if looks_trivial(paddle_md) and not looks_trivial(deepseek_md):
            final_md = deepseek_md
        elif looks_trivial(deepseek_md) and not looks_trivial(paddle_md):
            final_md = paddle_md
        else:
            sim = similarity(paddle_md, deepseek_md)
            if args.only_run_merger_when_different and sim >= args.similarity_threshold:
                # Pick the "richer" candidate (simple heuristic)
                final_md = paddle_md if len(paddle_md) >= len(deepseek_md) else deepseek_md
            else:
                prompt = build_merger_prompt(paddle_md, deepseek_md)
                out = run_merger_on_page(
                    model=model,
                    processor=processor,
                    page_image_path=page_img,
                    prompt_text=prompt,
                    max_new_tokens=args.max_new_tokens,
                )
                final_md = extract_markdown_from_fenced(out)

        # Always include a top anchor image of the page (so you never “lose” diagrams)
        header = f"<!-- PAGE {pnum:04d} -->\n\n![]({rel_assets}/page.jpg)\n\n"
        final_md = header + final_md.strip() + "\n"

        write_text(out_page_md, final_md)

        # Add to combined book.md; fix ../assets -> assets when concatenating at root
        combined_chunks.append("\n\n" + final_md.replace("(../assets/", "(assets/"))

        print(f"[OK] Page {pnum:04d} -> {out_page_md}")

    out_root.mkdir(parents=True, exist_ok=True)
    write_text(out_book, "\n".join(combined_chunks).strip() + "\n")
    print(f"[OK] Combined book -> {out_book}")


if __name__ == "__main__":
    main()