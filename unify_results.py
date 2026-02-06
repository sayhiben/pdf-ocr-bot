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
import json
import logging
import os
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


LOG = logging.getLogger("unify_results")


def setup_logging() -> None:
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


IMG_MD_RE = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')
IMG_HTML_RE = re.compile(r'<img[^>]*src=["\']([^"\']+)["\'][^>]*>', re.IGNORECASE)
LINK_SPAN_RE = re.compile(r'!?\[[^\]]*]\([^)]*\)')
PAGE_REF_RE = re.compile(
    r"\b(?P<prefix>pages?|pgs?|pp?)"
    r"(?P<dot>\.)?"
    r"(?P<gap>\s*)"
    r"(?P<start>(?:\d{1,4}|[ivxlcdm]{1,8}))"
    r"(?P<range>(?P<sep>\s*[-–]\s*)(?P<end>(?:\d{1,4}|[ivxlcdm]{1,8})))?",
    re.IGNORECASE,
)
HEADER_LINE_RE = re.compile(r"<!--\s*PAGE(?:_INDEX|_LABEL)?\b.*?-->", re.IGNORECASE)
ANCHOR_LINE_RE = re.compile(r'<a\s+id="page-[^"]+"\s*></a>', re.IGNORECASE)


def parse_page_range(expr: str, max_page: Optional[int] = None) -> List[int]:
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
            a = a.strip()
            b = b.strip()
            start = int(a) if a else 1
            if b:
                end = int(b)
            else:
                if max_page is None or max_page <= 0:
                    raise ValueError("Open-ended page ranges require existing page images.")
                end = max_page
            step = 1 if end >= start else -1
            for p in range(start, end + step, step):
                pages.add(p)
        else:
            pages.add(int(part))
    return sorted(pages)


def discover_page_images(pages_dir: Path) -> Dict[int, Path]:
    """
    Returns a map of page number -> image path, preferring jpg/jpeg over png/webp.
    """
    chosen: Dict[int, Path] = {}
    for ext in ("jpg", "jpeg", "png", "webp"):
        for p in pages_dir.glob(f"page_*.{ext}"):
            m = re.search(r"page_(\d+)\.", p.name)
            if not m:
                continue
            num = int(m.group(1))
            if num not in chosen:
                chosen[num] = p
    return chosen


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def normalize_page_label(label: str) -> str:
    raw = label.strip().lower()
    if raw.isdigit():
        return str(int(raw))
    norm = re.sub(r"[^a-z0-9]+", "-", raw).strip("-")
    if norm.isdigit():
        return str(int(norm))
    return norm


def make_anchor_id(label_norm: Optional[str], page_num: int) -> str:
    if label_norm:
        return f"page-{label_norm}-p{page_num:04d}"
    return f"page-p{page_num:04d}"


def build_page_header(page_num: int, page_label: Optional[str]) -> str:
    lines = [f"<!-- PAGE_INDEX {page_num:04d} -->"]
    if page_label:
        lines.append(f"<!-- PAGE_LABEL {page_label} -->")
    return "\n".join(lines) + "\n\n"


def is_page_image_line(line: str) -> bool:
    m = IMG_MD_RE.search(line)
    if not m:
        return False
    inside = m.group(2).strip()
    path = inside.split()[0]
    if "assets/page_" not in path:
        return False
    name = Path(path).name.lower()
    return name.startswith("page.")


def strip_existing_header(md_text: str) -> str:
    lines = md_text.splitlines()
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1

    saw_header = False
    while i < len(lines) and HEADER_LINE_RE.match(lines[i].strip()):
        saw_header = True
        i += 1
        while i < len(lines) and not lines[i].strip():
            i += 1

    if i < len(lines) and ANCHOR_LINE_RE.match(lines[i].strip()):
        saw_header = True
        i += 1
        while i < len(lines) and not lines[i].strip():
            i += 1

    if saw_header and i < len(lines) and is_page_image_line(lines[i].strip()):
        i += 1
        while i < len(lines) and not lines[i].strip():
            i += 1

    return "\n".join(lines[i:]).lstrip("\n")


def detect_page_label(md_text: str) -> Optional[str]:
    body = strip_existing_header(md_text)
    lines = []
    in_code = False
    for raw in body.splitlines():
        s = raw.strip()
        if s.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        if not s:
            continue
        if s.startswith("<!--"):
            continue
        if s.startswith("!"):
            continue
        lines.append(s)

    def _match_label(line: str, *, allow_loose: bool) -> Optional[str]:
        s = line.strip()
        s = re.sub(r"^[\W_]+|[\W_]+$", "", s)
        if not s:
            return None
        m = re.match(r"^(?:page|p\.?)\s*(\d{1,4})$", s, re.IGNORECASE)
        if m:
            return str(int(m.group(1)))
        m = re.match(r"^(\d{1,4})$", s)
        if m:
            return str(int(m.group(1)))
        m = re.match(r"^([ivxlcdm]{1,8})$", s, re.IGNORECASE)
        if m:
            return m.group(1).upper()
        if allow_loose:
            m = re.search(r"(?:^|\s)(\d{1,4})$", s)
            if m:
                return str(int(m.group(1)))
            m = re.search(r"(?:^|\s)([ivxlcdm]{1,8})$", s, re.IGNORECASE)
            if m:
                return m.group(1).upper()
        return None

    # Prefer footer-style numbers
    tail = list(reversed(lines[-6:]))
    for line in tail:
        label = _match_label(line, allow_loose=True)
        if label:
            return label

    head = lines[:6]
    for line in head:
        label = _match_label(line, allow_loose=False)
        if label:
            return label

    return None


def choose_page_label(*candidates: Optional[str]) -> Optional[str]:
    for text in candidates:
        if not text:
            continue
        label = detect_page_label(text)
        if label:
            return label
    return None


def rewrite_page_refs(
    md_text: str,
    *,
    label_to_page: Dict[str, int],
    page_to_label_norm: Dict[int, str],
    link_mode: str,
) -> str:
    def anchor_for_page(pnum: int) -> str:
        return make_anchor_id(page_to_label_norm.get(pnum), pnum)

    def link_target(pnum: int) -> str:
        anchor = anchor_for_page(pnum)
        if link_mode == "combined":
            return f"#{anchor}"
        return f"page_{pnum:04d}.md#{anchor}"

    def replace_match(m: re.Match, spans: List[Tuple[int, int]]) -> str:
        start_idx = m.start()
        for s, e in spans:
            if s <= start_idx < e:
                return m.group(0)

        prefix = m.group("prefix")
        dot = m.group("dot") or ""
        gap = m.group("gap") or ""
        start = m.group("start")
        sep = m.group("sep") or ""
        end = m.group("end")

        def _link(num_str: str) -> Optional[str]:
            raw = num_str.strip()
            if raw.isdigit():
                norm = str(int(raw))
            else:
                norm = normalize_page_label(raw)
            if not norm:
                return None
            dest = label_to_page.get(norm)
            if dest is None:
                return None
            return f"[{num_str}]({link_target(dest)})"

        start_link = _link(start) or start
        if end:
            end_link = _link(end) or end
            return f"{prefix}{dot}{gap}{start_link}{sep}{end_link}"
        return f"{prefix}{dot}{gap}{start_link}"

    out_lines: List[str] = []
    in_code = False
    for line in md_text.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith("```"):
            in_code = not in_code
            out_lines.append(line)
            continue
        if in_code:
            out_lines.append(line)
            continue
        spans = [(m.start(), m.end()) for m in LINK_SPAN_RE.finditer(line)]
        new_line = PAGE_REF_RE.sub(lambda m: replace_match(m, spans), line)
        out_lines.append(new_line)
    return "".join(out_lines)


PAGE_FILE_LINK_RE = re.compile(
    r"\((?:\./|../|[^/)]+/)?page_\d{4}\.md#(?P<anchor>page-[^)]+)\)"
)


def rewrite_links_for_combined(md_text: str) -> str:
    md_text = PAGE_FILE_LINK_RE.sub(lambda m: f"(#{m.group('anchor')})", md_text)
    md_text = re.sub(r"\((?:\./)?\.\./assets/", "(assets/", md_text)
    md_text = re.sub(
        r'(<img[^>]*\bsrc=["\'])(?:\./)?\.\./assets/',
        r"\1assets/",
        md_text,
        flags=re.IGNORECASE,
    )
    return md_text


def extract_linked_assets(md_text: str, rel_assets: str, asset_dir: Path) -> set[str]:
    keep: set[str] = set()

    def _handle(path_str: str) -> None:
        tokens = path_str.strip().split()
        if not tokens:
            return
        path = tokens[0]
        if path.startswith(rel_assets + "/"):
            name = path[len(rel_assets) + 1:]
            if name:
                keep.add(name)
            return
        # If the link is just a filename and it exists in this asset dir, keep it.
        if "/" not in path and (asset_dir / path).exists():
            keep.add(path)

    for m in IMG_MD_RE.finditer(md_text):
        _handle(m.group(2))
    for m in IMG_HTML_RE.finditer(md_text):
        _handle(m.group(1))
    return keep


def prune_unlinked_assets(asset_dir: Path, keep: set[str]) -> None:
    if not asset_dir.exists():
        return
    for p in asset_dir.iterdir():
        if not p.is_file():
            continue
        if p.name not in keep:
            p.unlink()

def looks_trivial(md: str) -> bool:
    stripped = re.sub(r"\s+", "", md)
    return len(stripped) < 80


def similarity(a: str, b: str, limit_chars: int = 20000) -> float:
    a0 = a[:limit_chars]
    b0 = b[:limit_chars]
    return difflib.SequenceMatcher(a=a0, b=b0).ratio()


def diff_stats(a: str, b: str) -> Dict[str, int]:
    a_lines = a.splitlines()
    b_lines = b.splitlines()
    sm = difflib.SequenceMatcher(a=a_lines, b=b_lines)
    counts = {
        "replace": 0,
        "delete": 0,
        "insert": 0,
        "equal": 0,
        "a_lines": len(a_lines),
        "b_lines": len(b_lines),
    }
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "replace":
            counts["replace"] += max(i2 - i1, j2 - j1)
        elif tag == "delete":
            counts["delete"] += (i2 - i1)
        elif tag == "insert":
            counts["insert"] += (j2 - j1)
        elif tag == "equal":
            counts["equal"] += (i2 - i1)
    return counts


def write_diff_file(
    diff_dir: Path,
    page_num: int,
    a: str,
    b: str,
    *,
    context: int,
    max_lines: int,
) -> Optional[Path]:
    if not diff_dir:
        return None
    diff_dir.mkdir(parents=True, exist_ok=True)
    a_lines = a.splitlines()
    b_lines = b.splitlines()
    diff_iter = difflib.unified_diff(
        a_lines,
        b_lines,
        fromfile=f"paddle/page_{page_num:04d}.md",
        tofile=f"deepseek/page_{page_num:04d}.md",
        lineterm="",
        n=context,
    )
    lines: List[str] = []
    for line in diff_iter:
        lines.append(line)
        if max_lines > 0 and len(lines) >= max_lines:
            lines.append(f"... (truncated at {max_lines} lines)")
            break
    out_path = diff_dir / f"page_{page_num:04d}.diff"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


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


def repetition_signal(md: str) -> Optional[str]:
    lines = []
    for raw in md.splitlines():
        s = re.sub(r"\s+", " ", raw.strip())
        if not s:
            continue
        if s.startswith("<!--") or s.startswith("```"):
            continue
        lines.append(s)

    if len(lines) < 10:
        return None

    counts = Counter(lines)
    most_common_line, count = counts.most_common(1)[0]
    ratio = count / max(1, len(lines))
    repeated_ratio = sum(c for c in counts.values() if c >= 3) / max(1, len(lines))

    if count >= 6 and (ratio >= 0.25 or repeated_ratio >= 0.5):
        preview = most_common_line
        if len(preview) > 80:
            preview = preview[:77] + "..."
        return f"repetition detected: '{preview}' repeats {count}x ({ratio:.0%} of lines)"
    return None


def build_merger_prompt(paddle_md: str, deepseek_md: str) -> str:
    # Strong constraints help reduce hallucination and keep image links valid.
    quality_notes = []
    paddle_rep = repetition_signal(paddle_md)
    deepseek_rep = repetition_signal(deepseek_md)
    if paddle_rep:
        quality_notes.append(f"Candidate A issue: {paddle_rep}. Treat Candidate A as unreliable for repeated sections.")
    if deepseek_rep:
        quality_notes.append(f"Candidate B issue: {deepseek_rep}. Treat Candidate B as unreliable for repeated sections.")
    quality_block = ""
    if quality_notes:
        quality_block = "Quality notes:\n- " + "\n- ".join(quality_notes) + "\n"

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
7) If the page is a categorized list or "table of categories" (like a skills list), output it as a hierarchical unordered list.
   Use top-level bullets for main categories and nested bullets for subcategories/items. Preserve column order (left-to-right, top-to-bottom).
8) If the page contains charts/diagrams/flowcharts/schematics, keep the original image link for them.
   Do NOT attempt to recreate charts/diagrams in Mermaid or ASCII.
9) If one candidate is clearly corrupted (e.g., repeated lines, garbled tokens, duplicated blocks), ignore the corrupted portions and rely on the other candidate and the page image.
10) CRITICAL: Use ONLY image links that already appear in Candidate A or Candidate B. Do not create new filenames.

{quality_block}

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
    setup_logging()
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
    ap.add_argument("--report-dir", default="", type=str,
                    help="If set, write per-page merge reports and candidate diffs to this directory.")
    ap.add_argument("--diff-context", default=3, type=int, help="Context lines for candidate diff files.")
    ap.add_argument("--diff-max-lines", default=400, type=int, help="Max lines to write per diff file (0 = no limit).")

    args = ap.parse_args()

    pages_dir = Path(args.pages_dir)
    paddle_root = Path(args.paddle_root)
    deepseek_root = Path(args.deepseek_root)
    out_root = Path(args.out_root)

    LOG.info("Pages dir=%s", pages_dir)
    LOG.info("Paddle root=%s DeepSeek root=%s", paddle_root, deepseek_root)
    LOG.info("Out root=%s Model=%s", out_root, args.model)

    out_pages_dir = out_root / "pages"
    out_assets_dir = out_root / "assets"
    out_book = out_root / "book.md"
    report_dir = Path(args.report_dir) if args.report_dir else None
    diff_dir = (report_dir / "diffs") if report_dir else None
    if report_dir:
        report_dir.mkdir(parents=True, exist_ok=True)
        if diff_dir:
            diff_dir.mkdir(parents=True, exist_ok=True)
        LOG.info("Reports dir=%s", report_dir)

    # Determine pages to process
    page_img_map = discover_page_images(pages_dir)
    max_page = max(page_img_map) if page_img_map else 0
    if args.page_range.strip():
        try:
            pages = parse_page_range(args.page_range, max_page=max_page)
        except ValueError as ex:
            raise SystemExit(str(ex))
        page_imgs = []
        for p in pages:
            img = page_img_map.get(p)
            if img is None:
                LOG.warning("Missing page image for %04d in %s; skipping", p, pages_dir)
                continue
            page_imgs.append(img)
    else:
        page_imgs = [page_img_map[p] for p in sorted(page_img_map)]

    if not page_imgs:
        raise SystemExit(f"No page images found in {pages_dir}")

    # Load merger model once
    model, processor = load_merger(args.model, args.prefer_flash_attn, args.min_pixels, args.max_pixels)
    total_pages = len(page_imgs)
    LOG.info("Processing %d pages", total_pages)

    combined_chunks: List[str] = []
    decision_counts: Dict[str, int] = {}
    skipped_existing = 0
    skipped_missing_candidates = 0
    processed_pages = 0
    page_records: List[Dict[str, object]] = []

    for idx, page_img in enumerate(page_imgs, start=1):
        m = re.search(r"page_(\d+)\.", page_img.name)
        if not m:
            continue
        pnum = int(m.group(1))
        LOG.info("Merge page %04d (%d/%d)", pnum, idx, total_pages)

        paddle_page_dir = paddle_root / f"page_{pnum:04d}"
        deepseek_page_dir = deepseek_root / f"page_{pnum:04d}"

        out_page_md = out_pages_dir / f"page_{pnum:04d}.md"
        out_asset_page_dir = out_assets_dir / f"page_{pnum:04d}"

        if out_page_md.exists() and not args.overwrite:
            page_md_text = read_text(out_page_md)
            page_label = detect_page_label(page_md_text)
            page_records.append({
                "pnum": pnum,
                "out_page_md": out_page_md,
                "page_label": page_label,
            })
            LOG.debug("Skip page %04d (final exists)", pnum)
            skipped_existing += 1
            continue

        # Load candidates
        paddle_md_path = find_candidate_markdown_paddle(paddle_page_dir) if paddle_page_dir.exists() else None
        deepseek_md_path = find_candidate_markdown_deepseek(deepseek_page_dir) if deepseek_page_dir.exists() else None
        if not paddle_md_path:
            LOG.debug("Page %04d: no Paddle markdown found", pnum)
        if not deepseek_md_path:
            LOG.debug("Page %04d: no DeepSeek markdown found", pnum)

        paddle_md_raw = read_text(paddle_md_path) if paddle_md_path else ""
        deepseek_md_raw = read_text(deepseek_md_path) if deepseek_md_path else ""

        # Ensure we at least have something
        if not paddle_md_raw and not deepseek_md_raw:
            LOG.warning("Page %04d: no candidates found, skipping.", pnum)
            skipped_missing_candidates += 1
            continue

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
        paddle_trivial = looks_trivial(paddle_md)
        deepseek_trivial = looks_trivial(deepseek_md)
        decision = ""
        used_merger = False
        sim = None
        if paddle_trivial and not deepseek_trivial:
            final_body = deepseek_md
            decision = "deepseek_trivial" if paddle_md_raw else "deepseek_only"
        elif deepseek_trivial and not paddle_trivial:
            final_body = paddle_md
            decision = "paddle_trivial" if deepseek_md_raw else "paddle_only"
        else:
            sim = similarity(paddle_md, deepseek_md)
            if args.only_run_merger_when_different and sim >= args.similarity_threshold:
                # Pick the "richer" candidate (simple heuristic)
                if len(paddle_md) >= len(deepseek_md):
                    final_body = paddle_md
                    decision = "fastpath_similar_paddle"
                else:
                    final_body = deepseek_md
                    decision = "fastpath_similar_deepseek"
                LOG.debug("Page %04d: candidates similar (%.3f), skipping merger", pnum, sim)
            else:
                prompt = build_merger_prompt(paddle_md, deepseek_md)
                out = run_merger_on_page(
                    model=model,
                    processor=processor,
                    page_image_path=page_img,
                    prompt_text=prompt,
                    max_new_tokens=args.max_new_tokens,
                )
                final_body = extract_markdown_from_fenced(out)
                used_merger = True
                decision = "merger"

        final_body = final_body.strip()

        page_label = choose_page_label(final_body, paddle_md_raw, deepseek_md_raw)
        header = build_page_header(pnum, page_label)
        final_md = header + final_body + "\n"

        write_text(out_page_md, final_md)

        keep_assets = extract_linked_assets(final_md, rel_assets, out_asset_page_dir)
        prune_unlinked_assets(out_asset_page_dir, keep_assets)

        page_records.append({
            "pnum": pnum,
            "out_page_md": out_page_md,
            "page_label": page_label,
        })

        decision_counts[decision] = decision_counts.get(decision, 0) + 1
        processed_pages += 1
        if report_dir:
            diff_path = write_diff_file(
                diff_dir,
                pnum,
                paddle_md,
                deepseek_md,
                context=args.diff_context,
                max_lines=args.diff_max_lines,
            )
            report = {
                "page": pnum,
                "page_image": str(page_img),
                "paddle_path": str(paddle_md_path) if paddle_md_path else None,
                "deepseek_path": str(deepseek_md_path) if deepseek_md_path else None,
                "decision": decision,
                "used_merger": used_merger,
                "similarity": sim,
                "only_run_merger_when_different": args.only_run_merger_when_different,
                "similarity_threshold": args.similarity_threshold,
                "trivial": {"paddle": paddle_trivial, "deepseek": deepseek_trivial},
                "lengths": {
                    "paddle_chars": len(paddle_md),
                    "deepseek_chars": len(deepseek_md),
                    "final_chars": len(final_body),
                },
                "diff_stats_candidates": diff_stats(paddle_md, deepseek_md),
                "diff_stats_final_vs_paddle": diff_stats(final_body, paddle_md),
                "diff_stats_final_vs_deepseek": diff_stats(final_body, deepseek_md),
                "diff_file": diff_path.name if diff_path else None,
                "page_label": page_label,
            }
            report_path = report_dir / f"page_{pnum:04d}.json"
            write_json(report_path, report)
            LOG.info("Page %04d: decision=%s sim=%s report=%s", pnum, decision, f"{sim:.3f}" if sim is not None else "n/a", report_path)

        LOG.info("Page %04d -> %s", pnum, out_page_md)

    label_to_page: Dict[str, int] = {}
    page_to_label_norm: Dict[int, str] = {}
    for rec in page_records:
        pnum = int(rec["pnum"])
        label = rec.get("page_label")
        if isinstance(label, str) and label.strip():
            norm = normalize_page_label(label)
            if norm:
                page_to_label_norm[pnum] = norm
                if norm in label_to_page and label_to_page[norm] != pnum:
                    LOG.warning(
                        "Duplicate page label '%s' found on pages %04d and %04d; keeping first.",
                        label,
                        label_to_page[norm],
                        pnum,
                    )
                else:
                    label_to_page[norm] = pnum

    for rec in page_records:
        pnum = int(rec["pnum"])
        out_page_md = Path(rec["out_page_md"])
        label = rec.get("page_label")
        page_md_text = read_text(out_page_md)
        body = strip_existing_header(page_md_text)
        header = build_page_header(pnum, label if isinstance(label, str) else None)
        final_md = header + body.strip() + "\n"
        write_text(out_page_md, final_md)

        out_asset_page_dir = out_assets_dir / f"page_{pnum:04d}"
        rel_assets = f"../assets/page_{pnum:04d}"
        keep_assets = extract_linked_assets(final_md, rel_assets, out_asset_page_dir)
        prune_unlinked_assets(out_asset_page_dir, keep_assets)

    for rec in page_records:
        out_page_md = Path(rec["out_page_md"])
        page_md_text = read_text(out_page_md)
        page_md_text = rewrite_links_for_combined(page_md_text)
        combined_chunks.append("\n\n" + page_md_text.strip())

    out_root.mkdir(parents=True, exist_ok=True)
    write_text(out_book, "\n".join(combined_chunks).strip() + "\n")
    LOG.info("Combined book -> %s", out_book)
    if report_dir:
        summary = {
            "total_pages": total_pages,
            "processed_pages": processed_pages,
            "skipped_existing": skipped_existing,
            "skipped_missing_candidates": skipped_missing_candidates,
            "decision_counts": decision_counts,
            "model": args.model,
        }
        write_json(report_dir / "summary.json", summary)
        LOG.info("Report summary -> %s", report_dir / "summary.json")


if __name__ == "__main__":
    main()
