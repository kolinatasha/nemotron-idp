"""Ingestion and structure-preserving extraction utilities.

This module provides a high-level, production-oriented skeleton for:
- Running nv-ingest (NeMo Retriever) in library mode to extract layout-aware text
- Detecting and cropping charts (YOLOX or other detector)
- Parsing tables into row/column-accurate Markdown/CSV (Camelot/Tabula)

Fill in model endpoints and detector weights as appropriate for your environment.
"""
from typing import List, Dict, Any
import os
from pathlib import Path

try:
    # nv-ingest (NeMo Retriever) expected API
    import nv_ingest
except Exception:
    nv_ingest = None

from PIL import Image
import cv2
import pandas as pd

# Optional table parser
try:
    import camelot
except Exception:
    camelot = None

# Optional detector placeholder (user can replace with YOLOX/Detectron)
try:
    import yolox
except Exception:
    yolox = None


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def ingest_documents(paths: List[str], out_dir: str) -> List[Dict[str, Any]]:
    """Ingest documents and produce structured outputs.

    Returns a list of document records: {id, text_blocks, images: [{type, bbox, path}], tables: [...]}
    """
    ensure_dir(out_dir)
    records = []
    for p in paths:
        doc_id = Path(p).stem
        out_sub = Path(out_dir) / doc_id
        ensure_dir(str(out_sub))

        # 1) Layout-aware OCR / text extraction via nv-ingest (NeMo Retriever)
        structured = {"text": "", "blocks": [], "tables": [], "images": []}
        if nv_ingest is not None:
            try:
                # nv-ingest has a few possible entrypoints; try common-sense ones and fallback
                if hasattr(nv_ingest, 'parse_document'):
                    structured = nv_ingest.parse_document(p, preserve_layout=True)
                elif hasattr(nv_ingest, 'ingest_document'):
                    structured = nv_ingest.ingest_document(p, preserve_layout=True)
                else:
                    # best-effort call
                    structured = nv_ingest.parse(p)
            except Exception:
                # Don't hard-fail ingestion; capture minimal metadata
                structured = {"text": "", "blocks": [], "tables": [], "images": []}
        else:
            # Fallback naive extraction (text only) — keep structure empty so downstream can still run
            try:
                with open(p, 'rb') as fh:
                    content = fh.read()
                structured = {"text": "", "blocks": [], "tables": [], "images": []}
            except Exception:
                structured = {"text": "", "blocks": [], "tables": [], "images": []}

        # 2) Parse tables (PDFs): use Camelot when available
        tables_out = []
        if camelot is not None and str(p).lower().endswith('.pdf'):
            try:
                # Use 'stream' flavor for layout-preserving tables and fallback to lattice when needed
                tables = camelot.read_pdf(p, pages='all', flavor='stream')
                for i, t in enumerate(tables):
                    csv_path = out_sub / f"table_{i}.csv"
                    t.df.to_csv(str(csv_path), index=False)
                    tables_out.append({"index": i, "csv": str(csv_path), "shape": t.df.shape})
            except Exception:
                tables_out = []
        else:
            # If nv-ingest returned table structures, include them
            if structured.get('tables'):
                tables_out = structured.get('tables')

        # 3) Detect and crop charts/figures. If nv-ingest provided images, save them.
        charts_dir = out_sub / "charts"
        ensure_dir(str(charts_dir))
        images_out = []
        # If nv-ingest returned image entries, attempt to save referenced images
        if structured.get('images'):
            for i, img_meta in enumerate(structured.get('images')):
                # img_meta may contain {path, bbox, type}
                try:
                    src = img_meta.get('path') or img_meta.get('uri')
                    if src and os.path.exists(src):
                        dst = charts_dir / f"image_{i}{Path(src).suffix}"
                        if 'bbox' in img_meta:
                            crop_chart(src, img_meta['bbox'], str(dst))
                        else:
                            Image.open(src).save(str(dst))
                        images_out.append({"path": str(dst), "meta": img_meta})
                except Exception:
                    continue

        # For PDFs without embedded images, optionally run a detector on rendered pages (user can enable)
        # Detector integration is left as a user-configurable step (YOLOX / Detectron)

        record = {
            "id": doc_id,
            "source_path": p,
            "structured": structured,
            "tables": tables_out,
            "images": images_out,
        }
        records.append(record)
    return records


def crop_chart(image_path: str, bbox: List[int], out_path: str):
    img = Image.open(image_path)
    # bbox expected as [x1, y1, x2, y2]
    x1, y1, x2, y2 = bbox
    cropped = img.crop((x1, y1, x2, y2))
    cropped.save(out_path)


def parse_tables_with_camelot(pdf_path: str, out_dir: str) -> List[Dict[str, Any]]:
    """Extract tables from PDF using Camelot and write CSVs to out_dir."""
    ensure_dir(out_dir)
    results = []
    if camelot is None:
        return results
    try:
        tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream')
        for i, t in enumerate(tables):
            csv_path = Path(out_dir) / f"table_{i}.csv"
            t.df.to_csv(str(csv_path), index=False)
            results.append({"index": i, "csv": str(csv_path), "shape": t.df.shape})
    except Exception:
        return results
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs='+', help="Document files to ingest")
    parser.add_argument("--out", default="./data/ingested", help="Output directory")
    args = parser.parse_args()
    ingest_documents(args.files, args.out)
