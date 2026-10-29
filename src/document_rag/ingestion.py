"""Docling-based PDF ingestion with compact progress and page provenance."""

import hashlib
import json
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableFormerMode,
)
from docling.datamodel.progress import (
    PageCompletedProgress,
    DocumentCompletedProgress,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem

from .config import (
    DOCLING_ENABLE_OCR,
    DOCLING_MODEL_DIR,
    DOCLING_OUTPUT_DIR,
    DOCLING_TABLE_MODE,
    FIGURES_DIR,
    PAGE_IMAGE_SCALE,
    PROJECT_ROOT,
)


def log_docling_progress(event: Any) -> None:
    """Update one terminal line with completed PDF page counts."""
    if isinstance(event, PageCompletedProgress):
        total = max(event.total_pages, 1)
        completed = event.completed_pages
        percentage = int(completed / total * 100)

        print(
            f"\r[1/5] Converting PDF: {completed}/{total} pages "
            f"({percentage}%)",
            end="",
            flush=True,
        )

    elif isinstance(event, DocumentCompletedProgress):
        status = getattr(event.status, "value", str(event.status))
        print(
            f"\r[1/5] Conversion finished: {status}".ljust(70),
            flush=True,
        )


def calculate_sha256(file_path: Path) -> str:
    """Calculate a stable SHA-256 hash of the source PDF."""
    hasher = hashlib.sha256()

    with file_path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(block)

    return hasher.hexdigest()


def create_converter() -> DocumentConverter:
    """Create a Docling converter using local models and configured OCR."""
    pipeline_options = PdfPipelineOptions(
        artifacts_path=(
            str(DOCLING_MODEL_DIR)
            if DOCLING_MODEL_DIR.exists()
            else None
        ),
        do_ocr=DOCLING_ENABLE_OCR,
        do_table_structure=True,
        generate_page_images=True,
        images_scale=PAGE_IMAGE_SCALE,
    )

    table_modes = {
        "FAST": TableFormerMode.FAST,
        "ACCURATE": TableFormerMode.ACCURATE,
    }

    if DOCLING_TABLE_MODE not in table_modes:
        raise ValueError("DOCLING_TABLE_MODE must be FAST or ACCURATE.")

    pipeline_options.table_structure_options.mode = (
        table_modes[DOCLING_TABLE_MODE]
    )

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            ),
        },
        show_progress=False,
    )


def relative_project_path(path: Path) -> str:
    """Return a project-relative path with portable separators."""
    return path.resolve().relative_to(
        PROJECT_ROOT.resolve()
    ).as_posix()


def save_page_images(
    document: Any,
    image_dir: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Save page previews and record their physical PDF page numbers."""
    image_dir.mkdir(parents=True, exist_ok=True)

    page_images = []
    warnings = []
    pages = list(document.pages.items())
    total_pages = len(pages)

    for index, (page_key, page) in enumerate(pages, start=1):
        print(
            f"\r[4/5] Saving page previews: {index}/{total_pages}",
            end="",
            flush=True,
        )

        page_number = int(page_key)
        page_image_ref = getattr(page, "image", None)
        image = getattr(page_image_ref, "pil_image", None)

        if image is None:
            warnings.append(
                f"No rendered image available for PDF page {page_number}."
            )
            continue

        image_path = image_dir / f"page_{page_number:04d}.png"
        image.save(image_path, format="PNG")

        page_images.append(
            {
                "page_number": page_number,
                "path": relative_project_path(image_path),
            }
        )

    print(
        f"\r[4/5] Page previews complete: "
        f"{len(page_images)}/{total_pages} saved.".ljust(70),
        flush=True,
    )

    return page_images, warnings


def save_visual_items(
    document: Any,
    image_dir: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Save renderable table and picture crops with page provenance."""
    image_dir.mkdir(parents=True, exist_ok=True)

    visual_assets = []
    warnings = []
    counters = {"table": 0, "picture": 0}

    visual_items = [
        (element, level)
        for element, level in document.iterate_items()
        if isinstance(element, (TableItem, PictureItem))
    ]

    total_items = len(visual_items)

    for index, (element, _level) in enumerate(visual_items, start=1):
        print(
            f"\r[5/5] Saving visual assets: {index}/{total_items}",
            end="",
            flush=True,
        )

        asset_type = (
            "table" if isinstance(element, TableItem) else "picture"
        )
        counters[asset_type] += 1

        page_numbers = sorted(
            {
                int(provenance.page_no)
                for provenance in getattr(element, "prov", [])
                if getattr(provenance, "page_no", None) is not None
            }
        )

        try:
            image = element.get_image(document)

            if image is None:
                warnings.append(
                    f"No rendered image for {asset_type} "
                    f"{counters[asset_type]}."
                )
                continue

            page_tag = (
                f"page_{page_numbers[0]:04d}"
                if page_numbers
                else "page_unknown"
            )

            image_path = image_dir / (
                f"{asset_type}_{counters[asset_type]:04d}_"
                f"{page_tag}.png"
            )
            image.save(image_path, format="PNG")

            visual_assets.append(
                {
                    "asset_type": asset_type,
                    "page_numbers": page_numbers,
                    "path": relative_project_path(image_path),
                }
            )

        except Exception as exc:
            warnings.append(
                f"Failed to save {asset_type} "
                f"{counters[asset_type]}: {exc}"
            )

    print(
        f"\r[5/5] Visual assets complete: "
        f"{len(visual_assets)}/{total_items} saved.".ljust(70),
        flush=True,
    )

    return visual_assets, warnings


def ingest_pdf(
    pdf_path: Path,
    converter: DocumentConverter | None = None,
    on_progress: Callable[[str, float | None], None] | None = None,
) -> dict[str, Any]:
    """Convert a PDF and save structured data, images, and its manifest."""
    pdf_path = pdf_path.expanduser().resolve()

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file: {pdf_path.name}")

    overall_start = time.perf_counter()
    if on_progress is not None:
        on_progress("Preparing PDF extraction", 0.0)

    print(f"\nIngesting: {pdf_path.name}", flush=True)
    print(
        f"OCR: {'enabled' if DOCLING_ENABLE_OCR else 'disabled'} | "
        f"Table mode: {DOCLING_TABLE_MODE}",
        flush=True,
    )

    # Initialize the converter only when one was not supplied.
    if converter is None:
        converter = create_converter()

    # Generate a content-based ID for stable document references.
    content_hash = calculate_sha256(pdf_path)
    document_id = f"doc_{content_hash[:24]}"

    document_dir = DOCLING_OUTPUT_DIR / document_id
    docling_assets_dir = document_dir / "docling_assets"
    page_image_dir = FIGURES_DIR / document_id

    document_dir.mkdir(parents=True, exist_ok=True)
    docling_assets_dir.mkdir(parents=True, exist_ok=True)
    page_image_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Run Docling with compact page-level progress.
    conversion_start = time.perf_counter()

    def report_conversion_progress(event: Any) -> None:
        log_docling_progress(event)
        if on_progress is None:
            return

        if isinstance(event, PageCompletedProgress):
            total = max(event.total_pages, 1)
            completed = event.completed_pages
            on_progress(
                f"Extracting PDF pages: {completed}/{total}",
                0.05 + 0.55 * min(completed / total, 1.0),
            )
        elif isinstance(event, DocumentCompletedProgress):
            on_progress("PDF extraction complete; saving document assets", 0.62)

    conversion_result = converter.convert(
        str(pdf_path),
        progress_callback=report_conversion_progress,
    )

    conversion_seconds = round(
        time.perf_counter() - conversion_start, 2
    )
    document = conversion_result.document

    if document is None:
        raise RuntimeError(
            f"Docling produced no document for {pdf_path.name}."
        )

    # Step 2: Save the canonical structured representation.
    if on_progress is not None:
        on_progress("Saving extracted document structure and assets", 0.65)
    print("[2/5] Saving native Docling JSON...", flush=True)
    json_path = document_dir / "document.json"

    document.save_as_json(
        json_path,
        artifacts_dir=docling_assets_dir,
        image_mode=ImageRefMode.REFERENCED,
        indent=2,
        ensure_ascii=False,
    )

    # Step 3: Save a readable Markdown representation.
    print("[3/5] Exporting Markdown...", flush=True)
    markdown_path = document_dir / "document.md"

    markdown_path.write_text(
        document.export_to_markdown(),
        encoding="utf-8",
    )

    # Step 4: Save the source-page previews.
    page_images, page_warnings = save_page_images(
        document,
        page_image_dir,
    )

    # Step 5: Save renderable table and picture crops.
    visual_assets, visual_warnings = save_visual_items(
        document,
        page_image_dir,
    )

    duration_seconds = round(
        time.perf_counter() - overall_start, 2
    )
    if on_progress is not None:
        on_progress("PDF extraction and asset saving complete", 0.75)

    conversion_status = getattr(
        conversion_result.status,
        "value",
        str(conversion_result.status),
    )

    # Persist the metadata required for citations and UI image mapping.
    manifest = {
        "document_id": document_id,
        "source_file": pdf_path.name,
        "content_sha256": content_hash,
        "docling_version": version("docling"),
        "conversion_status": conversion_status,
        "page_count": len(document.pages),
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "conversion_seconds": conversion_seconds,
        "processing_seconds": duration_seconds,
        "ocr_enabled": DOCLING_ENABLE_OCR,
        "table_mode": DOCLING_TABLE_MODE,
        "outputs": {
            "native_json": relative_project_path(json_path),
            "markdown": relative_project_path(markdown_path),
        },
        "page_images": page_images,
        "visual_assets": visual_assets,
        "conversion_errors": [
            str(error) for error in conversion_result.errors
        ],
        "warnings": page_warnings + visual_warnings,
    }

    manifest_path = document_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Print only a compact final summary.
    print(
        f"\nCompleted: {conversion_status} | {document_id}",
        flush=True,
    )
    print(
        f"Pages: {len(document.pages)} | "
        f"Previews: {len(page_images)}/{len(document.pages)} | "
        f"Visual assets: {len(visual_assets)}",
        flush=True,
    )
    print(
        f"Conversion: {conversion_seconds:.1f}s | "
        f"Total: {duration_seconds:.1f}s | "
        f"Warnings: {len(manifest['warnings'])} | "
        f"Errors: {len(manifest['conversion_errors'])}",
        flush=True,
    )
    print(f"Manifest: {manifest_path}", flush=True)

    return manifest