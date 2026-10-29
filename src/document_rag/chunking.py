"""Type-aware chunking for Docling JSON documents."""

import argparse
import json
import re
from pathlib import Path
from statistics import mean
from typing import Callable

from transformers import AutoTokenizer

from .config import CHUNKS_DIR, DOCLING_OUTPUT_DIR

TOKENIZER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TEXT_TOKENS = 220
MAX_TABLE_TOKENS = 220


def load_json(path):
    """Load a UTF-8 JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_reference_index(document):
    """Map Docling self references to their collection type and item."""
    index = {}
    for collection in (
        "texts", "tables", "pictures", "groups",
        "key_value_items", "form_items",
    ):
        for item in document.get(collection, []):
            reference = item.get("self_ref")
            if reference:
                index[reference] = (collection, item)
    return index


def resolve_ref(ref, index):
    """Resolve a Docling reference such as #/texts/3."""
    if isinstance(ref, dict):
        ref = ref.get("$ref")
    return index.get(ref) if isinstance(ref, str) else None


def ordered_elements(document, index):
    """Walk body references in reading order, expanding groups recursively."""
    visited = set()

    def walk(ref):
        """Yield individual text, table and image elements once."""
        reference = ref.get("$ref") if isinstance(ref, dict) else ref
        if not isinstance(reference, str) or reference in visited:
            return

        resolved = index.get(reference)
        if not resolved:
            return

        collection, item = resolved
        visited.add(reference)

        if collection == "groups":
            for child in item.get("children", []):
                yield from walk(child)
        elif collection == "texts":
            yield "text", item
        elif collection == "tables":
            yield "table", item
        elif collection == "pictures":
            yield "image", item
        elif collection in ("key_value_items", "form_items"):
            text = item.get("text") or item.get("value") or ""
            if text:
                yield "text", {
                    **item,
                    "text": str(text),
                    "label": collection.removesuffix("_items"),
                }

    for ref in document.get("body", {}).get("children", []):
        yield from walk(ref)


def pages_for(item):
    """Return sorted page numbers from an element's provenance."""
    return sorted({
        int(provenance["page_no"])
        for provenance in item.get("prov", [])
        if provenance.get("page_no") is not None
    })


def source_regions(item, element_type=None):
    """Preserve page and bounding-box provenance for source elements."""
    regions = []
    for provenance in item.get("prov", []):
        regions.append({
            "element_type": element_type or item.get("label", "unknown"),
            "page_number": provenance.get("page_no"),
            "bbox": provenance.get("bbox"),
        })
    return regions


def read_text(item):
    """Get the normalized text from a Docling text element."""
    return str(item.get("text") or item.get("orig") or "").strip()


def count_tokens(text, tokenizer):
    """Count tokens without truncating text."""
    return len(
        tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=False,
        )
    )


def heading_context(heading_stack):
    """Return active section headings in hierarchy order."""
    return [
        heading_stack[level]
        for level in sorted(heading_stack)
    ]


def render_text(headings, parts):
    """Add section context without repeating a heading already in the chunk."""
    body = "\n\n".join(
        part.strip() for part in parts if part.strip()
    ).strip()

    if not body:
        return ""

    if not headings:
        return body

    first_line = body.splitlines()[0].strip()
    if first_line in headings:
        return body

    section = " > ".join(headings)
    return f"Section: {section}\n\n{body}"


def split_text_by_tokens(text, tokenizer, max_tokens):
    """Split an oversized text element into token-bounded pieces."""
    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
        truncation=False,
    )

    if len(token_ids) <= max_tokens:
        return [text]

    pieces = []
    for start in range(0, len(token_ids), max_tokens):
        piece_ids = token_ids[start:start + max_tokens]
        piece = tokenizer.decode(
            piece_ids,
            skip_special_tokens=True,
        ).strip()

        if piece:
            pieces.append(piece)

    return pieces


def get_page_assets(manifest, pages):
    """Return manifest page-preview paths matching the supplied pages."""
    assets = manifest.get("page_images", {})
    matches = []

    if isinstance(assets, dict):
        entries = assets.items()
    elif isinstance(assets, list):
        entries = enumerate(assets, start=1)
    else:
        return matches

    for key, asset in entries:
        page = None

        if isinstance(key, int) or str(key).isdigit():
            page = int(key)

        if isinstance(asset, dict):
            page = asset.get(
                "page_number",
                asset.get("page_no", page),
            )
            path = next(
                (
                    asset[k]
                    for k in (
                        "path",
                        "file_path",
                        "relative_path",
                        "file",
                    )
                    if asset.get(k)
                ),
                None,
            )
        else:
            path = asset

        if page in pages and path:
            matches.append(str(path))

    return matches


def get_visual_assets(manifest, pages):
    """Return manifest visual assets for the supplied pages."""
    assets = manifest.get("visual_assets", [])

    if isinstance(assets, dict):
        assets = list(assets.values())

    if not isinstance(assets, list):
        return []

    matches = []
    for asset in assets:
        if not isinstance(asset, dict):
            matches.append(asset)
            continue

        page = asset.get(
            "page_number",
            asset.get("page_no", asset.get("page")),
        )

        if page is None or page in pages:
            matches.append(asset)

    return matches


def make_text_record(document_id, pending, tokenizer, manifest):
    """Build one text chunk with source metadata and section context."""
    text = render_text(
        pending["headings"],
        pending["parts"],
    )

    if not text:
        return None

    pages = sorted(pending["pages"])

    return {
        "document_id": document_id,
        "content_type": "text",
        "text": text,
        "page_numbers": pages,
        "headings": pending["headings"],
        "element_types": sorted(pending["element_types"]),
        "source_regions": pending["source_regions"],
        "page_images": get_page_assets(manifest, pages),
        "visual_assets": get_visual_assets(manifest, pages),
        "token_count": count_tokens(text, tokenizer),
    }


def split_text_for_context(text, headings, tokenizer, max_tokens):
    """Split text further when its section prefix would exceed the token target."""
    prefix = (
        "Section: " + " > ".join(headings)
        if headings else ""
    )
    prefix_tokens = count_tokens(prefix, tokenizer) if prefix else 0
    body_limit = max(
        16,
        max_tokens - min(prefix_tokens + 6, max_tokens - 16),
    )

    initial_pieces = split_text_by_tokens(
        text, tokenizer, body_limit
    )
    final_pieces = []
    queue = list(initial_pieces)

    while queue:
        piece = queue.pop(0)

        if count_tokens(
            render_text(headings, [piece]), tokenizer
        ) <= max_tokens:
            final_pieces.append(piece)
            continue

        ids = tokenizer.encode(
            piece,
            add_special_tokens=False,
            truncation=False,
        )

        if len(ids) <= 16:
            final_pieces.append(piece)
            continue

        midpoint = max(1, len(ids) // 2)
        left = tokenizer.decode(
            ids[:midpoint],
            skip_special_tokens=True,
        ).strip()
        right = tokenizer.decode(
            ids[midpoint:],
            skip_special_tokens=True,
        ).strip()

        if right:
            queue.insert(0, right)
        if left:
            queue.insert(0, left)

    return final_pieces


def append_text_item(
    item,
    heading_stack,
    pending,
    output,
    document_id,
    tokenizer,
    manifest,
):
    """Append an ordinary text element, flushing chunks at safe boundaries."""
    text = read_text(item)
    if not text:
        return

    pages = pages_for(item)
    headings = heading_context(heading_stack)
    label = str(item.get("label", "text"))
    regions = source_regions(item, label)

    if pending["parts"] and (
        pending["pages"] != set(pages)
        or pending["headings"] != headings
    ):
        flush_text(
            pending, output, document_id, tokenizer, manifest
        )

    pieces = split_text_for_context(
        text,
        headings,
        tokenizer,
        MAX_TEXT_TOKENS,
    )

    for piece in pieces:
        if pending["parts"]:
            candidate_text = render_text(
                headings,
                pending["parts"] + [piece],
            )

            if count_tokens(
                candidate_text, tokenizer
            ) > MAX_TEXT_TOKENS:
                flush_text(
                    pending,
                    output,
                    document_id,
                    tokenizer,
                    manifest,
                )

        standalone_text = render_text(headings, [piece])

        if not pending["parts"] and count_tokens(
            standalone_text, tokenizer
        ) > MAX_TEXT_TOKENS:
            output.append({
                "document_id": document_id,
                "content_type": "text",
                "text": standalone_text,
                "page_numbers": pages,
                "headings": headings,
                "element_types": [label],
                "source_regions": regions,
                "page_images": get_page_assets(manifest, pages),
                "visual_assets": get_visual_assets(manifest, pages),
                "token_count": count_tokens(
                    standalone_text, tokenizer
                ),
            })
            continue

        if not pending["parts"]:
            pending["pages"] = set(pages)
            pending["headings"] = headings
            pending["element_types"] = set()
            pending["source_regions"] = []

        pending["parts"].append(piece)
        pending["element_types"].add(label)

        for region in regions:
            if region not in pending["source_regions"]:
                pending["source_regions"].append(region)


def flush_text(pending, output, document_id, tokenizer, manifest):
    """Finalize the pending text chunk, if it contains any text."""
    if not pending["parts"]:
        return

    record = make_text_record(
        document_id,
        pending,
        tokenizer,
        manifest,
    )

    if record:
        output.append(record)

    pending.update({
        "parts": [],
        "pages": set(),
        "headings": [],
        "element_types": set(),
        "source_regions": [],
    })


def clean_cell(value):
    """Normalize table cell text for Markdown serialization."""
    return re.sub(
        r"\s+", " ", str(value or "")
    ).replace("|", r"\|").strip()


def table_to_matrix(table):
    """Rebuild the table grid from Docling cell offsets and spans."""
    data = table.get("data", {})
    row_count = int(data.get("num_rows", 0))
    col_count = int(data.get("num_cols", 0))

    matrix = [
        ["" for _ in range(col_count)]
        for _ in range(row_count)
    ]

    cells = data.get("table_cells", [])

    for cell in cells:
        row = int(cell.get("start_row_offset_idx", 0))
        col = int(cell.get("start_col_offset_idx", 0))
        text = clean_cell(cell.get("text", ""))

        if (
            not text
            or row < 0
            or col < 0
            or row >= row_count
            or col >= col_count
        ):
            continue

        matrix[row][col] = text

        if not cell.get("column_header"):
            row_span = int(cell.get("row_span", 1))

            for covered_row in range(
                row + 1,
                min(row + row_span, row_count),
            ):
                if not matrix[covered_row][col]:
                    matrix[covered_row][col] = text

    header_cells = [
        cell for cell in cells
        if cell.get("column_header")
    ]

    if header_cells:
        header_end = max(
            int(cell.get("end_row_offset_idx", 0))
            for cell in header_cells
        )

        headers = []
        for col in range(col_count):
            labels = []

            for cell in sorted(
                header_cells,
                key=lambda c: (
                    int(c.get("start_row_offset_idx", 0)),
                    int(c.get("start_col_offset_idx", 0)),
                ),
            ):
                start_col = int(
                    cell.get("start_col_offset_idx", 0)
                )
                col_span = int(cell.get("col_span", 1))
                label = clean_cell(cell.get("text", ""))

                if (
                    label
                    and start_col <= col < start_col + col_span
                    and label not in labels
                ):
                    labels.append(label)

            headers.append(
                " — ".join(labels)
                if labels
                else (
                    "Metric" if col == 0
                    else f"Column {col + 1}"
                )
            )

        rows = [
            row[:]
            for row in matrix[header_end:]
            if any(cell.strip() for cell in row)
        ]

    else:
        headers = [
            "Metric" if col == 0 else f"Column {col + 1}"
            for col in range(col_count)
        ]
        rows = [
            row[:]
            for row in matrix
            if any(cell.strip() for cell in row)
        ]

    rows = [
        [clean_cell(cell) for cell in row]
        for row in rows
    ]

    return headers, rows


def render_markdown_table(headers, rows):
    """Render a rectangular Markdown table with its column headers."""
    normalized_rows = [
        (row + [""] * len(headers))[:len(headers)]
        for row in rows
    ]

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]

    lines.extend(
        "| " + " | ".join(row) + " |"
        for row in normalized_rows
    )

    return "\n".join(lines)


def caption_text(item, index):
    """Resolve caption references into readable text strings."""
    captions = []

    for ref in item.get("captions", []) or []:
        resolved = resolve_ref(ref, index)

        if resolved and resolved[0] == "texts":
            text = read_text(resolved[1])

            if text and text not in captions:
                captions.append(text)

    return " ".join(captions)


def table_chunks(
    document_id,
    table,
    table_number,
    headings,
    index,
    tokenizer,
    manifest,
):
    """Create intact table chunks, splitting only between rows when needed."""
    headers, rows = table_to_matrix(table)

    if not headers:
        return []

    caption = caption_text(table, index)
    prefix_parts = []

    if headings:
        prefix_parts.append(
            "Section: " + " > ".join(headings)
        )
    if caption:
        prefix_parts.append("Table caption: " + caption)

    prefix = "\n".join(prefix_parts)
    full_table = render_markdown_table(headers, rows)
    full_text = "\n\n".join(
        part for part in (prefix, full_table) if part
    )

    parts = []

    if count_tokens(full_text, tokenizer) <= MAX_TABLE_TOKENS:
        parts = [full_text]
    else:
        current_rows = []

        for row in rows:
            trial_rows = current_rows + [row]
            trial_text = "\n\n".join(
                part for part in (
                    prefix,
                    render_markdown_table(headers, trial_rows),
                )
                if part
            )

            if (
                current_rows
                and count_tokens(trial_text, tokenizer)
                > MAX_TABLE_TOKENS
            ):
                parts.append("\n\n".join(
                    part for part in (
                        prefix,
                        render_markdown_table(headers, current_rows),
                    )
                    if part
                ))
                current_rows = [row]
            else:
                current_rows.append(row)

        if current_rows:
            parts.append("\n\n".join(
                part for part in (
                    prefix,
                    render_markdown_table(headers, current_rows),
                )
                if part
            ))

    pages = pages_for(table)
    regions = source_regions(table, "table")
    result = []

    for part_number, text in enumerate(parts, start=1):
        result.append({
            "document_id": document_id,
            "content_type": "table",
            "table_number": table_number,
            "table_part": part_number,
            "text": text,
            "page_numbers": pages,
            "headings": headings,
            "caption": caption,
            "element_types": ["table"],
            "source_regions": regions,
            "page_images": get_page_assets(manifest, pages),
            "visual_assets": get_visual_assets(manifest, pages),
            "token_count": count_tokens(text, tokenizer),
        })

    return result


def image_record(
    document_id,
    picture,
    headings,
    index,
    manifest,
    picture_number,
):
    """Preserve an image and any text Docling associated with it."""
    pages = pages_for(picture)
    image = picture.get("image", {}) or {}
    uri = image.get("uri")
    caption = caption_text(picture, index)
    child_texts = []

    for ref in picture.get("children", []) or []:
        resolved = resolve_ref(ref, index)

        if resolved and resolved[0] == "texts":
            text = read_text(resolved[1])

            if text and text not in child_texts:
                child_texts.append(text)

    descriptions = []

    if caption:
        descriptions.append("Caption: " + caption)
    if child_texts:
        descriptions.append(
            "Recognized text: " + " ".join(child_texts)
        )
    if not descriptions:
        descriptions.append(
            "Image or figure extracted from the source PDF."
        )
    if headings:
        descriptions.insert(
            0,
            "Section: " + " > ".join(headings),
        )

    return {
        "document_id": document_id,
        "content_type": "image",
        "image_number": picture_number,
        "text": "\n".join(descriptions),
        "image_uri": uri,
        "caption": caption,
        "recognized_text": child_texts,
        "page_numbers": pages,
        "headings": headings,
        "element_types": ["picture"],
        "source_regions": source_regions(picture, "picture"),
        "page_images": get_page_assets(manifest, pages),
        "visual_assets": get_visual_assets(manifest, pages),
        "token_count": None,
    }


def chunk_document(
    document_id,
    on_progress: Callable[[str, float | None], None] | None = None,
):
    """Chunk text, tables and images separately while preserving reading order."""
    if on_progress is not None:
        on_progress("Loading document for type-aware chunking", 0.76)
    print(f"[Chunking] Loading document: {document_id}")

    doc_dir = DOCLING_OUTPUT_DIR / document_id
    document = load_json(doc_dir / "document.json")

    manifest_path = doc_dir / "manifest.json"
    manifest = (
        load_json(manifest_path)
        if manifest_path.exists()
        else {}
    )

    index = build_reference_index(document)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    tokenizer.model_max_length = 1_000_000

    records = []
    heading_stack = {}

    pending = {
        "parts": [],
        "pages": set(),
        "headings": [],
        "element_types": set(),
        "source_regions": [],
    }

    table_number = 0
    picture_number = 0

    def flush():
        """Flush pending text before a table, image or section boundary."""
        flush_text(
            pending,
            records,
            document_id,
            tokenizer,
            manifest,
        )

    for kind, item in ordered_elements(document, index):
        if kind == "text":
            text = read_text(item)

            if not text:
                continue

            label = str(item.get("label", "text")).lower()

            if label in ("section_header", "title"):
                flush()

                level = item.get("level", 1)
                try:
                    level = max(1, int(level))
                except (TypeError, ValueError):
                    level = 1

                heading_stack = {
                    key: value
                    for key, value in heading_stack.items()
                    if key < level
                }
                heading_stack[level] = text

            append_text_item(
                item,
                heading_stack,
                pending,
                records,
                document_id,
                tokenizer,
                manifest,
            )

        elif kind == "table":
            flush()
            table_number += 1

            records.extend(
                table_chunks(
                    document_id,
                    item,
                    table_number,
                    heading_context(heading_stack),
                    index,
                    tokenizer,
                    manifest,
                )
            )

        elif kind == "image":
            flush()
            picture_number += 1

            records.append(
                image_record(
                    document_id,
                    item,
                    heading_context(heading_stack),
                    index,
                    manifest,
                    picture_number,
                )
            )

    flush()

    output_dir = CHUNKS_DIR / f"{document_id}_typeaware"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "chunks.jsonl"

    document_name = str(
        manifest.get("source_file") or document_id
    )
    for number, record in enumerate(records, start=1):
        record["document_name"] = document_name
        record["chunk_id"] = (
            f"{document_id}_chunk_{number:05d}"
        )

    with output_file.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    default=str,
                ) + "\n"
            )

    page_count = len(document.get("pages", {}))
    pages_covered = sorted({
        page
        for record in records
        for page in record.get("page_numbers", [])
    })

    typed_counts = {
        kind: sum(
            record.get("content_type") == kind
            for record in records
        )
        for kind in ("text", "table", "image")
    }

    summary = {
        "document_id": document_id,
        "total_chunks": len(records),
        "text_chunks": typed_counts["text"],
        "table_chunks": typed_counts["table"],
        "image_chunks": typed_counts["image"],
        "source_tables": len(document.get("tables", [])),
        "source_pictures": len(document.get("pictures", [])),
        "pages_covered": len(pages_covered),
        "page_count": page_count,
        "pages_missing": sorted(
            set(range(1, page_count + 1)) - set(pages_covered)
        ),
        "chunks_missing_page_metadata": sum(
            not record.get("page_numbers")
            for record in records
        ),
        "chunks_over_token_target": sum(
            record.get("token_count", 0) > (
                MAX_TABLE_TOKENS
                if record.get("content_type") == "table"
                else MAX_TEXT_TOKENS
            )
            for record in records
            if isinstance(record.get("token_count"), int)
        ),
        "average_text_tokens": round(
            mean([
                record["token_count"]
                for record in records
                if record.get("content_type") == "text"
                and isinstance(record.get("token_count"), int)
            ]) if typed_counts["text"] else 0,
            2,
        ),
        "output_file": str(output_file),
    }

    summary_file = output_dir / "chunk_summary.json"
    summary_file.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if on_progress is not None:
        on_progress(
            f"Chunking complete: {len(records)} chunks created",
            0.84,
        )

    print("\n[Type-aware Chunking] Summary")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print(f"Summary: {summary_file}")
    return summary


def main():
    """Parse command-line arguments and run type-aware chunking."""
    parser = argparse.ArgumentParser(
        description="Create type-aware chunks from a Docling document."
    )
    parser.add_argument(
        "document_id",
        help="Document ID from ingestion",
    )
    args = parser.parse_args()

    chunk_document(args.document_id)


if __name__ == "__main__":
    main()