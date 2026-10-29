"""Inspect Docling's original table extraction for PDF page 4."""

from pathlib import Path

from docling_core.types.doc import DoclingDocument

DOCUMENT_FILE = Path(
    "data/processed/docling/"
    "doc_a7a0d8261a0923404fc45446/"
    "document.json"
)
TARGET_PAGE = 4


def get_table_pages(table):
    """Return the PDF page numbers associated with a table."""
    return sorted({
        provenance.page_no
        for provenance in (table.prov or [])
        if getattr(provenance, "page_no", None) is not None
    })


def main():
    """Display the original table structure extracted from page 4."""
    if not DOCUMENT_FILE.exists():
        raise FileNotFoundError(
            f"Docling document not found: {DOCUMENT_FILE}"
        )

    document = DoclingDocument.model_validate_json(
        DOCUMENT_FILE.read_text(encoding="utf-8")
    )

    print(f"Total detected tables: {len(document.tables)}")

    matching_tables = [
        (index, table, get_table_pages(table))
        for index, table in enumerate(document.tables, start=1)
        if TARGET_PAGE in get_table_pages(table)
    ]

    print(f"Tables found on PDF page {TARGET_PAGE}: {len(matching_tables)}")

    for index, table, pages in matching_tables:
        print(f"\n{'=' * 60}")
        print(f"Table index: {index}")
        print(f"Pages: {pages}")

        try:
            dataframe = table.export_to_dataframe(doc=document)

            print(f"Rows: {dataframe.shape[0]}")
            print(f"Columns: {dataframe.shape[1]}")
            print("\nExtracted table:")
            print(dataframe.to_string(index=False, max_rows=100))
        except Exception as error:
            print(f"Table export failed: {type(error).__name__}: {error}")


if __name__ == "__main__":
    main()