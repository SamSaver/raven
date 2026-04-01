import tempfile
from pathlib import Path

import structlog

logger = structlog.get_logger()


class ParsedDocument:
    def __init__(
        self,
        content: str,
        pages: list[str] | None = None,
        tables: list[dict] | None = None,
        metadata: dict | None = None,
    ):
        self.content = content
        self.pages = pages or []
        self.tables = tables or []
        self.metadata = metadata or {}

    @property
    def page_count(self) -> int:
        return len(self.pages) if self.pages else 1


def detect_file_type(file_path: Path) -> str:
    import magic

    mime = magic.from_file(str(file_path), mime=True)
    return mime


def parse_with_docling(file_path: Path) -> ParsedDocument:
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(str(file_path))
    md_content = result.document.export_to_markdown()

    # Extract page-level content if available
    pages = []
    if hasattr(result.document, "pages"):
        for page in result.document.pages:
            pages.append(str(page))

    return ParsedDocument(
        content=md_content,
        pages=pages if pages else [md_content],
        metadata={
            "parser": "docling",
            "source": file_path.name,
        },
    )


def parse_with_pymupdf(file_path: Path) -> ParsedDocument:
    import fitz

    doc = fitz.open(str(file_path))
    pages = []
    full_text = []

    for page in doc:
        text = page.get_text("text")
        pages.append(text)
        full_text.append(text)

    doc.close()

    return ParsedDocument(
        content="\n\n".join(full_text),
        pages=pages,
        metadata={
            "parser": "pymupdf",
            "source": file_path.name,
        },
    )


def extract_tables(file_path: Path) -> list[dict]:
    import pdfplumber

    tables = []
    with pdfplumber.open(str(file_path)) as pdf:
        for i, page in enumerate(pdf.pages):
            page_tables = page.extract_tables()
            for j, table in enumerate(page_tables):
                if table and len(table) > 1:
                    headers = table[0]
                    rows = table[1:]
                    tables.append({
                        "page": i + 1,
                        "table_index": j,
                        "headers": headers,
                        "rows": rows,
                        "markdown": _table_to_markdown(headers, rows),
                    })
    return tables


def _table_to_markdown(headers: list, rows: list[list]) -> str:
    clean_headers = [str(h or "").strip() for h in headers]
    lines = [
        "| " + " | ".join(clean_headers) + " |",
        "| " + " | ".join(["---"] * len(clean_headers)) + " |",
    ]
    for row in rows:
        clean_row = [str(cell or "").strip() for cell in row]
        # Pad or truncate to match header count
        while len(clean_row) < len(clean_headers):
            clean_row.append("")
        lines.append("| " + " | ".join(clean_row[: len(clean_headers)]) + " |")
    return "\n".join(lines)


def parse_text_file(file_path: Path) -> ParsedDocument:
    content = file_path.read_text(encoding="utf-8", errors="replace")
    return ParsedDocument(
        content=content,
        pages=[content],
        metadata={"parser": "text", "source": file_path.name},
    )


def parse_document(file_path: Path, prefer_docling: bool = True) -> ParsedDocument:
    """Unified document parser. Tries docling first, falls back to PyMuPDF for PDFs."""
    file_type = detect_file_type(file_path)
    logger.info("parser.detect", file=file_path.name, type=file_type)

    if file_type == "application/pdf":
        tables = extract_tables(file_path)

        if prefer_docling:
            try:
                doc = parse_with_docling(file_path)
                doc.tables = tables
                doc.metadata["file_type"] = file_type
                return doc
            except Exception as e:
                logger.warning("parser.docling_failed", error=str(e), fallback="pymupdf")

        doc = parse_with_pymupdf(file_path)
        doc.tables = tables
        doc.metadata["file_type"] = file_type
        return doc

    elif file_type in ("text/plain", "text/markdown", "text/csv"):
        doc = parse_text_file(file_path)
        doc.metadata["file_type"] = file_type
        return doc

    else:
        # Attempt text extraction as fallback
        try:
            doc = parse_text_file(file_path)
            doc.metadata["file_type"] = file_type
            return doc
        except Exception:
            raise ValueError(f"Unsupported file type: {file_type}")


async def save_upload(file_content: bytes, filename: str, upload_dir: Path) -> Path:
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / filename
    file_path.write_bytes(file_content)
    return file_path
