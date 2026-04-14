# ingest_pipeline/parsers/pdf_parser.py
"""
Parser for .pdf files.

Strategy:
  - Use LangChain's PyPDFLoader (backed by pypdf) to extract text page-by-page.
  - Each page becomes one Document; the chunker will split further if needed.
  - Metadata includes the page number for traceability.
"""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def parse_pdf(path: Path) -> list[Document]:
    """
    Parse a PDF into one Document per page.

    PyPDFLoader already assigns `source` and `page` metadata; we augment that
    with a normalised set of fields consistent with the rest of the pipeline.
    """
    loader = PyPDFLoader(str(path))
    pages = loader.load()

    docs: list[Document] = []
    for page_doc in pages:
        text = page_doc.page_content.strip()
        if not text:
            continue

        docs.append(Document(
            page_content=text,
            metadata={
                "source_file":  str(path),
                "format":       "pdf",
                "content_type": "narrative",
                "title":        path.stem,
                "page":         page_doc.metadata.get("page", 0),
            },
        ))

    return docs
