
from dataclasses import dataclass

from app.rag.document_loader import Document


@dataclass
class DocumentChunk:
    content: str
    source: str
    chunk_id: str


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 500,
    overlap: int = 100
) -> list[DocumentChunk]:

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size"
        )

    chunks = []
    step = chunk_size - overlap

    for document in documents:

        text = document.content

        for index, start in enumerate(
            range(0, len(text), step)
        ):

            end = min(start + chunk_size, len(text))

            chunk_text = text[start:end].strip()

            if not chunk_text:
                continue

            chunks.append(
                DocumentChunk(
                    content=chunk_text,
                    source=document.source,
                    chunk_id=f"{document.source}:{index}"
                )
            )

            if end == len(text):
                break

    return chunks
