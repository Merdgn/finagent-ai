
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Document:
    content: str
    source: str


def load_documents(directory: str) -> list[Document]:
    """
    Load Markdown and text documents from a directory.
    """

    folder = Path(directory)

    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(
            f"Document directory not found: {directory}"
        )

    documents = []

    for file_path in sorted(folder.iterdir()):

        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in {".md", ".txt"}:
            continue

        content = file_path.read_text(
            encoding="utf-8"
        ).strip()

        if not content:
            continue

        documents.append(
            Document(
                content=content,
                source=file_path.name
            )
        )

    return documents
