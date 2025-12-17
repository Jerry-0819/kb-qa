from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.config import settings

import docx2txt
from pypdf import PdfReader

RAW_DIR = settings.data_raw
INDEX_DIR = settings.index_dir
INDEX_PATH = settings.index_path

def load_txt(path: Path):
    return path.read_text(encoding="utf-8")

def load_pdf(path: Path):
    reader = PdfReader(str(path))
    pages = [page.extract_text() for page in reader.pages]
    return "\n".join(pages)

def load_docx(path: Path):
    return docx2txt.process(str(path))

def load_documents():
    docs = []
    for file in RAW_DIR.iterdir():
        if file.suffix.lower() == ".pdf":
            text = load_pdf(file)
        elif file.suffix.lower() == ".docx":
            text = load_docx(file)
        elif file.suffix.lower() == ".txt":
            text = load_txt(file)
        else:
            continue

        docs.append({
            "content": text,
            "metadata": {"source": str(file)}
        })

    return docs


def main():
    print(f"Loading documents from: {RAW_DIR}")
    documents = load_documents()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    all_chunks = []
    for doc in documents:
        chunks = text_splitter.split_text(doc["content"])
        for chunk in chunks:
            all_chunks.append(
                {
                    "page_content": chunk,
                    "metadata": doc["metadata"]
                }
            )

    print(f"Total chunks: {len(all_chunks)}")

    embeddings = OpenAIEmbeddings(model=settings.embeddings_model)

    print("Building FAISS vector store...")
    vectorstore = FAISS.from_texts(
        [c["page_content"] for c in all_chunks],
        embedding=embeddings,
        metadatas=[c["metadata"] for c in all_chunks]
    )

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Saving FAISS index → {INDEX_PATH}")
    vectorstore.save_local(str(INDEX_DIR))

    print("Done. You can now run the API!")


if __name__ == "__main__":
    main()
