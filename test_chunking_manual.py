from src.document_loader import load_documents
from src.text_chunker import chunk_documents

documents = load_documents("documents")

chunks = chunk_documents(
    documents,
    chunk_size=300,
    overlap=50,
)

print(f"Documents loaded: {len(documents)}")
print(f"Chunks created: {len(chunks)}")

for chunk in chunks:
    print(
        f"\nSource: {chunk['source']}"
        f"\nChunk index: {chunk['chunk_index']}"
        f"\nCharacters: {len(chunk['text'])}"
        f"\nText preview: {chunk['text'][:100]}"
    )