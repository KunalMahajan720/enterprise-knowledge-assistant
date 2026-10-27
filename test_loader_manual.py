from src.document_loader import load_documents

documents = load_documents("documents")

print(f"Loaded {len(documents)} documents")

for document in documents:
    print(f"{document['source']}: {len(document['text'])} characters")