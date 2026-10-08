from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer


# -----------------------------
# Configuration
# -----------------------------

RECIPES_DIR = Path("recipes")
DB_DIR = "chroma_db"
COLLECTION_NAME = "recipes"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


# -----------------------------
# Load embedding model
# -----------------------------

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# -----------------------------
# Connect to ChromaDB
# -----------------------------

client = chromadb.PersistentClient(path=DB_DIR)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)


# -----------------------------
# Read documents
# -----------------------------

documents = []
metadatas = []
ids = []


def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


print(f"Reading recipes from: {RECIPES_DIR}")


for file_path in RECIPES_DIR.iterdir():

    if file_path.suffix.lower() not in [".txt", ".md"]:
        continue

    print(f"Processing: {file_path.name}")

    text = file_path.read_text(
        encoding="utf-8"
    )

    chunks = split_text(text)

    for i, chunk in enumerate(chunks):

        documents.append(chunk)

        metadatas.append({
            "source": file_path.name,
            "chunk": i
        })

        ids.append(
            f"{file_path.stem}_{i}"
        )


# -----------------------------
# Create embeddings
# -----------------------------

print(f"Creating embeddings for {len(documents)} chunks...")

embeddings = embedding_model.encode(
    documents
).tolist()


# -----------------------------
# Store in ChromaDB
# -----------------------------

collection.upsert(
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas,
    ids=ids
)


print()
print("Ingestion complete.")
print(f"Documents/chunks stored: {len(documents)}")