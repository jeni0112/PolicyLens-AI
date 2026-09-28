from langchain_chroma import Chroma
from config import embeddings

print("========== LANGCHAIN CHROMA TEST ==========", flush=True)

print("1. Creating Chroma object...", flush=True)

db = Chroma(
    persist_directory="vectorstore",
    embedding_function=embeddings
)

print("2. Chroma object created", flush=True)

print("3. Starting similarity search...", flush=True)

results = db.similarity_search(
    "tell me about leave policy",
    k=3
)

print("4. Similarity search completed", flush=True)

print("RESULT COUNT:", len(results), flush=True)

for i, doc in enumerate(results):
    print("\nRESULT", i + 1, flush=True)
    print("CONTENT:", doc.page_content[:300], flush=True)
    print("METADATA:", doc.metadata, flush=True)

print("========== TEST COMPLETE ==========", flush=True)