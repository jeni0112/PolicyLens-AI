from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import embeddings

print("========== TINY CHROMA TEST ==========", flush=True)

# 2 very small test documents
documents = [
    Document(
        page_content="XYZTech leave policy provides 18 days of earned leave per year."
    ),
    Document(
        page_content="XYZTech sick leave policy provides 12 days of sick and wellness leave per year."
    )
]

print("Creating tiny Chroma database...", flush=True)

db = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="tiny_vectorstore"
)

print("TINY CHROMA CREATED", flush=True)

print("Starting similarity search...", flush=True)

results = db.similarity_search(
    "tell me about leave policy",
    k=1
)

print("SIMILARITY SEARCH COMPLETE", flush=True)
print("RESULT COUNT:", len(results), flush=True)

for i, doc in enumerate(results):
    print(f"RESULT {i + 1}:", doc.page_content, flush=True)

print("========== TINY CHROMA TEST END ==========", flush=True)