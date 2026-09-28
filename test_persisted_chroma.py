import os
import chromadb

print("========== PERSISTED CHROMA TEST ==========", flush=True)

path = os.path.abspath("vectorstore")

print("VECTORSTORE PATH:", path, flush=True)
print("VECTORSTORE EXISTS:", os.path.exists(path), flush=True)

if os.path.exists(path):
    print("FILES:", os.listdir(path), flush=True)

print("CREATING PERSISTENT CLIENT...", flush=True)

client = chromadb.PersistentClient(
    path=path
)

print("PERSISTENT CLIENT CREATED", flush=True)

collections = client.list_collections()

print("COLLECTIONS:", collections, flush=True)

collection = client.get_collection("langchain")

print("COLLECTION OPENED", flush=True)

print("STARTING RAW QUERY...", flush=True)

result = collection.query(
    query_texts=["tell me about leave policy"],
    n_results=3
)

print("RAW QUERY FINISHED", flush=True)

print("RESULT:", result, flush=True)

print("========== TEST COMPLETE ==========", flush=True)