from fastapi import FastAPI
import chromadb

app = FastAPI()

chroma_client = chromadb.PersistentClient(path="./chroma_db")

collection = chroma_client.get_or_create_collection("uniper_collection")


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


@app.post("/add_document")
async def add_document(text: str, doc_id: str = None):
    if doc_id is None:
        doc_id = f"doc_{len(collection.get()['ids']) + 1}"
    
    collection.add(
        documents=[text],
        ids=[doc_id]
    )
    
    return {"message": "Document added", "id": doc_id}


@app.get("/search")
async def search_documents(query: str, n_results: int = 5):
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    
    return {
        "query": query,
        "results": results
    }
