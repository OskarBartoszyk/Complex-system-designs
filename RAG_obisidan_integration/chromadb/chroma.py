import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

def create_collection(collection_name):
    collection = client.get_or_create_collection(name=collection_name)
    return collection 

def get_collection(collection_name):
    collection = client.get_collection(name=collection_name)
    return collection

create_collection("vault_notes")
get_collection("vault_notes")
