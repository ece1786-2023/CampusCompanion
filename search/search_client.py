import chromadb
from chromadb.config import Settings

def searchRAG(query):
    chroma_client = chromadb.HttpClient(host="localhost", port = 8000, settings=Settings(allow_reset=True, anonymized_telemetry=False))
    collection_status = False
    while collection_status != True:
        try:
            document_collection = chroma_client.get_or_create_collection(name="sample_collection")
            collection_status = True
        except Exception as e:
            pass

    results = document_collection.query(query_texts=query, n_results=20)
    result_documents = results["documents"][0]
    print("RAG Search Results:")
    print(result_documents)
    return result_documents