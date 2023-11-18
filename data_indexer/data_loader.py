import chromadb
from chromadb.config import Settings

def split_course_data():
    file_path = 'CourseData.txt' 
    with open(file_path, 'r', encoding='utf-8') as file:
        chunks = []
        current_chunk = []

        for line in file:
            if line.strip():  # Check if the line is not empty
                current_chunk.append(line)
            else:
                if current_chunk:
                    chunks.append(''.join(current_chunk))
                    current_chunk = []

        if current_chunk:
            chunks.append(''.join(current_chunk))

    return chunks

def load_database():
    chroma_client = chromadb.HttpClient(host="localhost", port = 8000, settings=Settings(allow_reset=True, anonymized_telemetry=False))
    documents = []
    metadatas = []
    ids = []

    result_chunks = split_course_data()
    for i, chunk in enumerate(result_chunks, start=1):
        metadata = {'source': chunk.split('\n')[0].strip()}
        course_description = chunk.replace('\n', '.')

        ids.append(str(i+1))
        documents.append(course_description)
        metadatas.append(metadata)

    collection_status = False
    while collection_status != True:
        try:
            document_collection = chroma_client.get_or_create_collection(name="sample_collection")
            collection_status = True
        except Exception as e:
            pass

    document_collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print('Successfully Index Course Data')
