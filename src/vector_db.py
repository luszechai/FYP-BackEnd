"""Vector database management module"""
import json
from typing import List, Dict
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter


class ChromaDBManager:
    """Manages ChromaDB vector database operations"""

    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "sfu_admission"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1600,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="BAAI/bge-large-en-v1.5"
        )

        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        try:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=embedding_function
            )
        except (ValueError, Exception) as e:
            print(f"🗑️ Recreating collection due to error: {str(e)[:100]}")
            try:
                self.client.delete_collection(name=collection_name)
                self.collection = self.client.create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"},
                    embedding_function=embedding_function
                )
            except Exception as delete_error:
                raise Exception(f"Failed to recreate collection: {delete_error}")

        print(f"ChromaDB initialized: {self.collection.count()} documents")

    def add_documents_from_json(self, json_file: str):
        """Load and chunk documents from JSON using LangChain splitter"""
        print(f"📄 Reading JSON file: {json_file}")
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        documents_list = data.get('documents', [])
        print(f"📋 Found {len(documents_list)} documents in JSON")

        ids, documents, metadatas = [], [], []

        for i, doc in enumerate(documents_list):
            if i % 10 == 0:
                print(f"⏳ Processing document {i + 1}/{len(documents_list)}")

            content = doc.get('content', '').strip()
            if not content:
                continue

            chunks = self.text_splitter.split_text(content)
            print(f"  🧩 Document {i + 1}: {len(content)} chars → {len(chunks)} chunks")

            for j, chunk in enumerate(chunks):
                chunk_id = f"doc_{i}_chunk_{j}_{hash(chunk[:50]) % 1000000}"
                ids.append(chunk_id)
                documents.append(chunk)

                metadata = {
                    'section': doc.get('section', ''),
                    'source': doc.get('metadata', {}).get('source', ''),
                    'chunk_index': j,
                    'total_chunks': len(chunks),
                    'parent_doc_id': f"doc_{i}",
                    'structured': doc.get('metadata', {}).get('structured', False)
                }
                metadatas.append(metadata)

        print(f"🚀 Adding {len(documents)} chunks to ChromaDB...")

        try:
            self.collection.add(ids=ids, documents=documents, metadatas=metadatas)
            print(f"✅ Successfully added {len(documents)} chunks")
        except Exception as e:
            print(f"❌ Error adding documents: {e}")
            raise

    def query(self, query_text: str, n_results: int = 5):
        """Query the vector database"""
        return self.collection.query(query_texts=[query_text], n_results=n_results)

    def format_results(self, results: Dict) -> List[Dict]:
        """Format query results into structured format"""
        formatted = []
        if results.get('ids'):
            ids = results['ids'][0]
            documents = results.get('documents', [[]])[0]
            metadatas = results.get('metadatas', [[]])[0]
            distances = results.get('distances', [[]])[0]

            for i, doc_id in enumerate(ids):
                result = {
                    'rank': i + 1,
                    'id': doc_id,
                    'document': documents[i] if i < len(documents) else '',
                    'metadata': metadatas[i] if i < len(metadatas) else {},
                    'similarity': 1 - distances[i] if i < len(distances) else None
                }
                formatted.append(result)

        return formatted

