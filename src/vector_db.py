"""Vector database management module"""
import json
import os
from datetime import datetime, timezone
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .document_loader import DocumentLoaderFactory, PDFLoader, ImageLoader
# text_cleaner disabled -- was stripping useful content during ingestion
# from .text_cleaner import clean_text


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

                doc_metadata = doc.get('metadata', {})
                metadata = {
                    'section': doc.get('section', ''),
                    'source': doc_metadata.get('source', ''),
                    'chunk_index': j,
                    'total_chunks': len(chunks),
                    'parent_doc_id': f"doc_{i}",
                    'structured': doc_metadata.get('structured', False),
                    'url': doc_metadata.get('url', ''),
                    'link': doc_metadata.get('link', ''),
                    'source_url': doc_metadata.get('source_url', ''),
                    'ingested_at': datetime.now(tz=timezone.utc).isoformat(),
                }
                metadatas.append(metadata)

        print(f"🚀 Adding {len(documents)} chunks to ChromaDB...")

        try:
            self.collection.add(ids=ids, documents=documents, metadatas=metadatas)
            print(f"✅ Successfully added {len(documents)} chunks")
        except Exception as e:
            print(f"❌ Error adding documents: {e}")
            raise

    def add_documents_from_pdf(
        self,
        pdf_path: str,
        metadata: Optional[Dict] = None,
        min_text_length: int = 100,
        ocr_language: str = "eng+chi_sim+chi_tra",
        tesseract_path: Optional[str] = None
    ) -> int:
        """
        Load and add documents from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            metadata: Optional additional metadata to include
            min_text_length: Minimum text length before triggering OCR
            ocr_language: Language for OCR
            tesseract_path: Path to Tesseract executable
            
        Returns:
            Number of chunks added
        """
        print(f"📄 Processing PDF: {pdf_path}")
        
        loader = PDFLoader(
            min_text_length=min_text_length,
            ocr_language=ocr_language,
            tesseract_path=tesseract_path
        )
        
        documents = loader.load(pdf_path)
        return self._add_loaded_documents(documents, metadata)

    def add_documents_from_image(
        self,
        image_path: str,
        metadata: Optional[Dict] = None,
        ocr_language: str = "eng+chi_sim+chi_tra",
        tesseract_path: Optional[str] = None
    ) -> int:
        """
        Load and add documents from an image file using OCR.
        
        Args:
            image_path: Path to the image file
            metadata: Optional additional metadata to include
            ocr_language: Language for OCR
            tesseract_path: Path to Tesseract executable
            
        Returns:
            Number of chunks added
        """
        print(f"🖼️ Processing image: {image_path}")
        
        loader = ImageLoader(
            ocr_language=ocr_language,
            tesseract_path=tesseract_path
        )
        
        documents = loader.load(image_path)
        return self._add_loaded_documents(documents, metadata)

    def add_documents_from_directory(
        self,
        directory: str,
        extensions: Optional[List[str]] = None,
        recursive: bool = False,
        metadata: Optional[Dict] = None,
        min_text_length: int = 100,
        ocr_language: str = "eng+chi_sim+chi_tra",
        tesseract_path: Optional[str] = None
    ) -> int:
        """
        Load and add all supported documents from a directory.
        
        Args:
            directory: Path to the directory
            extensions: Optional list of extensions to filter (e.g., ['.pdf', '.png'])
            recursive: Whether to search subdirectories
            metadata: Optional additional metadata to include
            min_text_length: Minimum text length before triggering OCR for PDFs
            ocr_language: Language for OCR
            tesseract_path: Path to Tesseract executable
            
        Returns:
            Number of chunks added
        """
        print(f"📁 Processing directory: {directory}")
        
        factory = DocumentLoaderFactory(
            min_text_length=min_text_length,
            ocr_language=ocr_language,
            tesseract_path=tesseract_path
        )
        
        documents = factory.load_directory(
            directory=directory,
            extensions=extensions,
            recursive=recursive
        )
        
        return self._add_loaded_documents(documents, metadata)

    def _add_loaded_documents(
        self,
        documents: List[Dict],
        extra_metadata: Optional[Dict] = None
    ) -> int:
        """
        Internal method to add loaded documents to the database.
        
        Args:
            documents: List of documents with 'content' and 'metadata' keys
            extra_metadata: Optional additional metadata to merge
            
        Returns:
            Number of chunks added
        """
        if not documents:
            print("⚠️ No documents to add")
            return 0
        
        ids, texts, metadatas = [], [], []
        
        for i, doc in enumerate(documents):
            content = doc.get('content', '').strip()
            if not content:
                continue
            
            chunks = self.text_splitter.split_text(content)
            doc_metadata = doc.get('metadata', {})
            
            for j, chunk in enumerate(chunks):
                # Generate unique chunk ID
                chunk_id = f"{doc_metadata.get('type', 'doc')}_{doc_metadata.get('parent_doc_id', i)}_chunk_{j}_{hash(chunk[:50]) % 1000000}"
                ids.append(chunk_id)
                texts.append(chunk)
                
                chunk_metadata = {
                    'source': doc_metadata.get('source', ''),
                    'type': doc_metadata.get('type', 'document'),
                    'extraction_method': doc_metadata.get('extraction_method', 'unknown'),
                    'chunk_index': j,
                    'total_chunks': len(chunks),
                    'parent_doc_id': doc_metadata.get('parent_doc_id', f'doc_{i}'),
                    'file_modified_at': doc_metadata.get('file_modified_at', ''),
                    'ingested_at': doc_metadata.get('ingested_at',
                                                     datetime.now(tz=timezone.utc).isoformat()),
                }

                if doc_metadata.get('url'):
                    chunk_metadata['url'] = doc_metadata['url']
                if doc_metadata.get('section'):
                    chunk_metadata['section'] = doc_metadata['section']
                
                # Add page info for PDFs
                if 'page' in doc_metadata:
                    chunk_metadata['page'] = doc_metadata['page']
                    chunk_metadata['total_pages'] = doc_metadata.get('total_pages', 1)
                
                # Add image info
                if 'image_size' in doc_metadata:
                    chunk_metadata['image_size'] = doc_metadata['image_size']
                if 'format' in doc_metadata:
                    chunk_metadata['format'] = doc_metadata['format']
                
                # Merge extra metadata if provided
                if extra_metadata:
                    chunk_metadata.update(extra_metadata)
                
                metadatas.append(chunk_metadata)
        
        if not texts:
            print("⚠️ No text content to add after processing")
            return 0
        
        print(f"🚀 Adding {len(texts)} chunks to ChromaDB...")
        
        try:
            self.collection.add(ids=ids, documents=texts, metadatas=metadatas)
            print(f"✅ Successfully added {len(texts)} chunks")
            return len(texts)
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

