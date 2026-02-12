"""
Generate a synthetic evaluation testset from ChromaDB documents using Ragas.

This is a one-time script that:
1. Connects to ChromaDB and extracts all stored chunks
2. Reconstructs full documents by grouping chunks by parent_doc_id
3. Builds a Ragas KnowledgeGraph from the reconstructed documents
4. Generates ~100 synthetic Q&A pairs for evaluation
5. Saves the testset to eval_testset.json and the knowledge graph for reuse

Usage:
    python generate_testset.py
    python generate_testset.py --testset-size 50 --output my_testset.json
"""
import argparse
import json
import os
import sys
from collections import defaultdict

from dotenv import load_dotenv
from langchain_core.documents import Document as LangchainDocument
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph, NodeType
from ragas.testset.transforms import (
    Parallel,
    SummaryExtractor,
    EmbeddingExtractor,
    CustomNodeFilter,
)
from ragas.testset.transforms.extractors.llm_based import NERExtractor, ThemesExtractor
from ragas.testset.transforms.relationship_builders import (
    CosineSimilarityBuilder,
    OverlapScoreBuilder,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from config import Config
from src.vector_db import ChromaDBManager

load_dotenv()


def extract_all_chunks(db: ChromaDBManager) -> dict:
    """Extract all chunks from ChromaDB with their documents and metadata."""
    print("📦 Extracting all chunks from ChromaDB...")
    total = db.collection.count()
    print(f"   Total chunks in collection: {total}")

    if total == 0:
        print("❌ No documents found in ChromaDB. Ingest documents first.")
        sys.exit(1)

    # ChromaDB get() has a default limit; fetch in batches if needed
    batch_size = 5000
    all_ids, all_documents, all_metadatas = [], [], []

    for offset in range(0, total, batch_size):
        results = db.collection.get(
            include=["documents", "metadatas"],
            limit=batch_size,
            offset=offset,
        )
        all_ids.extend(results["ids"])
        all_documents.extend(results["documents"])
        all_metadatas.extend(results["metadatas"])

    print(f"   Extracted {len(all_ids)} chunks")
    return {
        "ids": all_ids,
        "documents": all_documents,
        "metadatas": all_metadatas,
    }


def reconstruct_documents(chunks: dict) -> list[LangchainDocument]:
    """
    Reconstruct full documents by grouping chunks by parent_doc_id,
    sorting by chunk_index, and joining text.
    Returns a list of LangChain Document objects.
    """
    print("🔄 Reconstructing full documents from chunks...")
    groups: dict[str, list] = defaultdict(list)

    for i, meta in enumerate(chunks["metadatas"]):
        parent_id = meta.get("parent_doc_id", f"unknown_{i}")
        chunk_index = meta.get("chunk_index", 0)
        groups[parent_id].append(
            {
                "text": chunks["documents"][i],
                "chunk_index": chunk_index,
                "metadata": meta,
            }
        )

    documents = []
    for parent_id, chunk_list in groups.items():
        # Sort by chunk_index
        chunk_list.sort(key=lambda c: c["chunk_index"])
        full_text = "\n\n".join(c["text"] for c in chunk_list)

        # Use metadata from the first chunk as representative
        first_meta = chunk_list[0]["metadata"]
        metadata = {
            "parent_doc_id": parent_id,
            "section": first_meta.get("section", ""),
            "source": first_meta.get("source", ""),
            "total_chunks": len(chunk_list),
        }

        doc = LangchainDocument(page_content=full_text, metadata=metadata)
        documents.append(doc)

    print(f"   Reconstructed {len(documents)} full documents")
    return documents


def build_custom_transforms(llm_wrapper, embeddings_wrapper):
    """
    Build a custom Ragas transform pipeline that works reliably with DeepSeek.

    This skips the HeadlinesExtractor + HeadlineSplitter steps from the default
    pipeline (which fail when DeepSeek doesn't return the expected headline format).
    Instead it uses summaries, NER, themes, embeddings, and cosine similarity --
    all the enrichment Ragas needs for high-quality testset generation.
    """
    def filter_docs(node):
        return node.type == NodeType.DOCUMENT

    summary_extractor = SummaryExtractor(
        llm=llm_wrapper,
        filter_nodes=filter_docs,
    )
    summary_emb_extractor = EmbeddingExtractor(
        embedding_model=embeddings_wrapper,
        property_name="summary_embedding",
        embed_property_name="summary",
        filter_nodes=filter_docs,
    )
    cosine_sim_builder = CosineSimilarityBuilder(
        property_name="summary_embedding",
        new_property_name="summary_similarity",
        threshold=0.5,
        filter_nodes=filter_docs,
    )
    ner_extractor = NERExtractor(llm=llm_wrapper)
    ner_overlap_sim = OverlapScoreBuilder(threshold=0.01)
    theme_extractor = ThemesExtractor(
        llm=llm_wrapper,
        filter_nodes=filter_docs,
    )
    node_filter = CustomNodeFilter(llm=llm_wrapper)

    transforms = [
        summary_extractor,
        node_filter,
        Parallel(summary_emb_extractor, theme_extractor, ner_extractor),
        Parallel(cosine_sim_builder, ner_overlap_sim),
    ]
    return transforms


def load_existing_knowledge_graph(
    kg_path: str,
) -> KnowledgeGraph | None:
    """Attempt to load a previously saved KnowledgeGraph. Returns None if not found."""
    if not os.path.exists(kg_path):
        return None
    try:
        print(f"   Found existing knowledge graph at {kg_path}")
        kg = KnowledgeGraph.load(kg_path)
        print("   ✅ Loaded existing knowledge graph")
        return kg
    except Exception as e:
        print(f"   ⚠️ Could not load existing graph: {e}. Will rebuild from scratch.")
        return None


def generate_testset(
    documents: list[LangchainDocument],
    llm_wrapper,
    embeddings_wrapper,
    testset_size: int = 100,
    kg_path: str = "eval_knowledge_graph.json",
) -> tuple[list[dict], KnowledgeGraph]:
    """
    Generate a synthetic testset using Ragas TestsetGenerator.

    If a saved KnowledgeGraph exists at kg_path it is reused (skipping the
    expensive transform/enrichment step).  Otherwise the KG is built from
    the documents automatically by generate_with_langchain_docs().

    Returns:
        (records, knowledge_graph) -- the testset rows and the KG that was used.
    """
    # Try to reuse a previously saved knowledge graph
    existing_kg = load_existing_knowledge_graph(kg_path)

    if existing_kg is not None:
        # KG already built -- use .generate() directly (no need to re-process docs)
        print(f"📝 Generating {testset_size} test questions (reusing existing KG)...")
        generator = TestsetGenerator(
            llm=llm_wrapper,
            embedding_model=embeddings_wrapper,
            knowledge_graph=existing_kg,
        )
        testset = generator.generate(testset_size=testset_size)
    else:
        # Build KG from scratch via generate_with_langchain_docs
        print(f"📝 Generating {testset_size} test questions (building KG from documents – this may take a while)...")
        generator = TestsetGenerator(
            llm=llm_wrapper,
            embedding_model=embeddings_wrapper,
        )
        # Use custom transforms that skip headline extraction (which fails with DeepSeek)
        custom_transforms = build_custom_transforms(llm_wrapper, embeddings_wrapper)
        testset = generator.generate_with_langchain_docs(
            documents=documents,
            testset_size=testset_size,
            transforms=custom_transforms,
        )
        # Save the newly built KG for future reuse
        generator.knowledge_graph.save(kg_path)
        print(f"   💾 Knowledge graph saved to {kg_path}")

    # Convert to list of dicts
    df = testset.to_pandas()
    records = df.to_dict(orient="records")

    print(f"   ✅ Generated {len(records)} test questions")
    return records, generator.knowledge_graph


def save_testset(records: list[dict], output_path: str):
    """Save the testset to a JSON file."""
    # Ensure all values are JSON-serializable
    serializable = []
    for record in records:
        item = {}
        for k, v in record.items():
            if hasattr(v, "tolist"):
                v = v.tolist()
            item[k] = v
        serializable.append(item)

    output = {
        "metadata": {
            "total_questions": len(serializable),
            "generator": "ragas",
        },
        "testset": serializable,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    print(f"💾 Testset saved to {output_path} ({len(serializable)} questions)")


def main():
    parser = argparse.ArgumentParser(description="Generate Ragas evaluation testset from ChromaDB documents")
    parser.add_argument("--testset-size", type=int, default=100, help="Number of test questions to generate (default: 100)")
    parser.add_argument("--output", type=str, default="eval_testset.json", help="Output testset file path (default: eval_testset.json)")
    parser.add_argument("--kg-output", type=str, default="eval_knowledge_graph.json", help="Knowledge graph output path (default: eval_knowledge_graph.json)")
    args = parser.parse_args()

    # Validate config
    Config.validate()

    # 1. Connect to ChromaDB
    print("🔌 Connecting to ChromaDB...")
    db = ChromaDBManager(
        persist_directory=Config.CHROMA_DB_DIR,
        collection_name=Config.CHROMA_COLLECTION_NAME,
    )

    # 2. Extract all chunks
    chunks = extract_all_chunks(db)

    # 3. Reconstruct full documents
    documents = reconstruct_documents(chunks)

    if not documents:
        print("❌ No documents reconstructed. Cannot generate testset.")
        sys.exit(1)

    # 4. Initialize LLM (DeepSeek via OpenAI-compatible API)
    print("🤖 Initializing DeepSeek LLM...")
    llm = ChatOpenAI(
        model=Config.DEEPSEEK_MODEL,
        base_url=Config.DEEPSEEK_BASE_URL,
        api_key=Config.DEEPSEEK_API_KEY,
        temperature=0.3,
        max_tokens=4096,
    )
    llm_wrapper = LangchainLLMWrapper(llm)

    # 5. Initialize embeddings
    print("📐 Initializing embeddings (BAAI/bge-large-en-v1.5)...")
    embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
    embeddings_wrapper = LangchainEmbeddingsWrapper(embeddings)

    # 6. Generate testset (builds or reuses knowledge graph automatically)
    records, kg = generate_testset(
        documents, llm_wrapper, embeddings_wrapper,
        testset_size=args.testset_size,
        kg_path=args.kg_output,
    )

    # 7. Save testset
    save_testset(records, args.output)

    print("\n✅ Testset generation complete!")
    print(f"   Questions: {len(records)}")
    print(f"   Output: {args.output}")
    print(f"   Knowledge graph: {args.kg_output}")


if __name__ == "__main__":
    main()
