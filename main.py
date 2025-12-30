"""Main entry point for SFU Admission Chatbot"""
import os
import pandas as pd
from src.chatbot import RAGChatbot
from src.llm_provider import LLMProvider
from src.vector_db import ChromaDBManager
from src.evaluation import generate_evaluation_dashboard
from config import Config


def main():
    """Main function to run the chatbot with evaluation"""
    print("🚀 SFU Admission Chatbot with RAG Evaluation")
    print("="*60)

    # Validate configuration
    try:
        Config.validate()
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        return

    try:
        print("\n🔧 Setting up components...")

        llm = LLMProvider(
            provider="deepseek",
            api_key=Config.DEEPSEEK_API_KEY,
            temperature=Config.LLM_TEMPERATURE
        )
        db = ChromaDBManager(
            persist_directory=Config.CHROMA_DB_DIR,
            collection_name=Config.CHROMA_COLLECTION_NAME
        )

        if db.collection.count() == 0:
            if os.path.exists(Config.DATA_FILE):
                db.add_documents_from_json(Config.DATA_FILE)
            else:
                print(f"❌ {Config.DATA_FILE} not found!")
                return
        else:
            print(f"📚 Loaded {db.collection.count()} documents from persistence.")

        chatbot = RAGChatbot(
            chroma_db=db, 
            llm_provider=llm,
            use_adaptive_config=Config.USE_ADAPTIVE_CONFIG
        )

        print("✅ Setup complete!")
        print(f"\n{'='*60}")
        print("💬 Interactive Chat Mode")
        print("   Type 'quit' to exit and generate evaluation report")
        print("   Type 'clear' to reset memory")
        print("   Type 'history' to view conversation history")
        print("   Type 'stats' to view current session statistics")
        print("="*60)

        while True:
            try:
                query = input("\nYou: ").strip()

                if query.lower() in ['quit', 'exit', 'bye']:
                    print("\nGoodbye! Generating evaluation report...")
                    generate_evaluation_dashboard(chatbot.session_metrics)
                    break

                if query.lower() == 'clear':
                    chatbot.memory.clear()
                    chatbot.session_metrics = []
                    print("✅ Memory and metrics cleared.")
                    continue

                if query.lower() == 'history':
                    history = chatbot.memory.get_recent_history()
                    print(f"\n📜 Conversation History ({len(history)} exchanges):")
                    for i, exchange in enumerate(history, 1):
                        print(f"\n{i}. User: {exchange['user_query']}")
                        print(f"   Bot: {exchange['bot_response'][:100]}...")
                    continue

                if query.lower() == 'stats':
                    if chatbot.session_metrics:
                        df = pd.DataFrame(chatbot.session_metrics)
                        print(f"\n📊 Current Session Statistics:")
                        print(f"   Total Queries: {len(df)}")
                        print(f"   Avg Response Time: {df['response_time'].mean():.3f}s")
                        print(f"   Avg Similarity: {df['avg_similarity'].mean():.3f}")
                        print(f"   Hit Rate: {df['hit'].sum() / len(df) * 100:.1f}%")
                    else:
                        print("No metrics available yet.")
                    continue

                if not query:
                    continue

                response = chatbot.chat(query)
                print(f"\n🤖 Bot: {response['answer']}")
                print(f"⏱️ [Response generated in {response['performance']['total_time']:.2f}s]")

            except KeyboardInterrupt:
                print("\n\nInterrupted. Generating evaluation report...")
                generate_evaluation_dashboard(chatbot.session_metrics)
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()

    except Exception as e:
        print(f"❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
