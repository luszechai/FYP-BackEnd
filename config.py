"""Configuration module for the SFU Admission Chatbot"""
import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Application configuration"""
    
    # API Configuration
    # API key must be set via environment variable DEEPSEEK_API_KEY
    # Create a .env file with: DEEPSEEK_API_KEY=your_api_key_here
    DEEPSEEK_API_KEY: Optional[str] = os.getenv("DEEPSEEK_API_KEY")
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    DEEPSEEK_MODEL: str = "deepseek-chat"
    
    # LLM Settings
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 1000
    
    # Vector Database Configuration
    CHROMA_DB_DIR: str = "./chroma_db"
    CHROMA_COLLECTION_NAME: str = "sfu_admission"
    EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"
    
    # Text Splitting Configuration
    CHUNK_SIZE: int = 1600
    CHUNK_OVERLAP: int = 200
    
    # Retrieval Configuration
    RETRIEVAL_K: int = 5
    
    # Memory Configuration
    MAX_CONVERSATION_HISTORY: int = 10
    
    # Adaptive Configuration
    USE_ADAPTIVE_CONFIG: bool = True  # Enable automatic parameter adjustment
    
    # Data Files
    DATA_FILE: str = "merged_rag_data.json"
    
    @classmethod
    def validate(cls):
        """Validate configuration"""
        if not cls.DEEPSEEK_API_KEY:
            raise ValueError("DEEPSEEK_API_KEY is not set")
        return True

