"""Configuration module for the SFU Admission Chatbot"""
import os
from typing import Optional, List
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
    LLM_MAX_TOKENS: int = 10000
    LLM_ENABLE_CACHE: bool = True  # Enable response caching
    LLM_ENABLE_STREAMING: bool = False  # Enable streaming (set to True for better UX)
    
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
    
    # Source URL Configuration
    SOURCE_BASE_URL: str = os.getenv("SOURCE_BASE_URL", "https://www.sfu.edu.hk")
    
    # OCR Configuration
    # Path to Tesseract executable (Windows default, update if installed elsewhere)
    # Download from: https://github.com/UB-Mannheim/tesseract/wiki
    TESSERACT_PATH: Optional[str] = os.getenv(
        "TESSERACT_PATH"
    )
    # OCR language: eng=English, chi_sim=Chinese Simplified, chi_tra=Chinese Traditional
    # Use + to combine languages, e.g., "eng+chi_sim+chi_tra"
    OCR_LANGUAGE: str = os.getenv("OCR_LANGUAGE", "eng+chi_sim+chi_tra")
    MIN_TEXT_LENGTH_FOR_OCR: int = 100  # Trigger OCR if extracted text is shorter
    
    # Document Processing Configuration
    SUPPORTED_EXTENSIONS: List[str] = [".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"]
    
    @classmethod
    def validate(cls):
        """Validate configuration"""
        if not cls.DEEPSEEK_API_KEY:
            raise ValueError("DEEPSEEK_API_KEY is not set")
        return True

