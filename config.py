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

    # Kimi AI (Moonshot) Configuration
    KIMI_API_KEY: Optional[str] = os.getenv("KIMI_API_KEY")
    KIMI_BASE_URL: str = "https://api.moonshot.cn/v1"
    KIMI_MODEL: str = "kimi-k2.5"
    # For kimi-k2.5: thinking mode uses temperature 1.0; non-thinking uses 0.6. Set False to allow temperature 0.6.
    KIMI_DISABLE_THINKING: bool = os.getenv("KIMI_DISABLE_THINKING", "true").lower() in ("true", "1", "yes")

    # Gemini configuration (used by Ragas evaluation)
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    # Ragas eval parallelism
    RAGAS_EVAL_MAX_WORKERS: int = max(1, int(os.getenv("RAGAS_EVAL_MAX_WORKERS", "2")))
    
    # LLM Settings
    LLM_TEMPERATURE: float = 0.6
    LLM_MAX_TOKENS: int = 1024
    LLM_ENABLE_CACHE: bool = True  # Enable response caching
    LLM_ENABLE_STREAMING: bool = False  # Enable streaming (set to True for better UX)
    # Timeout in seconds for LLM API requests (Kimi can be slow to first token). Set in .env as LLM_REQUEST_TIMEOUT if needed.
    LLM_REQUEST_TIMEOUT: float = float(os.getenv("LLM_REQUEST_TIMEOUT", "300"))
    
    # Vector Database Configuration
    CHROMA_DB_DIR: str = "./chroma_db"
    CHROMA_COLLECTION_NAME: str = "sfu_admission"
    # Embedding and reranker models load from Hugging Face at startup. If you see ReadTimeoutError,
    # set in .env: HF_HUB_ETAG_TIMEOUT=300, HF_HUB_DOWNLOAD_TIMEOUT=300. HF_HUB_OFFLINE=1 uses cache only
    # (avoids timeouts but disables the reranker unless BAAI/bge-reranker-base is already cached).
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
    
    # Reranker Configuration
    USE_RERANKER: bool = True  # Enable cross-encoder reranking (BAAI/bge-reranker-base)
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"
    
    # Data Files
    DATA_FILE: str = "merged_rag_data.json"
    
    # Source URL Configuration
    SOURCE_BASE_URL: str = os.getenv("SOURCE_BASE_URL", "https://www.sfu.edu.hk")
    
    # RBS Booking Portal
    RBS_BOOKING_URL: str = os.getenv("RBS_BOOKING_URL", "https://rbs.cihe.edu.hk")
    
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
    SUPPORTED_EXTENSIONS: List[str] = [
        ".pdf",
        ".png",
        ".jpg",
        ".jpeg",
        ".tiff",
        ".bmp",
        ".txt",
        ".csv",
        ".docx",
        ".xlsx",
    ]
    
    # Email / IMAP Configuration (used by fetch_emails_to_rag.py)
    EMAIL_IMAP_HOST: Optional[str] = os.getenv("EMAIL_IMAP_HOST")
    EMAIL_IMAP_PORT: int = int(os.getenv("EMAIL_IMAP_PORT", "993"))
    EMAIL_IMAP_USER: Optional[str] = os.getenv("EMAIL_IMAP_USER")
    EMAIL_IMAP_PASSWORD: Optional[str] = os.getenv("EMAIL_IMAP_PASSWORD")
    EMAIL_FROM_FILTER: Optional[str] = os.getenv("EMAIL_FROM_FILTER")
    EMAIL_LAST_RUN_FILE: str = os.getenv("EMAIL_LAST_RUN_FILE", ".last_email_fetch")
    EMAIL_ASSETS_DIR: str = os.getenv("EMAIL_ASSETS_DIR", "./email_assets")
    
    @classmethod
    def validate(cls):
        """Validate configuration"""
        if not cls.DEEPSEEK_API_KEY:
            raise ValueError("DEEPSEEK_API_KEY is not set")
        return True

