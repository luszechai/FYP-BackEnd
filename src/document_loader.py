"""Document loader module for PDF and image files with OCR support"""
import os
import hashlib
from typing import List, Dict, Optional, Tuple
from abc import ABC, abstractmethod

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import pytesseract
    from PIL import Image
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False


class DocumentLoader(ABC):
    """Abstract base class for document loaders"""
    
    @abstractmethod
    def load(self, file_path: str) -> List[Dict]:
        """
        Load a document and return extracted text with metadata.
        
        Returns:
            List of dicts with 'content', 'metadata' keys
        """
        pass
    
    @staticmethod
    def generate_doc_id(file_path: str) -> str:
        """Generate a unique document ID based on file path"""
        return hashlib.md5(file_path.encode()).hexdigest()[:12]


class PDFLoader(DocumentLoader):
    """
    Loader for PDF files.
    Uses PyMuPDF for text extraction, falls back to OCR for scanned pages.
    """
    
    def __init__(
        self,
        min_text_length: int = 100,
        ocr_language: str = "eng+chi_sim+chi_tra",
        tesseract_path: Optional[str] = None
    ):
        """
        Initialize PDF loader.
        
        Args:
            min_text_length: Minimum text length before triggering OCR
            ocr_language: Language for OCR (default: English + Chinese Simplified + Traditional)
            tesseract_path: Path to Tesseract executable (Windows)
        """
        if not PYMUPDF_AVAILABLE:
            raise ImportError("PyMuPDF (fitz) is required. Install with: pip install pymupdf")
        
        self.min_text_length = min_text_length
        self.ocr_language = ocr_language
        
        if tesseract_path and TESSERACT_AVAILABLE:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
    
    def load(self, file_path: str) -> List[Dict]:
        """
        Load a PDF file and extract text from all pages.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of documents (one per page) with content and metadata
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        documents = []
        doc_id = self.generate_doc_id(file_path)
        
        try:
            pdf_doc = fitz.open(file_path)
            total_pages = len(pdf_doc)
            
            print(f"📄 Loading PDF: {os.path.basename(file_path)} ({total_pages} pages)")
            
            for page_num in range(total_pages):
                page = pdf_doc[page_num]
                text = page.get_text().strip()
                extraction_method = "text"
                
                # If text extraction yields little content, try OCR
                if len(text) < self.min_text_length:
                    ocr_text = self._ocr_page(page)
                    if ocr_text and len(ocr_text) > len(text):
                        text = ocr_text
                        extraction_method = "ocr"
                
                if text:
                    documents.append({
                        'content': text,
                        'metadata': {
                            'source': file_path,
                            'type': 'pdf',
                            'page': page_num + 1,
                            'total_pages': total_pages,
                            'extraction_method': extraction_method,
                            'parent_doc_id': doc_id
                        }
                    })
                    print(f"  📃 Page {page_num + 1}: {len(text)} chars ({extraction_method})")
            
            pdf_doc.close()
            
        except Exception as e:
            raise Exception(f"Error loading PDF {file_path}: {e}")
        
        return documents
    
    def _ocr_page(self, page) -> Optional[str]:
        """
        Perform OCR on a PDF page.
        
        Args:
            page: PyMuPDF page object
            
        Returns:
            Extracted text or None if OCR is not available
        """
        if not TESSERACT_AVAILABLE:
            print("    ⚠️ Tesseract not available for OCR")
            return None
        
        try:
            # Render page to image at higher resolution for better OCR
            mat = fitz.Matrix(2, 2)  # 2x zoom for better quality
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Perform OCR
            text = pytesseract.image_to_string(img, lang=self.ocr_language)
            return text.strip()
            
        except Exception as e:
            print(f"    ⚠️ OCR failed: {e}")
            return None


class ImageLoader(DocumentLoader):
    """
    Loader for image files using OCR.
    Supports PNG, JPG, JPEG, TIFF, BMP formats.
    """
    
    SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp'}
    
    def __init__(
        self,
        ocr_language: str = "eng+chi_sim+chi_tra",
        tesseract_path: Optional[str] = None
    ):
        """
        Initialize image loader.
        
        Args:
            ocr_language: Language for OCR (default: English + Chinese Simplified + Traditional)
            tesseract_path: Path to Tesseract executable (Windows)
        """
        if not TESSERACT_AVAILABLE:
            raise ImportError(
                "pytesseract and Pillow are required. Install with: "
                "pip install pytesseract Pillow"
            )
        
        self.ocr_language = ocr_language
        
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
    
    def load(self, file_path: str) -> List[Dict]:
        """
        Load an image file and extract text using OCR.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            List with single document containing extracted text and metadata
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image file not found: {file_path}")
        
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported image format: {ext}. "
                f"Supported: {', '.join(self.SUPPORTED_EXTENSIONS)}"
            )
        
        doc_id = self.generate_doc_id(file_path)
        
        print(f"🖼️ Loading image: {os.path.basename(file_path)}")
        
        try:
            img = Image.open(file_path)
            
            # Convert to RGB if necessary (e.g., for RGBA images)
            if img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')
            
            # Perform OCR
            text = pytesseract.image_to_string(img, lang=self.ocr_language)
            text = text.strip()
            
            if not text:
                print(f"  ⚠️ No text extracted from image")
                return []
            
            print(f"  ✅ Extracted {len(text)} characters")
            
            return [{
                'content': text,
                'metadata': {
                    'source': file_path,
                    'type': 'image',
                    'format': ext[1:],  # Remove the dot
                    'extraction_method': 'ocr',
                    'parent_doc_id': doc_id,
                    'image_size': f"{img.width}x{img.height}"
                }
            }]
            
        except Exception as e:
            raise Exception(f"Error loading image {file_path}: {e}")


class DocumentLoaderFactory:
    """
    Factory class that auto-detects file type and returns appropriate loader.
    """
    
    PDF_EXTENSIONS = {'.pdf'}
    IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp'}
    
    def __init__(
        self,
        min_text_length: int = 100,
        ocr_language: str = "eng+chi_sim+chi_tra",
        tesseract_path: Optional[str] = None
    ):
        """
        Initialize the factory with common configuration.
        
        Args:
            min_text_length: Minimum text length before triggering OCR for PDFs
            ocr_language: Language for OCR (default: English + Chinese Simplified + Traditional)
            tesseract_path: Path to Tesseract executable
        """
        self.min_text_length = min_text_length
        self.ocr_language = ocr_language
        self.tesseract_path = tesseract_path
    
    def get_loader(self, file_path: str) -> DocumentLoader:
        """
        Get the appropriate loader for a file based on its extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Appropriate DocumentLoader instance
        """
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext in self.PDF_EXTENSIONS:
            return PDFLoader(
                min_text_length=self.min_text_length,
                ocr_language=self.ocr_language,
                tesseract_path=self.tesseract_path
            )
        elif ext in self.IMAGE_EXTENSIONS:
            return ImageLoader(
                ocr_language=self.ocr_language,
                tesseract_path=self.tesseract_path
            )
        else:
            raise ValueError(
                f"Unsupported file type: {ext}. "
                f"Supported: PDF ({', '.join(self.PDF_EXTENSIONS)}), "
                f"Images ({', '.join(self.IMAGE_EXTENSIONS)})"
            )
    
    def load(self, file_path: str) -> List[Dict]:
        """
        Load a document using the appropriate loader.
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of documents with content and metadata
        """
        loader = self.get_loader(file_path)
        return loader.load(file_path)
    
    def load_directory(
        self,
        directory: str,
        extensions: Optional[List[str]] = None,
        recursive: bool = False
    ) -> List[Dict]:
        """
        Load all supported documents from a directory.
        
        Args:
            directory: Path to the directory
            extensions: Optional list of extensions to filter (e.g., ['.pdf', '.png'])
            recursive: Whether to search subdirectories
            
        Returns:
            List of all documents from all files
        """
        if not os.path.isdir(directory):
            raise NotADirectoryError(f"Directory not found: {directory}")
        
        # Default to all supported extensions
        if extensions is None:
            extensions = list(self.PDF_EXTENSIONS | self.IMAGE_EXTENSIONS)
        
        # Normalize extensions
        extensions = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' 
                      for ext in extensions]
        
        all_documents = []
        files_processed = 0
        
        print(f"📁 Scanning directory: {directory}")
        print(f"   Extensions: {', '.join(extensions)}")
        
        if recursive:
            for root, _, files in os.walk(directory):
                for filename in files:
                    file_path = os.path.join(root, filename)
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in extensions:
                        try:
                            docs = self.load(file_path)
                            all_documents.extend(docs)
                            files_processed += 1
                        except Exception as e:
                            print(f"   ⚠️ Error loading {filename}: {e}")
        else:
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                if os.path.isfile(file_path):
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in extensions:
                        try:
                            docs = self.load(file_path)
                            all_documents.extend(docs)
                            files_processed += 1
                        except Exception as e:
                            print(f"   ⚠️ Error loading {filename}: {e}")
        
        print(f"✅ Processed {files_processed} files, extracted {len(all_documents)} document segments")
        
        return all_documents


def check_dependencies() -> Dict[str, bool]:
    """
    Check which dependencies are available.
    
    Returns:
        Dict with dependency availability status
    """
    status = {
        'pymupdf': PYMUPDF_AVAILABLE,
        'pytesseract': TESSERACT_AVAILABLE,
        'pdf2image': PDF2IMAGE_AVAILABLE,
    }
    
    # Check if Tesseract executable is accessible
    if TESSERACT_AVAILABLE:
        try:
            pytesseract.get_tesseract_version()
            status['tesseract_executable'] = True
        except Exception:
            status['tesseract_executable'] = False
    else:
        status['tesseract_executable'] = False
    
    return status


def print_dependency_status():
    """Print the status of all dependencies"""
    status = check_dependencies()
    
    print("\n📦 Document Loader Dependencies:")
    print("-" * 40)
    
    for dep, available in status.items():
        icon = "✅" if available else "❌"
        print(f"  {icon} {dep}: {'Available' if available else 'Not available'}")
    
    if not status['tesseract_executable']:
        print("\n⚠️  Tesseract OCR is not installed or not in PATH.")
        print("   Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
        print("   Then add to PATH or set TESSERACT_PATH in config.py")
    
    print("-" * 40)
