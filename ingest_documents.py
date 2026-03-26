#!/usr/bin/env python3
"""
Document Ingestion CLI Script

This script allows you to add PDF files and images to the vector database.
It supports both individual files and batch processing of directories.

Usage:
    # Add a single PDF file
    python ingest_documents.py --file path/to/document.pdf
    
    # Add a single image file
    python ingest_documents.py --file path/to/image.png
    
    # Add all documents from a directory
    python ingest_documents.py --directory path/to/documents/
    
    # Add with custom source metadata
    python ingest_documents.py --file brochure.pdf --source "2024 Admissions Brochure"
    
    # Add only PDFs from a directory (recursive)
    python ingest_documents.py --directory docs/ --extensions .pdf --recursive
    
    # Check dependencies
    python ingest_documents.py --check-deps
"""

import argparse
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from src.vector_db import ChromaDBManager
from src.document_loader import print_dependency_status, check_dependencies


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Ingest PDF and image documents into the vector database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ingest_documents.py --file document.pdf
  python ingest_documents.py --file scan.png --source "Student Handbook"
  python ingest_documents.py --directory ./data/pdfs/
  python ingest_documents.py --directory ./data/ --extensions .pdf .png --recursive
  python ingest_documents.py --check-deps
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--file", "-f",
        type=str,
        help="Path to a single PDF or image file to ingest"
    )
    input_group.add_argument(
        "--directory", "-d",
        type=str,
        help="Path to a directory containing documents to ingest"
    )
    
    # Directory options
    parser.add_argument(
        "--extensions", "-e",
        type=str,
        nargs="+",
        default=None,
        help="File extensions to process (e.g., .pdf .png). Defaults to all supported."
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Recursively search subdirectories"
    )
    
    # Metadata options
    parser.add_argument(
        "--source", "-s",
        type=str,
        default=None,
        help="Custom source name/description for the document(s)"
    )
    parser.add_argument(
        "--section",
        type=str,
        default=None,
        help="Section/category name for the document(s)"
    )
    
    # OCR options
    parser.add_argument(
        "--ocr-language",
        type=str,
        default=None,
        help=f"OCR language code (default: {Config.OCR_LANGUAGE})"
    )
    parser.add_argument(
        "--tesseract-path",
        type=str,
        default=None,
        help="Path to Tesseract executable"
    )
    parser.add_argument(
        "--min-text-length",
        type=int,
        default=None,
        help=f"Minimum text length before triggering OCR (default: {Config.MIN_TEXT_LENGTH_FOR_OCR})"
    )
    
    # Utility options
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check if all dependencies are installed"
    )
    parser.add_argument(
        "--db-stats",
        action="store_true",
        help="Show database statistics"
    )
    
    return parser.parse_args()


def get_ocr_config(args):
    """Get OCR configuration from args or config file"""
    return {
        'ocr_language': args.ocr_language or Config.OCR_LANGUAGE,
        'tesseract_path': args.tesseract_path or Config.TESSERACT_PATH,
        'min_text_length': args.min_text_length or Config.MIN_TEXT_LENGTH_FOR_OCR,
    }


def build_metadata(args):
    """Build metadata dict from command line arguments"""
    metadata = {}
    
    if args.source:
        metadata['custom_source'] = args.source
    
    if args.section:
        metadata['section'] = args.section
    
    return metadata if metadata else None


def ingest_file(db: ChromaDBManager, file_path: str, args):
    """Ingest a single file"""
    ocr_config = get_ocr_config(args)
    metadata = build_metadata(args)

    # Delegate to generic file ingestion; validation is handled there
    try:
        chunks_added = db.add_documents_from_file(
            file_path=file_path,
            metadata=metadata,
            min_text_length=ocr_config["min_text_length"],
            ocr_language=ocr_config["ocr_language"],
            tesseract_path=ocr_config["tesseract_path"],
        )
    except ValueError as e:
        # Unsupported extension or loader error
        print(f"❌ Error ingesting file: {e}")
        print(f"   Supported: {', '.join(Config.SUPPORTED_EXTENSIONS)}")
        return 0

    return chunks_added


def ingest_directory(db: ChromaDBManager, directory: str, args):
    """Ingest all documents from a directory"""
    ocr_config = get_ocr_config(args)
    metadata = build_metadata(args)
    
    chunks_added = db.add_documents_from_directory(
        directory=directory,
        extensions=args.extensions,
        recursive=args.recursive,
        metadata=metadata,
        min_text_length=ocr_config['min_text_length'],
        ocr_language=ocr_config['ocr_language'],
        tesseract_path=ocr_config['tesseract_path']
    )
    
    return chunks_added


def show_db_stats(db: ChromaDBManager):
    """Show database statistics"""
    count = db.collection.count()
    print("\n📊 Database Statistics")
    print("-" * 40)
    print(f"  Collection: {db.collection_name}")
    print(f"  Total documents: {count}")
    print(f"  Persist directory: {db.persist_directory}")
    print("-" * 40)


def main():
    """Main entry point"""
    args = parse_args()
    
    # Check dependencies
    if args.check_deps:
        print_dependency_status()
        deps = check_dependencies()
        if all(deps.values()):
            print("\n✅ All dependencies are available!")
            sys.exit(0)
        else:
            print("\n⚠️ Some dependencies are missing. Install them to enable full functionality.")
            sys.exit(1)
    
    # Show stats only
    if args.db_stats and not args.file and not args.directory:
        print("🔄 Connecting to database...")
        db = ChromaDBManager(
            persist_directory=Config.CHROMA_DB_DIR,
            collection_name=Config.CHROMA_COLLECTION_NAME
        )
        show_db_stats(db)
        sys.exit(0)
    
    # Validate input
    if not args.file and not args.directory:
        print("❌ Error: Please specify --file or --directory")
        print("   Use --help for usage information")
        sys.exit(1)
    
    # Validate file exists
    if args.file and not os.path.exists(args.file):
        print(f"❌ Error: File not found: {args.file}")
        sys.exit(1)
    
    # Validate directory exists
    if args.directory and not os.path.isdir(args.directory):
        print(f"❌ Error: Directory not found: {args.directory}")
        sys.exit(1)
    
    # Check required dependencies
    deps = check_dependencies()
    if not deps['pymupdf']:
        print("❌ Error: PyMuPDF is required for PDF processing")
        print("   Install with: pip install pymupdf")
        sys.exit(1)
    
    if args.file:
        ext = os.path.splitext(args.file)[1].lower()
        if ext in ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp']:
            if not deps['pytesseract'] or not deps['tesseract_executable']:
                print("❌ Error: Tesseract OCR is required for image processing")
                print("   1. Install pytesseract: pip install pytesseract Pillow")
                print("   2. Install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki")
                sys.exit(1)
    
    # Initialize database
    print("🔄 Connecting to database...")
    db = ChromaDBManager(
        persist_directory=Config.CHROMA_DB_DIR,
        collection_name=Config.CHROMA_COLLECTION_NAME
    )
    
    initial_count = db.collection.count()
    print(f"📊 Current documents in database: {initial_count}")
    print()
    
    # Process input
    try:
        if args.file:
            chunks_added = ingest_file(db, args.file, args)
        else:
            chunks_added = ingest_directory(db, args.directory, args)
        
        final_count = db.collection.count()
        
        print()
        print("=" * 50)
        print("📊 Ingestion Summary")
        print("=" * 50)
        print(f"  Chunks added: {chunks_added}")
        print(f"  Previous total: {initial_count}")
        print(f"  New total: {final_count}")
        print("=" * 50)
        
        if args.db_stats:
            show_db_stats(db)
        
    except Exception as e:
        print(f"\n❌ Error during ingestion: {e}")
        sys.exit(1)
    
    print("\n✅ Ingestion complete!")


if __name__ == "__main__":
    main()
