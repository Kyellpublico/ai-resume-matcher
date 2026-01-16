import os
import logging
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_resume(file_path: str) -> dict:
    """
    Parses a PDF resume and returns structured data.
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return None

    # 1. Configure the Pipeline Options (OCR, Tables, etc.)
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True

    # 2. Wrap options in PdfFormatOption (The Fix)
    # The converter expects a specific wrapper, not just the raw options.
    format_options = {
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }

    # 3. Initialize Converter
    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options=format_options
    )

    try:
        logger.info(f"Processing: {file_path}")
        result = converter.convert(file_path)
        
        # Export to Markdown
        markdown_text = result.document.export_to_markdown()
        
        return {
            "filename": os.path.basename(file_path),
            "content": markdown_text,
            "meta": result.document.origin
        }

    except Exception as e:
        logger.error(f"Failed to parse {file_path}: {e}")
        return None

if __name__ == "__main__":
    # Test path
    test_file = os.path.join("data", "test_resume.pdf")
    
    if os.path.exists(test_file):
        data = parse_resume(test_file)
        if data:
            print("\n--- SUCCESS: Resume Parsed ---")
            print(f"Preview (First 500 chars):\n{data['content'][:500]}")
    else:
        logger.warning(f"Please place a file at {test_file} to test this script.")