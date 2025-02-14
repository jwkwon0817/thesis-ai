from typing import Optional

import PyPDF2


def extract_pdf_title(pdf_path: str) -> Optional[str]:
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            
            if reader.metadata and reader.metadata.get('/Title'):
                return reader.metadata['/Title']
            
            first_page = reader.pages[0].extract_text()
            first_lines = first_page.split('\n')[:3]
            return max(first_lines, key=len).strip()
            
    except Exception as e:
        print(f"Error extracting title from {pdf_path}: {str(e)}")
        return None