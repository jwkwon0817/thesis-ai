from typing import Optional

from app.pdf.pdf_utils import extract_pdf_title


class PDFDocument:
    def __init__(self, filename: str, title: Optional[str] = None, path: str = None):
        self.filename = filename
        self.title = title
        self.path = path
        self._extract_title()
    
    def _extract_title(self):
        if not self.title and self.path:
            self.title = extract_pdf_title(self.path) or "Title not found"
            
    def to_dict(self) -> dict:
        return {
            'filename': self.filename,
            'title': self.title,
            'path': self.path
        }
        
    @classmethod
    def from_dict(cls, data: dict) -> 'PDFDocument':
        """Create PDFDocument instance from dictionary"""
        return cls(
            filename=data['filename'],
            title=data['title'],
            path=data['path']
        )
    
    def __repr__(self) -> str:
        return f"PDFDocument(filename='{self.filename}', title='{self.title}', path='{self.path}')"