import json
import os
from typing import Dict, List

from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader

from app.pdf.document import PDFDocument


def save_pdf_docs(pdf_docs: Dict[str, PDFDocument], directory: str = "./databases/titleMap"):
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, "pdf_titles.json")
    
    docs_dict = {
        filename: doc.to_dict()
        for filename, doc in pdf_docs.items()
    }
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(docs_dict, f, ensure_ascii=False, indent=2)

def load_pdf_docs(directory: str = "./databases/titleMap") -> Dict[str, PDFDocument]:
    file_path = os.path.join(directory, "pdf_titles.json")
    
    if not os.path.exists(file_path):
        return {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        docs_dict = json.load(f)
    
    return {
        filename: PDFDocument.from_dict(doc_data)
        for filename, doc_data in docs_dict.items()
    }

def load_pdfs_from_directory(directory_path: str, pdf_docs: Dict[str, PDFDocument]) -> List[Document]:
    documents = []
    
    for filename in os.listdir(directory_path):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(directory_path, filename)
            
            try:
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                documents.extend(docs)
                
                if filename not in pdf_docs:
                    pdf_docs[filename] = PDFDocument(
                        filename=filename,
                        path=file_path
                    )
                print(f"Successfully loaded: {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")
    
    save_pdf_docs(pdf_docs)
    
    return documents