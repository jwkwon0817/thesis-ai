from langchain_community.document_loaders import PyPDFLoader
import os
import json
from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

import PyPDF2
from typing import Dict, Optional

from dotenv import load_dotenv

load_dotenv()

def load_pdfs_from_directory(directory_path: str) -> List[Document]:
    documents = []
    
    for filename in os.listdir(directory_path):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(directory_path, filename)
            
            try:
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                documents.extend(docs)
                
                pdf_docs[filename] = PDFDocument(
                    filename=filename,
                    path=file_path
                )
                print(f"Successfully loaded: {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")
    
    return documents
  
def create_chunks(documents: List[Document], 
                 chunk_size: int = 1000, 
                 chunk_overlap: int = 200) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks
  
  
def get_or_create_vector_store(
    chunks: List[Document] = None,
    persist_directory: str = "../databases/chroma.db",
    model_name: str = "nomic-embed-text"
):
    base_url = os.getenv('BASE_URL')
    if not base_url:
        raise ValueError("BASE_URL not found in .env file")
    
    embeddings = OllamaEmbeddings(
        model=model_name,
        base_url=base_url
    )
    
    if os.path.exists(persist_directory):
        print(f"Loading existing vector database from {persist_directory}")
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
    else:
        if chunks is None:
            raise ValueError("chunks must be provided to create new vector database")
        
        print(f"Creating new vector database in {persist_directory}")
        os.makedirs(os.path.dirname(persist_directory), exist_ok=True)
        return Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory
        )
      

def create_rag_chain(
    vectorstore,
    model_name: str = 'exaone3.5:32b',
    base_url: str = None
):
    callbacks = [StreamingStdOutCallbackHandler()]
  
    llm = ChatOllama(
        model=model_name,
        temperature=0,
        base_url=base_url,
        streaming=True,
        callbacks=callbacks
    )
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )
    
    return chain
  
def get_answer_from_rag(
    query: str,
    chain,
) -> str:
    result = chain.invoke({"question": query, 'chat_history': []})
    
    # print("\nRaw Result Object:")
    # pprint(result, indent=2, width=80, depth=None)
    
    return {
        "answer": result["answer"],
        "source_documents": result["source_documents"]
    }
    
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
      
def save_pdf_docs(pdf_docs: Dict[str, PDFDocument], directory: str = "./databases/titleMap"):
    """Save PDF documents to JSON file"""
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

def load_pdfs_from_directory(directory_path: str) -> List[Document]:
    documents = []
    global pdf_docs
    
    pdf_docs = load_pdf_docs()
    
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

pdf_docs: Dict[str, PDFDocument] = load_pdf_docs()