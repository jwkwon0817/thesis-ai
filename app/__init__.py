import json
import os
from typing import List

import PyPDF2
from dotenv import load_dotenv
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import ChatOllama, OllamaEmbeddings

load_dotenv()
  
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