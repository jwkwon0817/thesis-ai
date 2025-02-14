import os

from dotenv import load_dotenv

from app import (create_chunks, create_rag_chain, get_answer_from_rag,
                 get_or_create_vector_store)
from app.pdf.storage import load_pdf_docs, load_pdfs_from_directory

load_dotenv()

def get_absolute_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))

def main():
    asset_directory = './assets'
    persist_directory = get_absolute_path("../databases/chroma")
    base_url = os.getenv('BASE_URL')
    model_name = os.getenv('MODEL_NAME')
    embedding_model_name = os.getenv('EMBEDDING_MODEL_NAME')

    pdf_docs = load_pdf_docs()

    if not base_url:
        print("Error: BASE_URL environment variable not found")
        return

    chunks = None
    if not os.path.exists(persist_directory):
        print("\nLoading documents...")
        try:
            documents = load_pdfs_from_directory(asset_directory)
            print(f"\nTotal documents loaded: {len(documents)}")
            print("\nLoaded PDF docs:")
            for filename, doc in pdf_docs.items():
                print(f"- {filename}: {doc.title}")

            chunks = create_chunks(
                documents,
                chunk_size=1000,
                chunk_overlap=200
            )

            print("\nChunk samples:")
            for i, chunk in enumerate(chunks[:2]):
                print(f"\nChunk {i+1}:")
                print(f"Length: {len(chunk.page_content)} characters")
                print(f"Content preview: {chunk.page_content[:200]}...")

        except Exception as e:
            print(f"Error during document loading/chunking: {str(e)}")
            return

    try:
        vectordb = get_or_create_vector_store(
            chunks,
            persist_directory=persist_directory,
            model_name=embedding_model_name
        )

        chain = create_rag_chain(
            vectorstore=vectordb,
            model_name=model_name,
            base_url=base_url
        )

        while True:
            query = input('Question: ')
            if query.lower() in ['/bye', 'quit', 'exit']:
                break

            print('\nAnswer: ', end='')
            try:
                result = get_answer_from_rag(query, chain)
                
                refs = set()
                source_docs_len = len(result['source_documents'])
                
                for doc in result["source_documents"]:
                    full_path = doc.metadata['source']
                    filename = os.path.basename(full_path)
                    refs.add((filename, doc.metadata['page_label']))
                    

                print(f"\n\nReferences (found {source_docs_len} documents, {source_docs_len - len(refs)} duplicated):")
                for filename, page_label in sorted(refs):
                    doc = pdf_docs[filename]
                    print(f"- {filename} | {doc.title} (p.{page_label})")
                print('\n')

            except Exception as e:
                print(f"\nError processing query: {str(e)}")
                print("Please try another question or type '/bye' to exit.")

    except Exception as e:
        print(f"Error setting up RAG system: {str(e)}")
        return

    print('Goodbye!')

if __name__ == "__main__":
    main()