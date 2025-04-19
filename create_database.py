from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_community.embeddings.bedrock import BedrockEmbeddings
import os
import shutil
from dotenv import load_dotenv

# Load environment variables (optional for Ollama)
load_dotenv()

CHROMA_PATH = "chroma"
DATA_PATH = "data/podcasts_transscript"
PDF_DATA_PATH = "data/pdfs"

def main():
    # generate_data_store() - uncomment for md files
    generate_data_store_for_pdfs()

def generate_data_store_for_pdfs():
    documents = load_pdf_documents()
    chunks = split_documents(documents)
    save_to_chroma(chunks)

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

def load_pdf_documents():
    document_loader = PyPDFDirectoryLoader(PDF_DATA_PATH)
    documents = document_loader.load()
    return documents 

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    return documents

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )

    return text_splitter.split_documents(documents)

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks

def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents using Ollama embeddings.
    db = Chroma.from_documents(
        chunks, OllamaEmbeddings(model="nomic-embed-text"), persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

if __name__ == "__main__":
    main()
