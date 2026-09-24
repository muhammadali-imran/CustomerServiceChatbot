import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

def load_and_split_documents(file_path: str, chunk_size: int = 500, chunk_overlap: int = 50):
    """
    Loads a text document, splits it into chunks, and returns the chunks.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Source file not found at: {file_path}")

    # Read file directly to bypass langchain-community deprecation warnings
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    documents = [Document(page_content=text, metadata={"source": file_path})]

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    return chunks

def build_vector_store(chunks, persist_directory: str = "./chroma_db"):
    """
    Builds a Chroma vector store from text chunks and automatically saves it to disk.
    """
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY is missing. Check your .env file.")

    # Create embeddings using Google Generative AI
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    # Chroma auto-persists when persist_directory is provided
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    return vector_store

if __name__ == "__main__":
    chunks = load_and_split_documents("data/faq.txt")
    print(f"Loaded {len(chunks)} chunks.")

    vectorstore = build_vector_store(chunks)
    print("Vectorstore successfully built and persisted to ./chroma_db")

