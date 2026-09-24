import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_core.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

def load_and_split_documents(file_path:str , chunk_size:int = 500, chunk_overlap:int = 50):
    """
    Loads a text document from the specified file path, splits it into chunks, and returns the chunks.
    Args:
        file_path (str): The path to the text document.
        chunk_size (int): The maximum size of each chunk. Default is 500 characters.
        chunk_overlap (int): The number of overlapping characters between chunks. Default is 50 characters.
    Returns:
        list: A list of text chunks.
    """
    #load the document
    loader = TextLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunk = splitter.split_documents(documents)
    return chunk

def build_vector_store(chunks, persist_directory:str = "./chroma_db"):
    """
    Builds a vector store from the provided text chunks and persists it to disk.
    Args:
        chunks (list): A list of text chunks.
        persist_directory (str): The directory where the vector store will be persisted. Default is "./chroma_db".
    Returns:
        Chroma: The built vector store.
    """
    # Create embeddings using Google Generative AI
    embeddings = ChatGoogleGenerativeAIEmbeddings(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    # Create a Chroma vector store from the chunks and embeddings
    vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory)
    
    # Persist the vector store to disk
    vector_store.persist()
    
    return vector_store

if __name__ == "__main__":
    chunks = load_and_split_documents("data/faq.txt")
    print(f"Loaded {len(chunks)} chunks.")

    vectorstore = build_vector_store(chunks)
    print("Vectorstore built and persisted to ./chroma_db")

