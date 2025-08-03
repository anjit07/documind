import chromadb
from chromadb.utils import embedding_functions
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from app.configuration.config import settings, LLMProvider

class VectorDB:
    def __init__(self):
        self.embeddings = self._initialize_embeddings()
        self.client = chromadb.PersistentClient(path=settings.chroma_db_path)
        
    def _initialize_embeddings(self):
        if settings.embedding_provider == LLMProvider.OPENAI:
            return OpenAIEmbeddings(
                model=settings.hugging_face_model,
                openai_api_key=settings.openai_api_key,
                base_url=settings.openai_base_url
            )
        elif settings.embedding_provider == LLMProvider.HUGGINGFACE:
            # Use HuggingFace embeddings
             print(f"Using HuggingFace model: {settings.hugging_face_model}")
             return HuggingFaceEmbeddings(
                        model_name=settings.hugging_face_model
                    )
        elif settings.embedding_provider == LLMProvider.DEEPSEEK:
            return OpenAIEmbeddings(
                model=settings.hugging_face_model,
                openai_api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
    
    def get_collection(self, collection_name: str):
        return Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=self.embeddings
        )
    
    def create_collection(self, collection_name: str, documents: list, ids: list):
        # Handle both str and Document objects
        if not all(isinstance(doc, str) for doc in documents):
            texts = [doc.page_content for doc in documents]
        else:
            texts = documents

        return Chroma.from_texts(
            texts=texts,
            embedding=self.embeddings,
            collection_name=collection_name,
            ids=ids,
            client=self.client
        )
    

    def search(self, collection_name: str, query: str, top_k: int = 3) -> list:
        """
        Dummy implementation: returns top_k chunks from the collection.
        Replace with actual vector similarity search logic.
        """
        collection = self.get_collection(collection_name)
        # Assume collection["documents"] is a list of chunk texts
        return collection["documents"][:top_k]