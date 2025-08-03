from enum import Enum
from pydantic_settings import BaseSettings

class LLMProvider(str, Enum):
    OPENAI = "openai"
    DEEPSEEK = "deepseek"
    HUGGINGFACE = "huggingface"
    OPENROUTER = "openrouter"

class Settings(BaseSettings):
    # General settings
    llm_provider: LLMProvider = LLMProvider.OPENROUTER
    embedding_provider: LLMProvider = LLMProvider.HUGGINGFACE
    chroma_db_path: str = "./chroma_db"
    model_name: str = "gpt-3.5-turbo"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    hugging_face_model: str = "all-MiniLM-L6-v2"

    # OpenAI-specific settings
    openai_api_key: str = "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"
    openai_base_url: str = "https://api.openai.com/v1"
    
    # DeepSeek-specific settings
    deepseek_api_key: str = "sk-ff6aa8d95bd64fafbb3c0abbc4e6607c"
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    
    # openrouter-specific settings
    openrouter_api_key: str = "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"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "deepseek/deepseek-chat-v3-0324:free"



    class Config:
        env_file = ".env"

settings = Settings()