from enum import Enum
from pydantic_settings import BaseSettings

class LLMProvider(str, Enum):
    OPENAI = "openai"
    DEEPSEEK = "deepseek"

class Settings(BaseSettings):
    # General settings
    llm_provider: LLMProvider = LLMProvider.OPENAI
    chroma_db_path: str = "./chroma_db"
    model_name: str = "gpt-3.5-turbo"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # OpenAI-specific settings
    openai_api_key: str = "sk-ff6aa8d95bd64fafbb3c0abbc4e6607c"
    openai_base_url: str = "https://api.openai.com/v1"
    
    # DeepSeek-specific settings
    deepseek_api_key: str = "sk-ff6aa8d95bd64fafbb3c0abbc4e6607c"
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    
    class Config:
        env_file = ".env"

settings = Settings()