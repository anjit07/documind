from typing import Any, Dict, List, Optional

from langchain_core.embeddings import Embeddings
import requests

class DeepseekEmbeddings(Embeddings):
    """Deepseek Embeddings wrapper"""

    model_name: str = "text-embedding"
    deepseek_api_key: str
    base_url: str = "https://api.deepseek.com/v1"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using Deepseek API"""
        embeddings = []
        for text in texts:
            embeddings.append(self._get_embedding(text))
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query using Deepseek API"""
        return self._get_embedding(text)

    def _get_embedding(self, text: str) -> List[float]:
        headers = {
            "Authorization": f"Bearer {self.deepseek_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "input": text
        }
        
        response = requests.post(
            f"{self.base_url}/embeddings",
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            raise ValueError(f"Deepseek API error: {response.text}")
        
        return response.json()["data"][0]["embedding"]