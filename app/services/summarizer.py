from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
#from langchain_community.chat_models import ChatDeepseek
from langchain_core.runnables import RunnablePassthrough
from app.utils.config import settings, LLMProvider

class Summarizer:
    def __init__(self):
        self.llm = self._initialize_llm()
        self.prompt = ChatPromptTemplate.from_template(
            "Summarize the following document in 3-5 bullet points:\n\n{document}"
        )

    def _initialize_llm(self):
        if settings.llm_provider == LLMProvider.OPENAI:
            return ChatOpenAI(
                model_name=settings.model_name,
                openai_api_key=settings.openai_api_key,
                base_url=settings.openai_base_url
            )
        elif settings.llm_provider == LLMProvider.DEEPSEEK:
            return ChatOpenAI(
                model_name=settings.model_name,
                deepseek_api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")

    def summarize(self, document: str) -> str:
        chain = (
            {"document": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return chain.invoke(document)