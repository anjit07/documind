from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from openai import OpenAI
from app.configuration.config import settings, LLMProvider

class Summarizer:

     def __init__(self):
         self.llm = self._initialize_llm()
         self.openai = self._initialize_openai()

     def _initialize_llm(self):
        if settings.llm_provider == LLMProvider.OPENROUTER:
            return ChatOpenAI(
                model_name=settings.openrouter_model,
                openai_api_key=settings.openrouter_api_key,
                base_url=settings.openrouter_base_url
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
    
     def _initialize_openai(self):
         return OpenAI(
             api_key=settings.openrouter_api_key,
             base_url=settings.openrouter_base_url
         )


     def summarize(self, document: str) -> str:
        prompt = ChatPromptTemplate.from_template(
            "Summarize the following document in 3-5 bullet points:\n\n{document}"
        )
        chain = (
            {"document": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain.invoke(document)
     
     def ask(self, document: str, query: str) -> str:
        prompt = ChatPromptTemplate.from_template(
            "Answer the following question based on the document:\n\n{document}\n\nQuestion: {query}"
        )
        chain = (
            {"document": RunnablePassthrough(), "query": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain.invoke({"document": document, "query": query})
     
     def chart_with(self, context: str, query: str) -> str:
        prompt = f"""Use the following context to answer the question:

        Context:
        {context}

        Question: {query}
        Answer:"""

        response = self.openai.chat.completions.create(
            model=settings.openrouter_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        print("response:", response)
        return response.choices[0].message.content