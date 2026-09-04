"""LLM answer generation with citations."""

from typing import List, Generator

from langchain.schema import Document
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .config import config


class AnswerGenerator:
    """Generate answers with source citations."""
    
    SYSTEM_PROMPT = """You are a helpful assistant that answers questions about SEC 10-K filings.
Answer the question based ONLY on the provided context.
If the context doesn't contain enough information, say "I don't have enough information to answer this question."
Always cite your sources by referencing the document section."""
    
    CONTEXT_PROMPT = """Context:
{context}

Question: {question}

Answer (include citations to source documents):"""
    
    def __init__(self):
        self.llm = ChatOllama(
            model=config.ollama.llm_model,
            temperature=config.ollama.temperature,
            num_ctx=config.ollama.num_ctx,
            base_url=config.ollama.base_url
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT),
            ("human", self.CONTEXT_PROMPT)
        ])
        
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    def _format_context(self, documents: List[Document]) -> str:
        """Format documents into context string."""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "Unknown")
            section = doc.metadata.get("section", "general")
            context_parts.append(
                f"[Source {i}: {source} - {section}]\n{doc.page_content}"
            )
        return "\n\n".join(context_parts)
    
    def generate(self, query: str, documents: List[Document]) -> str:
        """Generate an answer."""
        context = self._format_context(documents)
        
        response = self.chain.invoke({
            "context": context,
            "question": query
        })
        
        return response
    
    def stream(self, query: str, documents: List[Document]) -> Generator[str, None, None]:
        """Stream an answer."""
        context = self._format_context(documents)
        
        for chunk in self.chain.stream({
            "context": context,
            "question": query
        }):
            yield chunk
