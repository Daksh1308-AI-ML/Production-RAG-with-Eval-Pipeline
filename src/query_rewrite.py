"""Query rewriting and expansion."""

from typing import List

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from .config import config


class QueryRewriter:
    """Rewrite and expand queries."""
    
    REWRITE_PROMPT = """You are a query rewriting assistant.
Given the user's question, generate 3-5 alternative search queries that would help
find relevant information. Focus on different phrasings and keywords.

Original question: {query}

Output only the rewritten queries, one per line:"""
    
    EXPAND_PROMPT = """You are a query expansion assistant.
Given the user's question, expand it with relevant context and synonyms.
Make the query more specific and searchable.

Original question: {query}

Expanded query:"""
    
    def __init__(self):
        self.llm = ChatOllama(
            model=config.ollama.llm_model,
            temperature=0.3,
            base_url=config.ollama.base_url
        )
    
    def rewrite(self, query: str) -> List[str]:
        """Generate multiple search queries."""
        prompt = ChatPromptTemplate.from_template(self.REWRITE_PROMPT)
        chain = prompt | self.llm
        
        response = chain.invoke({"query": query})
        
        # Parse response
        queries = [q.strip() for q in response.content.split("\n") if q.strip()]
        return queries[:5]
    
    def expand(self, query: str) -> str:
        """Expand a single query."""
        prompt = ChatPromptTemplate.from_template(self.EXPAND_PROMPT)
        chain = prompt | self.llm
        
        response = chain.invoke({"query": query})
        return response.content.strip()
    
    def should_rewrite(self, query: str) -> bool:
        """Determine if query needs rewriting."""
        # Rewrite if query is short or ambiguous
        words = query.split()
        if len(words) < 5:
            return True
        if "?" not in query and len(words) < 10:
            return True
        return False
