"""
RAG query engine using knowledge graph for enhanced retrieval.
"""
import os
from typing import List, Dict, Any, Optional, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from ..graph.database import Neo4jDatabase

class GraphRAGQueryEngine:
    """Query engine for graph-based RAG."""
    
    def __init__(
        self, 
        neo4j_db: Neo4jDatabase,
        embedding_model: str = "all-MiniLM-L6-v2",
        openai_api_key: Optional[str] = None
    ):
        """
        Initialize the Graph RAG query engine.
        
        Args:
            neo4j_db: Neo4j database instance
            embedding_model: Sentence transformer model for embeddings
            openai_api_key: OpenAI API key (defaults to environment variable)
        """
        self.db = neo4j_db
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Set up LLM
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required")
        
        self.llm = OpenAI(api_key=self.openai_api_key, temperature=0.2)
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """
        Get embedding vector for a query.
        
        Args:
            query: Query text
            
        Returns:
            Embedding vector
        """
        return self.embedding_model.encode(query).tolist()
    
    def _retrieve_relevant_entities(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant entities using semantic search.
        
        Args:
            query: Query text
            top_k: Number of top entities to retrieve
            
        Returns:
            List of relevant entities
        """
        query_embedding = self._get_query_embedding(query)
        return self.db.semantic_search(query_embedding, top_k=top_k)
    
    def _expand_graph_context(self, entity_ids: List[str], max_distance: int = 2) -> Dict[str, Any]:
        """
        Expand graph context from seed entities.
        
        Args:
            entity_ids: List of entity IDs to expand from
            max_distance: Maximum path distance
            
        Returns:
            Dict with expanded graph context
        """
        expanded_context = {
            "nodes": [],
            "relationships": []
        }
        
        for entity_id in entity_ids:
            neighbors = self.db.get_entity_neighbors(entity_id, max_distance=max_distance)
            
            # Merge results
            expanded_context["nodes"].extend(neighbors["nodes"])
            expanded_context["relationships"].extend(neighbors["relationships"])
        
        return expanded_context
    
    def _format_graph_context(self, graph_context: Dict[str, Any]) -> str:
        """
        Format graph context for LLM prompt.
        
        Args:
            graph_context: Graph context with nodes and relationships
            
        Returns:
            Formatted context string
        """
        context_str = "Knowledge Graph Context:\n"
        
        # Format nodes
        context_str += "Entities:\n"
        for node in graph_context["nodes"]:
            labels = ", ".join(node.get("labels", []))
            context_str += f"- {node.get('text', 'Unknown')} (Type: {labels})\n"
        
        # Format relationships
        context_str += "\nRelationships:\n"
        
        # Create a node lookup for relationship formatting
        node_lookup = {node["id"]: node for node in graph_context["nodes"]}
        
        for rel in graph_context["relationships"]:
            source_id = rel["source"]
            target_id = rel["target"]
            
            if source_id in node_lookup and target_id in node_lookup:
                source_text = node_lookup[source_id].get("text", "Unknown")
                target_text = node_lookup[target_id].get("text", "Unknown")
                rel_type = rel["type"]
                
                context_str += f"- {source_text} --[{rel_type}]--> {target_text}\n"
        
        return context_str
    
    def query(self, query: str, top_k: int = 5, max_hops: int = 2) -> Dict[str, Any]:
        """
        Process a query using graph-based RAG.
        
        Args:
            query: User query
            top_k: Number of top entities to retrieve
            max_hops: Maximum number of hops for graph expansion
            
        Returns:
            Dict with response and context
        """
        # Step 1: Retrieve relevant entities
        relevant_entities = self._retrieve_relevant_entities(query, top_k=top_k)
        
        if not relevant_entities:
            return {
                "answer": "I couldn't find relevant information in the knowledge graph to answer your query.",
                "context": None
            }
        
        # Step 2: Expand graph context
        entity_ids = [entity["entity_id"] for entity in relevant_entities]
        graph_context = self._expand_graph_context(entity_ids, max_distance=max_hops)
        
        # Step 3: Format context for LLM
        formatted_context = self._format_graph_context(graph_context)
        
        # Step 4: Generate response using LLM
        prompt_template = """
        You are an AI assistant that answers questions based on a knowledge graph.
        
        {context}
        
        User Query: {query}
        
        Using the knowledge graph context above, please provide a detailed and accurate answer.
        If the knowledge graph doesn't contain relevant information, state that you don't have enough information.
        
        Answer:
        """
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "query"]
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        response = chain.run(context=formatted_context, query=query)
        
        return {
            "answer": response,
            "context": {
                "relevant_entities": relevant_entities,
                "graph_context": graph_context
            }
        }


class GraphAugmentedRetriever:
    """Graph-augmented retriever for enhanced document retrieval."""
    
    def __init__(
        self,
        neo4j_db: Neo4jDatabase,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize the graph-augmented retriever.
        
        Args:
            neo4j_db: Neo4j database instance
            embedding_model: Sentence transformer model for embeddings
        """
        self.db = neo4j_db
        self.embedding_model = SentenceTransformer(embedding_model)
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """
        Get embedding vector for a query.
        
        Args:
            query: Query text
            
        Returns:
            Embedding vector
        """
        return self.embedding_model.encode(query).tolist()
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve documents using graph-augmented retrieval.
        
        Args:
            query: Query text
            top_k: Number of top documents to retrieve
            
        Returns:
            List of retrieved documents
        """
        # Step 1: Get query embedding
        query_embedding = self._get_query_embedding(query)
        
        # Step 2: Find relevant entities
        relevant_entities = self.db.semantic_search(query_embedding, top_k=top_k)
        
        if not relevant_entities:
            return []
        
        # Step 3: Use graph structure to find related documents
        # This is a simplified example - in a real system, you would have document nodes
        # linked to entity nodes and retrieve those documents
        
        # For this example, we'll just return the entities as "documents"
        return relevant_entities
