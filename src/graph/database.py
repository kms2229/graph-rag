"""
Graph database connection and operations for knowledge graph.
"""
import os
from typing import Dict, List, Any, Optional, Union
import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Neo4jDatabase:
    """Neo4j graph database connection and operations."""
    
    def __init__(self, uri: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None, database: Optional[str] = None):
        """
        Initialize the Neo4j database connection.
        
        Args:
            uri: Neo4j URI (defaults to environment variable)
            user: Neo4j username (defaults to environment variable)
            password: Neo4j password (defaults to environment variable)
            database: Neo4j database name (for Aura instances)
        """
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")
        self.database = database or os.getenv("NEO4J_DATABASE", None)
        
        self.driver = None
        
    def connect(self) -> None:
        """Establish connection to Neo4j database."""
        try:
            # Create driver with database parameter if specified
            if self.database:
                self.driver = GraphDatabase.driver(
                    self.uri, 
                    auth=(self.user, self.password),
                    database=self.database
                )
            else:
                self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
                
            # Test connection
            session_params = {}
            if self.database:
                session_params['database'] = self.database
                
            with self.driver.session(**session_params) as session:
                session.run("RETURN 1")
            print("Successfully connected to Neo4j database")
        except Exception as e:
            print(f"Failed to connect to Neo4j: {e}")
            raise
    
    def close(self) -> None:
        """Close the database connection."""
        if self.driver:
            self.driver.close()
            self.driver = None
    
    def create_constraints(self) -> None:
        """Create necessary constraints for the knowledge graph."""
        if not self.driver:
            self.connect()
        
        session_params = {}
        if self.database:
            session_params['database'] = self.database
            
        with self.driver.session(**session_params) as session:
            # Create constraint on Entity nodes
            session.run("""
                CREATE CONSTRAINT entity_id IF NOT EXISTS
                FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE
            """)
            
            print("Created constraints for knowledge graph")
    
    def clear_database(self) -> None:
        """Clear all nodes and relationships in the database."""
        if not self.driver:
            self.connect()
        
        session_params = {}
        if self.database:
            session_params['database'] = self.database
            
        with self.driver.session(**session_params) as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("Database cleared")
    
    def import_entities(self, entities_df: pd.DataFrame) -> None:
        """
        Import entities into the graph database.
        
        Args:
            entities_df: DataFrame with entity data
        """
        if not self.driver:
            self.connect()
        
        # Convert DataFrame to list of dictionaries
        entities = entities_df.to_dict('records')
        
        with self.driver.session() as session:
            # Use batching for better performance
            batch_size = 500
            for i in range(0, len(entities), batch_size):
                batch = entities[i:i+batch_size]
                
                # Use UNWIND for batch import
                session.run("""
                    UNWIND $batch AS entity
                    MERGE (e:Entity {entity_id: entity.entity_id})
                    SET e.text = entity.text,
                        e.label = entity.label
                    RETURN count(*)
                """, {"batch": batch})
            
            print(f"Imported {len(entities)} entities")
    
    def import_relationships(self, relationships_df: pd.DataFrame) -> None:
        """
        Import relationships into the graph database.
        
        Args:
            relationships_df: DataFrame with relationship data
        """
        if not self.driver:
            self.connect()
        
        # Convert DataFrame to list of dictionaries
        relationships = relationships_df.to_dict('records')
        
        with self.driver.session() as session:
            # Use batching for better performance
            batch_size = 500
            for i in range(0, len(relationships), batch_size):
                batch = relationships[i:i+batch_size]
                
                # Use UNWIND for batch import
                session.run("""
                    UNWIND $batch AS rel
                    MATCH (source:Entity {entity_id: toLower(rel.source)})
                    MATCH (target:Entity {entity_id: toLower(rel.target)})
                    MERGE (source)-[r:RELATED_TO {type: replace(toUpper(rel.relation), ' ', '_')}]->(target)
                    RETURN count(*)
                """, {"batch": batch})
            
            print(f"Imported {len(relationships)} relationships")
    
    def import_knowledge_graph(self, entities_path: str, relationships_path: str) -> None:
        """
        Import knowledge graph data from CSV files.
        
        Args:
            entities_path: Path to entities CSV file
            relationships_path: Path to relationships CSV file
        """
        entities_df = pd.read_csv(entities_path)
        relationships_df = pd.read_csv(relationships_path)
        
        # Ensure entity_id is lowercase for consistent matching
        entities_df['entity_id'] = entities_df['entity_id'].str.lower()
        relationships_df['source'] = relationships_df['source'].str.lower()
        relationships_df['target'] = relationships_df['target'].str.lower()
        
        self.import_entities(entities_df)
        self.import_relationships(relationships_df)
    
    def query(self, cypher_query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query and return results.
        
        Args:
            cypher_query: Cypher query string
            params: Query parameters
            
        Returns:
            List of result records as dictionaries
        """
        if not self.driver:
            self.connect()
        
        with self.driver.session() as session:
            result = session.run(cypher_query, params or {})
            return [record.data() for record in result]
    
    def get_entity_neighbors(self, entity_id: str, max_distance: int = 2) -> Dict[str, Any]:
        """
        Get neighboring entities and relationships for a given entity.
        
        Args:
            entity_id: ID of the entity to find neighbors for
            max_distance: Maximum path length to traverse
            
        Returns:
            Dict with nodes and relationships
        """
        if not self.driver:
            self.connect()
        
        cypher_query = f"""
        MATCH path = (e:Entity {{entity_id: $entity_id}})-[*1..{max_distance}]-(neighbor)
        RETURN path
        LIMIT 100
        """
        
        with self.driver.session() as session:
            result = session.run(cypher_query, {"entity_id": entity_id.lower()})
            
            # Process results into a format suitable for visualization
            nodes = set()
            relationships = []
            
            for record in result:
                path = record["path"]
                
                for node in path.nodes:
                    node_data = dict(node.items())
                    node_data["id"] = node.id
                    node_data["labels"] = list(node.labels)
                    nodes.add(str(node_data))  # Convert to string for set deduplication
                
                for rel in path.relationships:
                    relationships.append({
                        "source": rel.start_node.id,
                        "target": rel.end_node.id,
                        "type": rel.type
                    })
            
            # Convert nodes back to dictionaries
            nodes_list = [eval(node_str) for node_str in nodes]
            
            return {
                "nodes": nodes_list,
                "relationships": relationships
            }
    
    def semantic_search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform semantic search using vector similarity.
        
        Args:
            query_embedding: Vector embedding of the query
            top_k: Number of top results to return
            
        Returns:
            List of matching entities with similarity scores
        """
        if not self.driver:
            self.connect()
        
        # This assumes entities have embedding vectors stored
        cypher_query = """
        MATCH (e:Entity)
        WHERE e.embedding IS NOT NULL
        WITH e, gds.similarity.cosine(e.embedding, $query_embedding) AS similarity
        ORDER BY similarity DESC
        LIMIT $top_k
        RETURN e.entity_id AS entity_id, e.text AS text, e.label AS label, similarity
        """
        
        with self.driver.session() as session:
            result = session.run(cypher_query, {
                "query_embedding": query_embedding,
                "top_k": top_k
            })
            
            return [record.data() for record in result]
