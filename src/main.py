"""
Main script to demonstrate the knowledge graph RAG system.
"""
import os
import sys
import argparse
from dotenv import load_dotenv
import pandas as pd

from src.data.ingestion import DocumentProcessor, KnowledgeGraphBuilder
from src.graph.database import Neo4jDatabase
from src.utils.embeddings import EmbeddingGenerator
from src.rag.query_engine import GraphRAGQueryEngine

# Load environment variables
load_dotenv()

def process_documents(input_dir: str, output_dir: str) -> None:
    """
    Process documents and build knowledge graph data.
    
    Args:
        input_dir: Directory containing input documents
        output_dir: Directory to save output files
    """
    print(f"Processing documents from {input_dir}...")
    
    # Initialize document processor
    processor = DocumentProcessor()
    
    # Process documents
    results = processor.process_directory(input_dir)
    
    # Build knowledge graph
    kg_builder = KnowledgeGraphBuilder()
    for result in results:
        kg_builder.add_processed_data(result)
    
    # Export to CSV
    entities_path, relationships_path = kg_builder.export_to_csv(output_dir)
    
    print(f"Exported entities to {entities_path}")
    print(f"Exported relationships to {relationships_path}")
    
    # Generate embeddings
    print("Generating embeddings...")
    embedding_generator = EmbeddingGenerator()
    entities_df = pd.read_csv(entities_path)
    entities_with_embeddings = embedding_generator.generate_entity_embeddings(entities_df)
    
    # Save embeddings
    embeddings_path = os.path.join(output_dir, 'entities_with_embeddings.csv')
    embedding_generator.save_embeddings(entities_with_embeddings, embeddings_path)
    
    return entities_path, relationships_path, embeddings_path

def import_to_neo4j(entities_path: str, relationships_path: str) -> None:
    """
    Import knowledge graph data to Neo4j.
    
    Args:
        entities_path: Path to entities CSV file
        relationships_path: Path to relationships CSV file
    """
    print("Importing knowledge graph to Neo4j...")
    
    # Initialize Neo4j database
    db = Neo4jDatabase()
    
    try:
        # Connect to database
        db.connect()
        
        # Create constraints
        db.create_constraints()
        
        # Clear existing data
        db.clear_database()
        
        # Import data
        db.import_knowledge_graph(entities_path, relationships_path)
        
        print("Knowledge graph imported successfully")
    finally:
        db.close()

def demo_query(query: str) -> None:
    """
    Demonstrate RAG query using the knowledge graph.
    
    Args:
        query: Query string
    """
    print(f"Processing query: {query}")
    
    # Initialize Neo4j database
    db = Neo4jDatabase()
    
    try:
        # Connect to database
        db.connect()
        
        # Initialize query engine
        query_engine = GraphRAGQueryEngine(db)
        
        # Process query
        result = query_engine.query(query)
        
        # Display result
        print("\nAnswer:")
        print(result["answer"])
        
        print("\nRelevant entities:")
        for entity in result["context"]["relevant_entities"]:
            print(f"- {entity['text']} ({entity['label']}): {entity['similarity']:.4f}")
        
    except Exception as e:
        print(f"Error processing query: {e}")
    finally:
        db.close()

def main():
    parser = argparse.ArgumentParser(description="Knowledge Graph RAG System")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Process documents command
    process_parser = subparsers.add_parser("process", help="Process documents and build knowledge graph")
    process_parser.add_argument("--input", required=True, help="Input directory with documents")
    process_parser.add_argument("--output", default="data", help="Output directory for knowledge graph data")
    
    # Import command
    import_parser = subparsers.add_parser("import", help="Import knowledge graph to Neo4j")
    import_parser.add_argument("--entities", required=True, help="Path to entities CSV file")
    import_parser.add_argument("--relationships", required=True, help="Path to relationships CSV file")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the knowledge graph")
    query_parser.add_argument("--query", required=True, help="Query string")
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command == "process":
        process_documents(args.input, args.output)
    elif args.command == "import":
        import_to_neo4j(args.entities, args.relationships)
    elif args.command == "query":
        demo_query(args.query)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
