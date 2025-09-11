#!/usr/bin/env python
"""
Command-line interface for the knowledge graph RAG system.
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
from src.utils.sample_data import generate_ai_documents, generate_structured_knowledge_graph
from src.utils.visualization import create_networkx_graph, visualize_graph

# Load environment variables
load_dotenv()

def process_command(args):
    """Process documents and build knowledge graph data."""
    print(f"Processing documents from {args.input_dir}...")
    
    # Initialize document processor
    processor = DocumentProcessor(spacy_model=args.spacy_model)
    
    # Process documents
    results = processor.process_directory(args.input_dir)
    
    # Build knowledge graph
    kg_builder = KnowledgeGraphBuilder()
    for result in results:
        kg_builder.add_processed_data(result)
    
    # Export to CSV
    entities_path, relationships_path = kg_builder.export_to_csv(args.output_dir)
    
    print(f"Exported entities to {entities_path}")
    print(f"Exported relationships to {relationships_path}")
    
    # Generate embeddings if requested
    if args.generate_embeddings:
        print("Generating embeddings...")
        embedding_generator = EmbeddingGenerator(model_name=args.embedding_model)
        entities_df = pd.read_csv(entities_path)
        entities_with_embeddings = embedding_generator.generate_entity_embeddings(entities_df)
        
        # Save embeddings
        embeddings_path = os.path.join(args.output_dir, 'entities_with_embeddings.csv')
        embedding_generator.save_embeddings(entities_with_embeddings, embeddings_path)
        print(f"Saved embeddings to {embeddings_path}")
    
    # Visualize if requested
    if args.visualize:
        print("Generating visualization...")
        entities_df = pd.read_csv(entities_path)
        relationships_df = pd.read_csv(relationships_path)
        
        G = create_networkx_graph(entities_df, relationships_df)
        viz_path = os.path.join(args.output_dir, 'knowledge_graph.html')
        visualize_graph(G, output_path=viz_path)

def import_command(args):
    """Import knowledge graph data to Neo4j."""
    print("Importing knowledge graph to Neo4j...")
    
    # Initialize Neo4j database
    db = Neo4jDatabase(
        uri=os.getenv("NEO4J_URI"),
        user=os.getenv("NEO4J_USER"),
        password=os.getenv("NEO4J_PASSWORD")
    )
    
    try:
        # Connect to database
        db.connect()
        
        # Create constraints
        db.create_constraints()
        
        # Clear existing data if requested
        if args.clear:
            db.clear_database()
        
        # Import data
        db.import_knowledge_graph(args.entities, args.relationships)
        
        print("Knowledge graph imported successfully")
    except Exception as e:
        print(f"Error importing to Neo4j: {e}")
    finally:
        db.close()

def query_command(args):
    """Query the knowledge graph."""
    print(f"Processing query: {args.query}")
    
    # Initialize Neo4j database
    db = Neo4jDatabase(
        uri=os.getenv("NEO4J_URI"),
        user=os.getenv("NEO4J_USER"),
        password=os.getenv("NEO4J_PASSWORD")
    )
    
    try:
        # Connect to database
        db.connect()
        
        # Initialize query engine
        query_engine = GraphRAGQueryEngine(
            db, 
            embedding_model=args.embedding_model,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Process query
        result = query_engine.query(args.query)
        
        # Display result
        print("\nAnswer:")
        print(result["answer"])
        
        print("\nRelevant entities:")
        for entity in result["context"]["relevant_entities"]:
            print(f"- {entity['text']} ({entity['label']}): {entity['similarity']:.4f}")
        
        # Visualize if requested
        if args.visualize:
            from src.utils.visualization import visualize_neo4j_results
            viz_path = "query_results.html"
            visualize_neo4j_results(result["context"]["graph_context"], output_path=viz_path)
            
    except Exception as e:
        print(f"Error processing query: {e}")
    finally:
        db.close()

def generate_command(args):
    """Generate sample data."""
    if args.structured:
        print("Generating structured knowledge graph data...")
        entities_path, relationships_path = generate_structured_knowledge_graph(args.output_dir)
        print(f"Generated structured knowledge graph data:")
        print(f"  - Entities: {entities_path}")
        print(f"  - Relationships: {relationships_path}")
        
        # Visualize if requested
        if args.visualize:
            print("Generating visualization...")
            entities_df = pd.read_csv(entities_path)
            relationships_df = pd.read_csv(relationships_path)
            
            G = create_networkx_graph(entities_df, relationships_df)
            viz_path = os.path.join(args.output_dir, 'sample_knowledge_graph.html')
            visualize_graph(G, output_path=viz_path)
    else:
        print(f"Generating {args.num_docs} sample documents...")
        file_paths = generate_ai_documents(args.output_dir, args.num_docs)
        print(f"Generated {len(file_paths)} sample documents in {args.output_dir}")

def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="Knowledge Graph RAG System")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Process documents command
    process_parser = subparsers.add_parser("process", help="Process documents and build knowledge graph")
    process_parser.add_argument("--input-dir", required=True, help="Input directory with documents")
    process_parser.add_argument("--output-dir", default="data/output", help="Output directory for knowledge graph data")
    process_parser.add_argument("--spacy-model", default="en_core_web_md", help="spaCy model to use")
    process_parser.add_argument("--generate-embeddings", action="store_true", help="Generate embeddings for entities")
    process_parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", help="Embedding model to use")
    process_parser.add_argument("--visualize", action="store_true", help="Generate visualization")
    process_parser.set_defaults(func=process_command)
    
    # Import command
    import_parser = subparsers.add_parser("import", help="Import knowledge graph to Neo4j")
    import_parser.add_argument("--entities", required=True, help="Path to entities CSV file")
    import_parser.add_argument("--relationships", required=True, help="Path to relationships CSV file")
    import_parser.add_argument("--clear", action="store_true", help="Clear existing data before import")
    import_parser.set_defaults(func=import_command)
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the knowledge graph")
    query_parser.add_argument("--query", required=True, help="Query string")
    query_parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", help="Embedding model to use")
    query_parser.add_argument("--visualize", action="store_true", help="Visualize query results")
    query_parser.set_defaults(func=query_command)
    
    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate sample data")
    generate_parser.add_argument("--output-dir", default="data/sample", help="Output directory for sample data")
    generate_parser.add_argument("--num-docs", type=int, default=10, help="Number of documents to generate")
    generate_parser.add_argument("--structured", action="store_true", help="Generate structured knowledge graph data")
    generate_parser.add_argument("--visualize", action="store_true", help="Generate visualization")
    generate_parser.set_defaults(func=generate_command)
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Execute command
    args.func(args)

if __name__ == "__main__":
    main()
