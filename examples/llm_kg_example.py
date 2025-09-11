#!/usr/bin/env python
"""
Example script demonstrating the LLM-based Knowledge Graph Builder.
"""
import os
import sys
import pandas as pd
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from pyvis.network import Network
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# Import project modules
from src.data.llm_ingestion import LLMDocumentProcessor, LLMKnowledgeGraphBuilder
from src.graph.database import Neo4jDatabase
from src.utils.visualization import create_networkx_graph, visualize_graph

def main():
    """Main function to demonstrate LLM-based knowledge graph construction."""
    # Create sample directory if it doesn't exist
    sample_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sample")
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "output")
    
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if sample files exist, if not create them
    sample_file = os.path.join(sample_dir, "ai_sample.txt")
    if not os.path.exists(sample_file):
        print("Creating sample document...")
        with open(sample_file, 'w') as f:
            f.write("""
            OpenAI released GPT-4 in March 2023, which demonstrated significant improvements over previous models.
            Microsoft has integrated GPT-4 into various products including Bing Chat and GitHub Copilot.
            Google responded with their own advanced model called Gemini, developed by Google DeepMind.
            These large language models are trained on vast amounts of text data from the internet.
            Researchers at universities like Stanford and MIT are studying the capabilities and limitations of these AI systems.
            """)
    
    # Choose model type based on available API keys
    model_type = "openai" if os.getenv("OPENAI_API_KEY") else "bert"
    print(f"Using model type: {model_type}")
    
    # Initialize LLM document processor
    processor = LLMDocumentProcessor(model_type=model_type)
    
    # Process documents
    print("Processing documents...")
    results = processor.process_directory(sample_dir)
    
    # Build knowledge graph
    print("Building knowledge graph...")
    kg_builder = LLMKnowledgeGraphBuilder()
    for result in results:
        kg_builder.add_processed_data(result)
    
    # Get entities and relationships
    entities_df = kg_builder.get_unique_entities()
    relationships_df = kg_builder.get_relationships_df()
    
    print(f"Extracted {len(entities_df)} unique entities and {len(relationships_df)} relationships")
    
    # Display sample entities
    print("\nSample entities:")
    print(entities_df.head())
    
    # Display sample relationships
    print("\nSample relationships:")
    print(relationships_df.head())
    
    # Export to CSV
    entities_path, relationships_path = kg_builder.export_to_csv(output_dir)
    print(f"Exported entities to {entities_path}")
    print(f"Exported relationships to {relationships_path}")
    
    # Generate embeddings
    print("Generating embeddings...")
    entities_with_embeddings = kg_builder.enrich_entities_with_embeddings()
    
    # Save embeddings
    embeddings_path = os.path.join(output_dir, 'llm_entities_with_embeddings.csv')
    entities_with_embeddings.to_csv(embeddings_path, index=False)
    print(f"Saved embeddings to {embeddings_path}")
    
    # Visualize the knowledge graph
    print("Generating visualization...")
    try:
        G = create_networkx_graph(entities_df, relationships_df)
        viz_path = os.path.join(output_dir, 'llm_knowledge_graph.html')
        visualize_graph(G, output_path=viz_path)
        print(f"Visualization saved to {viz_path}")
    except Exception as e:
        print(f"Error generating visualization: {e}")
    
    # Import to Neo4j if connection details are available
    if all([os.getenv("NEO4J_URI"), os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")]):
        print("Importing to Neo4j...")
        try:
            db = Neo4jDatabase()
            db.connect()
            db.create_constraints()
            
            # Clear existing data
            db.clear_database()
            
            # Import data
            db.import_knowledge_graph(entities_path, relationships_path)
            print("Knowledge graph imported to Neo4j successfully")
            db.close()
        except Exception as e:
            print(f"Error importing to Neo4j: {e}")
    else:
        print("Neo4j connection details not found in environment variables. Skipping import.")
    
    print("\nLLM-based knowledge graph construction complete!")

if __name__ == "__main__":
    main()
