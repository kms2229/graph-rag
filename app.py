"""
Streamlit interface for the Knowledge Graph RAG system.
"""
import os
import sys
import tempfile
import pandas as pd
import streamlit as st
import networkx as nx
from pathlib import Path
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from streamlit_agraph import agraph, Node, Edge, Config

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

# Import project modules
from src.data.ingestion import DocumentProcessor, KnowledgeGraphBuilder
from src.data.llm_ingestion import LLMDocumentProcessor, LLMKnowledgeGraphBuilder
from src.graph.database import Neo4jDatabase
from src.rag.query_engine import GraphRAGQueryEngine
from src.utils.embeddings import EmbeddingGenerator

# Helper functions
def get_color_for_label(label):
    """Return a color based on entity label."""
    color_map = {
        "PERSON": "#FF6B6B",
        "ORGANIZATION": "#4ECDC4",
        "LOCATION": "#FFD166",
        "PRODUCT": "#6B5B95",
        "EVENT": "#88D8B0",
        "WORK_OF_ART": "#F6AE2D",
        "LAW": "#86BBD8",
        "DATE": "#C38D9E",
        "TIME": "#E8A87C",
        "PERCENT": "#85C1E9",
        "MONEY": "#76B041",
        "QUANTITY": "#D4A5A5",
        "TECHNOLOGY": "#45B69C",
        "CONCEPT": "#F172A1"
    }
    
    return color_map.get(label, "#CCCCCC")  # Default gray for unknown labels

# Set page configuration
st.set_page_config(
    page_title="Knowledge Graph RAG System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "neo4j_connected" not in st.session_state:
    st.session_state.neo4j_connected = False
    
if "db" not in st.session_state:
    st.session_state.db = None
    
if "query_engine" not in st.session_state:
    st.session_state.query_engine = None
    
if "kg_builder" not in st.session_state:
    st.session_state.kg_builder = None
    
if "entities_df" not in st.session_state:
    st.session_state.entities_df = None
    
if "relationships_df" not in st.session_state:
    st.session_state.relationships_df = None
    
if "graph" not in st.session_state:
    st.session_state.graph = None
    
# Attempt to connect to Neo4j automatically on startup
neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
neo4j_user = os.getenv("NEO4J_USER", "neo4j")
neo4j_password = os.getenv("NEO4J_PASSWORD", "")

# Only try to connect if we have credentials
if neo4j_uri and neo4j_user and neo4j_password and not st.session_state.neo4j_connected:
    try:
        st.session_state.db = Neo4jDatabase(
            uri=neo4j_uri,
            user=neo4j_user,
            password=neo4j_password
        )
        st.session_state.db.connect()
        st.session_state.neo4j_connected = True
        
        # Initialize query engine if OpenAI API key is available
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            st.session_state.query_engine = GraphRAGQueryEngine(
                st.session_state.db,
                embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
                openai_api_key=openai_api_key
            )
    except Exception:
        # Silently fail - we'll show connection status in the sidebar
        pass

# Sidebar for configuration
with st.sidebar:
    st.title("🧠 Knowledge Graph RAG")
    
    # Navigation
    st.header("Navigation")
    page = st.radio("Go to", ["Home", "Data Ingestion", "Knowledge Graph", "Query Engine", "Settings"])
    
    # Database connection
    st.header("Database Connection")
    
    # Show connection status
    if st.session_state.neo4j_connected:
        st.success("✅ Connected to Neo4j")
        if st.button("Disconnect"):
            st.session_state.neo4j_connected = False
            st.session_state.db = None
            st.session_state.query_engine = None
            st.rerun()
    else:
        st.warning("❌ Not connected to Neo4j")
        
        # Connection form
        neo4j_uri = st.text_input("Neo4j URI", value=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
        neo4j_user = st.text_input("Neo4j User", value=os.getenv("NEO4J_USER", "neo4j"))
        neo4j_password = st.text_input("Neo4j Password", value=os.getenv("NEO4J_PASSWORD", ""), type="password")
        
        if st.button("Connect to Neo4j"):
            try:
                st.session_state.db = Neo4jDatabase(
                    uri=neo4j_uri,
                    user=neo4j_user,
                    password=neo4j_password
                )
                st.session_state.db.connect()
                st.session_state.neo4j_connected = True
                
                # Initialize query engine if OpenAI API key is available
                openai_api_key = os.getenv("OPENAI_API_KEY")
                if openai_api_key:
                    st.session_state.query_engine = GraphRAGQueryEngine(
                        st.session_state.db,
                        embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
                        openai_api_key=openai_api_key
                    )
                st.rerun()
            except Exception as e:
                st.error(f"Error connecting to Neo4j: {e}")
    
    # OpenAI API key
    st.header("OpenAI API Key")
    openai_api_key = st.text_input("OpenAI API Key", value=os.getenv("OPENAI_API_KEY", ""), type="password")
    if st.button("Save API Key"):
        os.environ["OPENAI_API_KEY"] = openai_api_key
        st.success("API key saved!")
        
        # Initialize query engine if connected to Neo4j
        if st.session_state.neo4j_connected:
            st.session_state.query_engine = GraphRAGQueryEngine(
                st.session_state.db,
                openai_api_key=openai_api_key
            )
            st.success("Query engine initialized!")

# Home page
if page == "Home":
    st.title("Knowledge Graph RAG System")
    st.write("""
    Welcome to the Knowledge Graph RAG (Retrieval-Augmented Generation) System!
    
    This application allows you to:
    1. **Ingest data** from text documents to build a knowledge graph
    2. **Visualize the knowledge graph** and explore relationships
    3. **Query the knowledge graph** using natural language
    
    Use the sidebar to navigate between different sections of the application.
    """)
    
    # System status
    st.header("System Status")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Neo4j Connection",
            value="Connected" if st.session_state.neo4j_connected else "Disconnected"
        )
    
    with col2:
        openai_status = "Available" if os.getenv("OPENAI_API_KEY") else "Not configured"
        st.metric(label="OpenAI API", value=openai_status)
    
    with col3:
        query_engine_status = "Ready" if st.session_state.query_engine else "Not initialized"
        st.metric(label="Query Engine", value=query_engine_status)
    
    # Quick start guide
    st.header("Quick Start Guide")
    st.write("""
    1. Connect to Neo4j using the sidebar
    2. Go to Data Ingestion to process documents
    3. Explore the Knowledge Graph visualization
    4. Use the Query Engine to ask questions
    """)

# Data Ingestion page
elif page == "Data Ingestion":
    st.title("Data Ingestion")
    st.write("Process documents to extract entities and relationships for the knowledge graph.")
    
    # Choose ingestion method
    ingestion_method = st.radio(
        "Select ingestion method",
        ["Traditional NLP (spaCy)", "LLM-based (OpenAI/BERT)"]
    )
    
    # File upload
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload text documents",
        accept_multiple_files=True,
        type=["txt", "md", "csv"]
    )
    
    # Text input
    st.header("Or enter text directly")
    text_input = st.text_area("Enter text to process", height=200)
    
    # Process button
    if st.button("Process Data"):
        if not uploaded_files and not text_input:
            st.error("Please upload files or enter text to process.")
        else:
            # Create temporary directory for uploaded files
            temp_dir = tempfile.mkdtemp()
            file_paths = []
            
            # Save uploaded files to temp directory
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    file_paths.append(file_path)
            
            # Save text input to temp file if provided
            if text_input:
                text_file_path = os.path.join(temp_dir, "input_text.txt")
                with open(text_file_path, "w") as f:
                    f.write(text_input)
                file_paths.append(text_file_path)
            
            # Process documents based on selected method
            with st.spinner("Processing documents..."):
                if ingestion_method == "Traditional NLP (spaCy)":
                    processor = DocumentProcessor()
                    kg_builder = KnowledgeGraphBuilder()
                else:  # LLM-based
                    # Default to BERT model
                    processor = LLMDocumentProcessor(model_type="bert")
                    kg_builder = LLMKnowledgeGraphBuilder()
                
                # Process each file
                for file_path in file_paths:
                    try:
                        processed_data = processor.process_file(file_path)
                        kg_builder.add_processed_data(processed_data)
                    except Exception as e:
                        st.error(f"Error processing {os.path.basename(file_path)}: {e}")
                
                # Get entities and relationships
                entities_df = kg_builder.get_unique_entities()
                relationships_df = kg_builder.get_relationships_df()
                
                # Store in session state
                st.session_state.kg_builder = kg_builder
                st.session_state.entities_df = entities_df
                st.session_state.relationships_df = relationships_df
                
                # Create graph for visualization
                G = nx.DiGraph()
                
                # Add nodes
                for _, row in entities_df.iterrows():
                    G.add_node(row['text'], label=row['label'])
                
                # Add edges
                for _, row in relationships_df.iterrows():
                    G.add_edge(row['source'], row['target'], label=row['relation'])
                
                st.session_state.graph = G
                
                st.success(f"Processed {len(file_paths)} documents. Extracted {len(entities_df)} entities and {len(relationships_df)} relationships.")
    
    # Export options
    if st.session_state.kg_builder:
        st.header("Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export to CSV"):
                # Create output directory
                output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "output")
                os.makedirs(output_dir, exist_ok=True)
                
                # Export to CSV
                entities_path, relationships_path = st.session_state.kg_builder.export_to_csv(output_dir)
                
                st.success(f"Exported to:\n- {entities_path}\n- {relationships_path}")
        
        with col2:
            if st.button("Import to Neo4j"):
                if not st.session_state.neo4j_connected:
                    st.error("Please connect to Neo4j first.")
                else:
                    # Create output directory
                    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "output")
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Export to CSV
                    entities_path, relationships_path = st.session_state.kg_builder.export_to_csv(output_dir)
                    
                    # Import to Neo4j
                    with st.spinner("Importing to Neo4j..."):
                        try:
                            st.session_state.db.create_constraints()
                            st.session_state.db.import_knowledge_graph(entities_path, relationships_path)
                            st.success("Knowledge graph imported to Neo4j successfully!")
                        except Exception as e:
                            st.error(f"Error importing to Neo4j: {e}")

# Knowledge Graph page
elif page == "Knowledge Graph":
    st.title("Knowledge Graph Visualization")
    
    if st.session_state.graph:
        # Display graph statistics
        st.header("Graph Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Entities", len(st.session_state.graph.nodes))
        
        with col2:
            st.metric("Relationships", len(st.session_state.graph.edges))
        
        with col3:
            # Count entity types
            if st.session_state.entities_df is not None:
                entity_types = st.session_state.entities_df['label'].nunique()
                st.metric("Entity Types", entity_types)
        
        # Display interactive graph
        st.header("Interactive Graph")
        
        # Convert NetworkX graph to agraph format
        nodes = []
        edges = []
        
        # Add nodes
        for node, attrs in st.session_state.graph.nodes(data=True):
            label = attrs.get('label', 'ENTITY')
            nodes.append(Node(id=node, label=node, size=25, color=get_color_for_label(label)))
        
        # Add edges
        for source, target, attrs in st.session_state.graph.edges(data=True):
            relation = attrs.get('label', 'related_to')
            edges.append(Edge(source=source, target=target, label=relation))
        
        # Configure graph
        config = Config(
            width=800,
            height=600,
            directed=True,
            physics=True,
            hierarchical=False
        )
        
        # Display graph
        agraph(nodes=nodes, edges=edges, config=config)
        
        # Display data tables
        st.header("Data Tables")
        
        tab1, tab2 = st.tabs(["Entities", "Relationships"])
        
        with tab1:
            st.dataframe(st.session_state.entities_df)
        
        with tab2:
            st.dataframe(st.session_state.relationships_df)
    
    elif st.session_state.neo4j_connected:
        st.info("No graph data in current session. Fetching from Neo4j...")
        
        try:
            # Query Neo4j for entities and relationships
            with st.spinner("Fetching data from Neo4j..."):
                # Get entities
                entities_query = "MATCH (n) RETURN n.text as text, labels(n)[0] as label"
                entities_df = st.session_state.db.run_query(entities_query)
                
                # Get relationships
                relationships_query = """
                MATCH (s)-[r]->(t)
                RETURN s.text as source, t.text as target, type(r) as relation
                """
                relationships_df = st.session_state.db.run_query(relationships_query)
                
                # Create graph for visualization
                G = nx.DiGraph()
                
                # Add nodes
                for _, row in entities_df.iterrows():
                    G.add_node(row['text'], label=row['label'])
                
                # Add edges
                for _, row in relationships_df.iterrows():
                    G.add_edge(row['source'], row['target'], label=row['relation'])
                
                # Store in session state
                st.session_state.entities_df = entities_df
                st.session_state.relationships_df = relationships_df
                st.session_state.graph = G
                
                st.success(f"Loaded {len(entities_df)} entities and {len(relationships_df)} relationships from Neo4j.")
                st.rerun()
        except Exception as e:
            st.error(f"Error fetching data from Neo4j: {e}")
    
    else:
        st.warning("No graph data available. Please process documents in the Data Ingestion page or connect to Neo4j.")

# Query Engine page
elif page == "Query Engine":
    st.title("Query Engine")
    st.write("Ask questions about your knowledge graph using natural language.")
    
    # Check if query engine is initialized
    if not st.session_state.query_engine:
        if not st.session_state.neo4j_connected:
            st.error("Please connect to Neo4j first.")
        elif not os.getenv("OPENAI_API_KEY"):
            st.error("Please configure your OpenAI API key in the Settings page.")
        else:
            try:
                st.session_state.query_engine = GraphRAGQueryEngine(
                    st.session_state.db,
                    openai_api_key=os.getenv("OPENAI_API_KEY")
                )
                st.success("Query engine initialized!")
            except Exception as e:
                st.error(f"Error initializing query engine: {e}")
    
    # Query input
    query = st.text_input("Enter your question")
    
    # Advanced options
    with st.expander("Advanced Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            top_k = st.slider("Number of top entities to retrieve", 1, 10, 3)
        
        with col2:
            max_hops = st.slider("Maximum hops for graph expansion", 1, 3, 2)
    
    # Process query
    if query and st.button("Submit Query"):
        if not st.session_state.query_engine:
            st.error("Query engine not initialized. Please check your Neo4j connection and OpenAI API key.")
        else:
            with st.spinner("Processing query..."):
                try:
                    result = st.session_state.query_engine.query(
                        query,
                        top_k=top_k,
                        max_hops=max_hops
                    )
                    
                    # Display answer
                    st.header("Answer")
                    st.write(result["answer"])
                    
                    # Display retrieved context
                    st.header("Retrieved Context")
                    st.json(result["context"])
                    
                    # Display graph context
                    if "graph_context" in result:
                        st.header("Graph Context")
                        
                        # Convert to NetworkX graph
                        G = nx.DiGraph()
                        
                        # Add nodes and edges from graph context
                        for entity in result["graph_context"]["entities"]:
                            G.add_node(entity["text"], label=entity["label"])
                        
                        for rel in result["graph_context"]["relationships"]:
                            G.add_edge(rel["source"], rel["target"], label=rel["relation"])
                        
                        # Convert to agraph format
                        nodes = []
                        edges = []
                        
                        for node, attrs in G.nodes(data=True):
                            label = attrs.get('label', 'ENTITY')
                            nodes.append(Node(id=node, label=node, size=25, color=get_color_for_label(label)))
                        
                        for source, target, attrs in G.edges(data=True):
                            relation = attrs.get('label', 'related_to')
                            edges.append(Edge(source=source, target=target, label=relation))
                        
                        # Configure graph
                        config = Config(
                            width=800,
                            height=400,
                            directed=True,
                            physics=True,
                            hierarchical=False
                        )
                        
                        # Display graph
                        agraph(nodes=nodes, edges=edges, config=config)
                
                except Exception as e:
                    st.error(f"Error processing query: {e}")

# Settings page
elif page == "Settings":
    st.title("Settings")
    
    # Neo4j settings
    st.header("Neo4j Database")
    
    if st.button("Clear Database"):
        if not st.session_state.neo4j_connected:
            st.error("Please connect to Neo4j first.")
        else:
            with st.spinner("Clearing database..."):
                try:
                    st.session_state.db.clear_database()
                    st.success("Database cleared successfully!")
                except Exception as e:
                    st.error(f"Error clearing database: {e}")
    
    # Model settings
    st.header("Model Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        embedding_model = st.selectbox(
            "Embedding Model",
            ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "paraphrase-multilingual-MiniLM-L12-v2"]
        )
        
    with col2:
        llm_model = st.selectbox(
            "LLM Model for Entity Extraction",
            ["bert", "openai"],
            index=0,
            help="Select the LLM model to use for entity extraction"
        )
    
    if st.button("Save Model Settings"):
        # Update environment variables
        os.environ["EMBEDDING_MODEL"] = embedding_model
        os.environ["LLM_MODEL"] = llm_model
        
        st.success("Model settings saved!")
        
        # Reinitialize query engine if connected
        if st.session_state.neo4j_connected and os.getenv("OPENAI_API_KEY"):
            st.session_state.query_engine = GraphRAGQueryEngine(
                st.session_state.db,
                embedding_model=embedding_model,
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
            st.success("Query engine reinitialized with new settings!")

# Helper functions
def get_color_for_label(label):
    """Return a color based on entity label."""
    color_map = {
        "PERSON": "#FF6B6B",
        "ORGANIZATION": "#4ECDC4",
        "LOCATION": "#FFD166",
        "PRODUCT": "#6B5B95",
        "EVENT": "#88D8B0",
        "WORK_OF_ART": "#F6AE2D",
        "LAW": "#86BBD8",
        "DATE": "#C38D9E",
        "TIME": "#E8A87C",
        "PERCENT": "#85C1E9",
        "MONEY": "#76B041",
        "QUANTITY": "#D4A5A5",
        "TECHNOLOGY": "#45B69C",
        "CONCEPT": "#F172A1"
    }
    
    return color_map.get(label, "#CCCCCC")  # Default gray for unknown labels
