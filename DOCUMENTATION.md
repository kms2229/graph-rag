# Knowledge Graph RAG System Documentation

This document provides comprehensive documentation for the Knowledge Graph RAG (Retrieval-Augmented Generation) system. The system combines knowledge graphs with language models to provide more context-aware and accurate responses to queries.

## Table of Contents

1. [System Overview](#system-overview)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Usage](#usage)
   - [Command-Line Interface](#command-line-interface)
   - [Python API](#python-api)
5. [System Components](#system-components)
   - [Data Ingestion](#data-ingestion)
   - [Knowledge Graph Construction](#knowledge-graph-construction)
   - [Graph Database](#graph-database)
   - [RAG Query Engine](#rag-query-engine)
6. [Examples](#examples)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Usage](#advanced-usage)

## System Overview

The Knowledge Graph RAG system enhances traditional RAG approaches by representing information as a graph rather than isolated chunks. This allows the system to capture relationships between entities and provide more context-aware responses.

Key features:
- Knowledge graph construction from text documents
- Entity and relationship extraction using NLP
- Graph-based retrieval for RAG queries
- Neo4j integration for graph storage and querying
- Visualization tools for knowledge graph exploration

## Installation

### Prerequisites

- Python 3.8+
- Neo4j Database (local or cloud)

### Steps

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd graph_rag
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download required spaCy model:
   ```bash
   python -m spacy download en_core_web_md
   ```

4. Set up Neo4j:
   - Install Neo4j Desktop or use Neo4j Aura cloud
   - Create a new database
   - Set environment variables (see [Configuration](#configuration))

## Configuration

Create a `.env` file in the project root with the following variables:

```
# Neo4j Database Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password_here

# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Embedding Model Configuration
# EMBEDDING_MODEL=all-MiniLM-L6-v2

# Optional: Data Directories
# DATA_DIR=./data
# OUTPUT_DIR=./data/output
```

## Usage

### Command-Line Interface

The system provides a command-line interface (CLI) for easy interaction:

#### Process Documents

Process text documents to extract entities and relationships:

```bash
./cli.py process --input-dir data/sample --output-dir data/output --generate-embeddings --visualize
```

Options:
- `--input-dir`: Directory containing input documents
- `--output-dir`: Directory to save output files
- `--spacy-model`: spaCy model to use (default: en_core_web_md)
- `--generate-embeddings`: Generate embeddings for entities
- `--embedding-model`: Embedding model to use (default: all-MiniLM-L6-v2)
- `--visualize`: Generate visualization

#### Import to Neo4j

Import knowledge graph data to Neo4j:

```bash
./cli.py import --entities data/output/entities.csv --relationships data/output/relationships.csv --clear
```

Options:
- `--entities`: Path to entities CSV file
- `--relationships`: Path to relationships CSV file
- `--clear`: Clear existing data before import

#### Query the Knowledge Graph

Query the knowledge graph:

```bash
./cli.py query --query "What is the relationship between OpenAI and GPT-4?" --visualize
```

Options:
- `--query`: Query string
- `--embedding-model`: Embedding model to use (default: all-MiniLM-L6-v2)
- `--visualize`: Visualize query results

#### Generate Sample Data

Generate sample data for testing:

```bash
./cli.py generate --output-dir data/sample --num-docs 10
```

Options:
- `--output-dir`: Output directory for sample data
- `--num-docs`: Number of documents to generate
- `--structured`: Generate structured knowledge graph data
- `--visualize`: Generate visualization

### Python API

You can also use the system programmatically:

```python
from src.data.ingestion import DocumentProcessor, KnowledgeGraphBuilder
from src.graph.database import Neo4jDatabase
from src.rag.query_engine import GraphRAGQueryEngine

# Process documents
processor = DocumentProcessor()
results = processor.process_directory('data/sample')

# Build knowledge graph
kg_builder = KnowledgeGraphBuilder()
for result in results:
    kg_builder.add_processed_data(result)

# Export to CSV
entities_path, relationships_path = kg_builder.export_to_csv('data/output')

# Import to Neo4j
db = Neo4jDatabase()
db.connect()
db.create_constraints()
db.import_knowledge_graph(entities_path, relationships_path)

# Query the knowledge graph
query_engine = GraphRAGQueryEngine(db)
result = query_engine.query("What is the relationship between OpenAI and GPT-4?")
print(result["answer"])
```

## System Components

### Data Ingestion

The data ingestion module processes text documents to extract entities and relationships:

- `DocumentProcessor`: Processes text documents using spaCy for NLP tasks
- Methods:
  - `process_text(text)`: Process a text string
  - `process_file(file_path)`: Process a text file
  - `process_directory(dir_path)`: Process all text files in a directory

### Knowledge Graph Construction

The knowledge graph builder constructs a knowledge graph from processed data:

- `KnowledgeGraphBuilder`: Builds knowledge graph data from processed documents
- Methods:
  - `add_processed_data(processed_data)`: Add processed document data
  - `get_unique_entities()`: Get unique entities with their types
  - `get_relationships_df()`: Get relationships as a DataFrame
  - `export_to_csv(output_dir)`: Export entities and relationships to CSV files

### Graph Database

The graph database module handles interaction with Neo4j:

- `Neo4jDatabase`: Neo4j graph database connection and operations
- Methods:
  - `connect()`: Establish connection to Neo4j database
  - `create_constraints()`: Create necessary constraints
  - `clear_database()`: Clear all nodes and relationships
  - `import_entities(entities_df)`: Import entities
  - `import_relationships(relationships_df)`: Import relationships
  - `import_knowledge_graph(entities_path, relationships_path)`: Import knowledge graph data
  - `query(cypher_query, params)`: Execute a Cypher query
  - `get_entity_neighbors(entity_id, max_distance)`: Get neighboring entities and relationships

### RAG Query Engine

The RAG query engine uses the knowledge graph for enhanced retrieval:

- `GraphRAGQueryEngine`: Query engine for graph-based RAG
- Methods:
  - `query(query)`: Process a query using graph-based RAG

- `GraphAugmentedRetriever`: Graph-augmented retriever for enhanced document retrieval
- Methods:
  - `retrieve(query, top_k)`: Retrieve documents using graph-augmented retrieval

## Examples

### Example 1: Building a Knowledge Graph from Documents

```python
from src.data.ingestion import DocumentProcessor, KnowledgeGraphBuilder

# Initialize document processor
processor = DocumentProcessor()

# Process documents
results = processor.process_directory('data/sample')

# Build knowledge graph
kg_builder = KnowledgeGraphBuilder()
for result in results:
    kg_builder.add_processed_data(result)

# Export to CSV
entities_path, relationships_path = kg_builder.export_to_csv('data/output')
```

### Example 2: Visualizing the Knowledge Graph

```python
import pandas as pd
from src.utils.visualization import create_networkx_graph, visualize_graph

# Load entities and relationships
entities_df = pd.read_csv('data/output/entities.csv')
relationships_df = pd.read_csv('data/output/relationships.csv')

# Create and visualize graph
G = create_networkx_graph(entities_df, relationships_df)
visualize_graph(G, output_path='knowledge_graph.html')
```

### Example 3: Querying the Knowledge Graph

```python
from src.graph.database import Neo4jDatabase
from src.rag.query_engine import GraphRAGQueryEngine

# Initialize Neo4j database and query engine
db = Neo4jDatabase()
db.connect()

# Initialize query engine
query_engine = GraphRAGQueryEngine(db)

# Process query
result = query_engine.query("What is the relationship between OpenAI and GPT-4?")
print(result["answer"])
```

## Troubleshooting

### Common Issues

1. **Neo4j Connection Issues**
   - Ensure Neo4j is running
   - Check connection details in `.env` file
   - Verify network connectivity

2. **Missing Dependencies**
   - Run `pip install -r requirements.txt`
   - Check for error messages during installation

3. **spaCy Model Issues**
   - Run `python -m spacy download en_core_web_md`
   - Try using a different model (e.g., en_core_web_sm)

4. **OpenAI API Issues**
   - Verify API key in `.env` file
   - Check API usage limits

## Advanced Usage

### Custom Entity Extraction

You can extend the `DocumentProcessor` class to add custom entity extraction:

```python
from src.data.ingestion import DocumentProcessor

class CustomDocumentProcessor(DocumentProcessor):
    def __init__(self, spacy_model="en_core_web_md"):
        super().__init__(spacy_model)
        
        # Add custom components
        self.nlp.add_pipe("custom_component", last=True)
    
    def process_text(self, text):
        result = super().process_text(text)
        
        # Add custom processing
        # ...
        
        return result
```

### Custom Graph Queries

You can execute custom Cypher queries against the Neo4j database:

```python
from src.graph.database import Neo4jDatabase

db = Neo4jDatabase()
db.connect()

# Execute custom query
results = db.query("""
    MATCH (e1:Entity)-[r]->(e2:Entity)
    WHERE e1.label = 'ORGANIZATION' AND e2.label = 'TECHNOLOGY'
    RETURN e1.text AS org, type(r) AS relation, e2.text AS tech
    LIMIT 10
""")

for result in results:
    print(f"{result['org']} {result['relation']} {result['tech']}")
```

### Embedding Customization

You can customize the embedding generation process:

```python
from src.utils.embeddings import EmbeddingGenerator

# Use a different embedding model
embedding_generator = EmbeddingGenerator(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Generate embeddings
entities_with_embeddings = embedding_generator.generate_entity_embeddings(entities_df)
```
