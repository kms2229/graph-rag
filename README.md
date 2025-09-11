# Knowledge Graph RAG System

This project implements a Retrieval-Augmented Generation (RAG) system using a knowledge graph approach. By representing information as a graph rather than isolated chunks, we can capture relationships between entities and provide more context-aware responses.

## Features

- Knowledge graph construction from text documents
- Entity and relationship extraction using NLP
- Graph-based retrieval for RAG queries
- Neo4j integration for graph storage and querying

## Project Structure

```
graph_rag/
├── data/                # Data storage
├── notebooks/          # Jupyter notebooks for exploration and demos
├── src/                # Source code
│   ├── data/           # Data processing modules
│   ├── graph/          # Knowledge graph construction and querying
│   ├── rag/            # RAG implementation
│   └── utils/          # Utility functions
└── tests/              # Unit tests
```

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set up Neo4j:
   - Install Neo4j Desktop or use Neo4j Aura cloud
   - Create a new database
   - Set environment variables for connection

3. Run the example:
   ```
   python src/main.py
   ```

## Environment Variables

Create a `.env` file with:

```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```
