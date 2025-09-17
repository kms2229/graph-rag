from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="graph_rag",
    version="0.1.0",
    author="Graph RAG Team",
    author_email="example@example.com",
    description="Knowledge Graph RAG System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/graph_rag",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        # Core dependencies
        "networkx>=3.1",
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        
        # Graph database
        "neo4j>=5.14.1",
        
        # Vector embeddings
        "sentence-transformers>=2.2.2",
        
        # RAG components
        "langchain>=0.1.0",
        "langchain-community>=0.0.13",
        "langchain-openai>=0.0.2",
        "openai>=1.0.0",
        
        # Visualization
        "matplotlib>=3.8.0",
        "pyvis>=0.3.2",
        
        # Web interface
        "streamlit>=1.32.0",
        "streamlit-agraph>=0.0.45",
        
        # Utilities
        "tqdm>=4.66.1",
        "python-dotenv>=1.0.0",
        
        # NLP - install last to ensure dependencies are met
        "spacy>=3.7.2",
    ],
    python_requires=">=3.8,<3.14",
    entry_points={
        "console_scripts": [
            "graph-rag=cli:main",
        ],
    },
)
