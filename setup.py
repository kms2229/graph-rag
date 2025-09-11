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
    python_requires=">=3.8",
    install_requires=[
        "networkx>=3.1",
        "spacy>=3.7.2",
        "numpy>=1.24.3",
        "pandas>=2.0.3",
        "neo4j>=5.14.1",
        "sentence-transformers>=2.2.2",
        "langchain>=0.0.335",
        "langchain-community>=0.0.13",
        "matplotlib>=3.7.3",
        "pyvis>=0.3.2",
        "tqdm>=4.66.1",
        "python-dotenv>=1.0.0",
    ],
    entry_points={
        "console_scripts": [
            "graph-rag=cli:main",
        ],
    },
)
