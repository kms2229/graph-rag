#!/bin/bash

# Exit on error
set -e

echo "Setting up Graph RAG environment..."

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install core dependencies first
pip install --upgrade pip
pip install numpy>=1.26.0 pandas>=2.2.0

# Install remaining requirements
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_md

# Create necessary directories
mkdir -p data/sample
mkdir -p data/output

echo "Installation complete! Activate the environment with:"
echo "source venv/bin/activate"
