"""
Test script for JSON file processing in the knowledge graph system.
"""
import os
import json
import pandas as pd
from src.data.ingestion import DocumentProcessor
from src.data.llm_ingestion import LLMDocumentProcessor

# Create a test JSON file
test_json_path = "test_data.json"
test_data = {
    "title": "Apple Inc.",
    "content": "Apple Inc. is an American multinational technology company headquartered in Cupertino, California. Tim Cook is the CEO of Apple. Apple designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories.",
    "entities": [
        {"name": "Apple Inc.", "type": "ORGANIZATION"},
        {"name": "Tim Cook", "type": "PERSON"},
        {"name": "Cupertino", "type": "LOCATION"},
        {"name": "California", "type": "LOCATION"}
    ],
    "relationships": [
        {"source": "Tim Cook", "target": "Apple Inc.", "relation": "CEO_OF"},
        {"source": "Apple Inc.", "target": "Cupertino", "relation": "HEADQUARTERED_IN"},
        {"source": "Cupertino", "target": "California", "relation": "LOCATED_IN"}
    ]
}

# Write test data to JSON file
with open(test_json_path, 'w') as f:
    json.dump(test_data, f, indent=2)

print(f"Created test JSON file: {test_json_path}")

# Test with standard DocumentProcessor
print("\n--- Testing with DocumentProcessor ---")
doc_processor = DocumentProcessor()

try:
    # This should fail as DocumentProcessor doesn't support JSON
    result = doc_processor.process_file(test_json_path)
    print("DocumentProcessor processed JSON (unexpected):", result.keys())
except Exception as e:
    print(f"DocumentProcessor failed as expected: {e}")

# Test with LLMDocumentProcessor
print("\n--- Testing with LLMDocumentProcessor ---")
try:
    # Try with OpenAI mode first
    llm_processor = LLMDocumentProcessor(mode="openai")
    result = llm_processor.process_file(test_json_path)
    print("LLMDocumentProcessor (OpenAI) processed JSON:", result.keys())
    print(f"Extracted {len(result.get('entities', []))} entities and {len(result.get('relationships', []))} relationships")
except Exception as e:
    print(f"LLMDocumentProcessor (OpenAI) failed: {e}")
    print("Falling back to BERT mode...")
    
    # Fall back to BERT mode if OpenAI fails
    try:
        llm_processor = LLMDocumentProcessor(mode="bert")
        result = llm_processor.process_file(test_json_path)
        print("LLMDocumentProcessor (BERT) processed JSON:", result.keys())
        print(f"Extracted {len(result.get('entities', []))} entities and {len(result.get('relationships', []))} relationships")
        
        # Display entities and relationships
        if result.get('entities'):
            df = pd.DataFrame(result['entities'])
            print("\nExtracted entities:")
            print(df.head())
        
        if result.get('relationships'):
            df = pd.DataFrame(result['relationships'])
            print("\nExtracted relationships:")
            print(df.head())
    except Exception as e:
        print(f"LLMDocumentProcessor (BERT) failed: {e}")

# Clean up
print("\nCleaning up...")
if os.path.exists(test_json_path):
    os.remove(test_json_path)
    print(f"Removed test file: {test_json_path}")

print("\nTest completed.")
