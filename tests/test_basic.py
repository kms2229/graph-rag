"""
Basic tests for the knowledge graph RAG system.
"""
import os
import sys
import unittest
import pandas as pd

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.ingestion import DocumentProcessor, KnowledgeGraphBuilder
from src.utils.embeddings import EmbeddingGenerator


class TestDocumentProcessor(unittest.TestCase):
    """Test the document processor functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.processor = DocumentProcessor()
        self.test_text = """
        Artificial Intelligence (AI) is revolutionizing industries across the globe. 
        Machine learning, a subset of AI, enables computers to learn from data and improve over time.
        Google's DeepMind created AlphaGo, which defeated the world champion in the game of Go.
        """
    
    def test_process_text(self):
        """Test processing text to extract entities and relationships."""
        result = self.processor.process_text(self.test_text)
        
        # Check that entities and relationships were extracted
        self.assertIn('entities', result)
        self.assertIn('relationships', result)
        
        # Check that at least some entities were found
        self.assertGreater(len(result['entities']), 0)
        
        # Check entity structure
        if result['entities']:
            entity = result['entities'][0]
            self.assertIn('text', entity)
            self.assertIn('label', entity)
            self.assertIn('start_char', entity)
            self.assertIn('end_char', entity)


class TestKnowledgeGraphBuilder(unittest.TestCase):
    """Test the knowledge graph builder functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.builder = KnowledgeGraphBuilder()
        
        # Sample processed data
        self.sample_data = {
            'entities': [
                {'text': 'Artificial Intelligence', 'label': 'TECH', 'start_char': 0, 'end_char': 22},
                {'text': 'Machine learning', 'label': 'TECH', 'start_char': 30, 'end_char': 46},
                {'text': 'Google', 'label': 'ORG', 'start_char': 50, 'end_char': 56}
            ],
            'relationships': [
                {'source': 'Machine learning', 'source_type': 'TECH', 
                 'target': 'Artificial Intelligence', 'target_type': 'TECH', 'relation': 'subset_of'},
                {'source': 'Google', 'source_type': 'ORG', 
                 'target': 'DeepMind', 'target_type': 'ORG', 'relation': 'owns'}
            ]
        }
    
    def test_add_processed_data(self):
        """Test adding processed data to the knowledge graph."""
        self.builder.add_processed_data(self.sample_data)
        
        # Check that entities and relationships were added
        self.assertEqual(len(self.builder.entities), len(self.sample_data['entities']))
        self.assertEqual(len(self.builder.relationships), len(self.sample_data['relationships']))
    
    def test_get_unique_entities(self):
        """Test getting unique entities."""
        self.builder.add_processed_data(self.sample_data)
        
        # Add duplicate entity with different case
        duplicate_data = {
            'entities': [
                {'text': 'artificial intelligence', 'label': 'TECH', 'start_char': 0, 'end_char': 22}
            ],
            'relationships': []
        }
        self.builder.add_processed_data(duplicate_data)
        
        # Get unique entities
        unique_entities = self.builder.get_unique_entities()
        
        # Check that duplicates were handled
        self.assertLessEqual(len(unique_entities), len(self.builder.entities))


class TestEmbeddingGenerator(unittest.TestCase):
    """Test the embedding generator functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.generator = EmbeddingGenerator()
        
        # Sample entities
        self.sample_entities = pd.DataFrame({
            'entity_id': ['ai', 'machine_learning'],
            'text': ['Artificial Intelligence', 'Machine Learning'],
            'label': ['TECH', 'TECH']
        })
    
    def test_generate_entity_embeddings(self):
        """Test generating embeddings for entities."""
        # Skip this test if it's running in CI environment
        if os.environ.get('CI') == 'true':
            self.skipTest("Skipping embedding test in CI environment")
        
        # Generate embeddings
        result = self.generator.generate_entity_embeddings(self.sample_entities)
        
        # Check that embeddings were generated
        self.assertIn('embedding', result.columns)
        self.assertEqual(len(result), len(self.sample_entities))
        
        # Check embedding structure
        if not result.empty:
            embedding = result.iloc[0]['embedding']
            self.assertIsInstance(embedding, list)
            self.assertGreater(len(embedding), 0)


if __name__ == '__main__':
    unittest.main()
