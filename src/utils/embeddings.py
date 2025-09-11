"""
Utility functions for generating and managing embeddings.
"""
import os
from typing import List, Dict, Any, Optional, Union
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

class EmbeddingGenerator:
    """Generate embeddings for entities in the knowledge graph."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the sentence transformer model
        """
        self.model = SentenceTransformer(model_name)
    
    def generate_entity_embeddings(self, entities_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate embeddings for entities.
        
        Args:
            entities_df: DataFrame with entity data
            
        Returns:
            DataFrame with added embedding column
        """
        # Create a copy to avoid modifying the original
        df = entities_df.copy()
        
        # Generate embeddings for each entity text
        embeddings = []
        
        for text in tqdm(df['text'], desc="Generating embeddings"):
            embedding = self.model.encode(text)
            embeddings.append(embedding.tolist())
        
        df['embedding'] = embeddings
        return df
    
    def save_embeddings(self, entities_df: pd.DataFrame, output_path: str) -> None:
        """
        Save entities with embeddings to a file.
        
        Args:
            entities_df: DataFrame with entity data and embeddings
            output_path: Path to save the data
        """
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to CSV (note: this will convert embeddings to strings)
        entities_df.to_csv(output_path, index=False)
        
        print(f"Saved entity embeddings to {output_path}")
    
    def load_embeddings(self, input_path: str) -> pd.DataFrame:
        """
        Load entities with embeddings from a file.
        
        Args:
            input_path: Path to load the data from
            
        Returns:
            DataFrame with entity data and embeddings
        """
        df = pd.read_csv(input_path)
        
        # Convert embedding strings back to lists
        if 'embedding' in df.columns:
            df['embedding'] = df['embedding'].apply(lambda x: np.fromstring(
                x.strip('[]'), sep=',').tolist() if isinstance(x, str) else x
            )
        
        return df
