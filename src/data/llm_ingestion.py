"""
LLM-based data ingestion module for building knowledge graphs from text documents.
"""
import os
import json
from typing import List, Dict, Any, Tuple, Optional, Union
import pandas as pd
from tqdm import tqdm
import numpy as np
from dotenv import load_dotenv
import requests
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Load environment variables
load_dotenv()

class LLMDocumentProcessor:
    """Process documents using LLMs to extract entities and relationships for knowledge graph construction."""
    
    def __init__(
        self, 
        model_type: str = "openai",
        openai_api_key: Optional[str] = None,
        bert_model: str = "dslim/bert-base-NER",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize the LLM document processor.
        
        Args:
            model_type: Type of model to use ('openai' or 'bert')
            openai_api_key: OpenAI API key (defaults to environment variable)
            bert_model: BERT model for NER if using 'bert'
            embedding_model: Sentence transformer model for embeddings
        """
        self.model_type = model_type
        
        # Set up OpenAI
        if model_type == "openai":
            self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
            if not self.openai_api_key:
                raise ValueError("OpenAI API key is required for OpenAI model")
            self.llm = OpenAI(api_key=self.openai_api_key, temperature=0.0)
        
        # Set up BERT
        elif model_type == "bert":
            self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
            self.model = AutoModelForTokenClassification.from_pretrained(bert_model)
            self.ner_pipeline = pipeline("ner", model=self.model, tokenizer=self.tokenizer, aggregation_strategy="simple")
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Set up embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
    
    def _extract_entities_openai(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities using OpenAI.
        
        Args:
            text: The text content to process
            
        Returns:
            List of extracted entities
        """
        prompt_template = """
        Extract named entities from the following text. Return the result as a JSON array where each entity has "text", "label", "start_char", and "end_char" properties.
        
        Labels to use: PERSON, ORGANIZATION, LOCATION, PRODUCT, EVENT, WORK_OF_ART, LAW, DATE, TIME, PERCENT, MONEY, QUANTITY, TECHNOLOGY, CONCEPT
        
        Text: {text}
        
        JSON output:
        """
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["text"]
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        result = chain.run(text=text)
        
        try:
            entities = json.loads(result)
            return entities
        except json.JSONDecodeError:
            print(f"Error parsing OpenAI response: {result}")
            return []
    
    def _extract_relationships_openai(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract relationships using OpenAI.
        
        Args:
            text: The text content to process
            entities: List of extracted entities
            
        Returns:
            List of extracted relationships
        """
        # Create a list of entity texts for the prompt
        entity_texts = [f"{entity['text']} ({entity['label']})" for entity in entities]
        entity_list = "\n".join(entity_texts)
        
        prompt_template = """
        Extract relationships between entities in the following text. Return the result as a JSON array where each relationship has "source", "source_type", "target", "target_type", and "relation" properties.
        
        Text: {text}
        
        Entities:
        {entity_list}
        
        Extract meaningful relationships between these entities. The "relation" should be a verb or phrase describing how the source entity relates to the target entity.
        
        JSON output:
        """
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["text", "entity_list"]
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        result = chain.run(text=text, entity_list=entity_list)
        
        try:
            relationships = json.loads(result)
            return relationships
        except json.JSONDecodeError:
            print(f"Error parsing OpenAI response: {result}")
            return []
    
    def _extract_entities_bert(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities using BERT NER.
        
        Args:
            text: The text content to process
            
        Returns:
            List of extracted entities
        """
        ner_results = self.ner_pipeline(text)
        
        entities = []
        for entity in ner_results:
            entities.append({
                "text": entity["word"],
                "label": entity["entity_group"],
                "start_char": entity["start"],
                "end_char": entity["end"]
            })
        
        return entities
    
    def _extract_relationships_bert(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract relationships using a rule-based approach with BERT entities.
        
        Args:
            text: The text content to process
            entities: List of extracted entities
            
        Returns:
            List of extracted relationships
        """
        relationships = []
        
        # Simple co-occurrence within a window
        for i, ent1 in enumerate(entities):
            for j, ent2 in enumerate(entities):
                if i != j:
                    # Check if entities are close to each other in the text
                    distance = abs(ent1["start_char"] - ent2["start_char"])
                    if distance < 100:  # Within 100 characters
                        relationships.append({
                            "source": ent1["text"],
                            "source_type": ent1["label"],
                            "target": ent2["text"],
                            "target_type": ent2["label"],
                            "relation": "related_to"
                        })
        
        return relationships
    
    def process_text(self, text: str) -> Dict[str, Any]:
        """
        Process a text document to extract entities and relationships using LLMs.
        
        Args:
            text: The text content to process
            
        Returns:
            Dict containing extracted entities and relationships
        """
        # Extract entities
        if self.model_type == "openai":
            entities = self._extract_entities_openai(text)
            relationships = self._extract_relationships_openai(text, entities)
        else:  # bert
            entities = self._extract_entities_bert(text)
            relationships = self._extract_relationships_bert(text, entities)
        
        return {
            "entities": entities,
            "relationships": relationships
        }
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process a text file to extract entities and relationships.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            Dict containing extracted entities and relationships
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        result = self.process_text(text)
        result['source_file'] = file_path
        return result
    
    def process_directory(self, dir_path: str) -> List[Dict[str, Any]]:
        """
        Process all text files in a directory.
        
        Args:
            dir_path: Path to the directory containing text files
            
        Returns:
            List of dicts containing extracted entities and relationships
        """
        results = []
        
        for root, _, files in os.walk(dir_path):
            for file in tqdm(files, desc="Processing files with LLM"):
                if file.endswith(('.txt', '.md', '.csv')):
                    file_path = os.path.join(root, file)
                    try:
                        result = self.process_file(file_path)
                        results.append(result)
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
        
        return results


class LLMKnowledgeGraphBuilder:
    """Build knowledge graph data from LLM-processed documents."""
    
    def __init__(self):
        """Initialize the LLM knowledge graph builder."""
        self.entities = []
        self.relationships = []
    
    def add_processed_data(self, processed_data: Dict[str, Any]) -> None:
        """
        Add processed document data to the knowledge graph.
        
        Args:
            processed_data: Dict containing entities and relationships
        """
        self.entities.extend(processed_data.get('entities', []))
        self.relationships.extend(processed_data.get('relationships', []))
    
    def get_unique_entities(self) -> pd.DataFrame:
        """
        Get unique entities with their types.
        
        Returns:
            DataFrame of unique entities
        """
        # Return empty DataFrame if no entities
        if not self.entities:
            return pd.DataFrame(columns=['entity_id', 'text', 'label'])
        
        try:
            # Create DataFrame from entities
            entities_df = pd.DataFrame(self.entities)
            
            # Handle completely empty DataFrame
            if entities_df.empty:
                return pd.DataFrame(columns=['entity_id', 'text', 'label'])
            
            # Map common column names to expected column names
            column_mapping = {
                'name': 'text',
                'type': 'label',
                'entity': 'text',
                'entity_type': 'label',
                'category': 'label'
            }
            
            # Apply column mapping if needed
            for source, target in column_mapping.items():
                if source in entities_df.columns and target not in entities_df.columns:
                    entities_df[target] = entities_df[source]
            
            # Ensure required columns exist
            if 'text' not in entities_df.columns:
                entities_df['text'] = 'Unknown Entity'
            
            if 'label' not in entities_df.columns:
                entities_df['label'] = 'UNKNOWN'
            
            # Create entity_id from text
            entities_df['entity_id'] = entities_df['text'].str.lower()
            
            # Simple deduplication if we can't do the more complex approach
            result = entities_df.drop_duplicates('entity_id')
            
            # Ensure we have all required columns
            for col in ['entity_id', 'text', 'label']:
                if col not in result.columns:
                    if col == 'entity_id':
                        result['entity_id'] = result['text'].str.lower() if 'text' in result.columns else ['entity_' + str(i) for i in range(len(result))]  
                    elif col == 'text':
                        result['text'] = 'Unknown Entity'
                    elif col == 'label':
                        result['label'] = 'UNKNOWN'
            
            # Return only the columns we need
            return result[['entity_id', 'text', 'label']]
            
        except Exception as e:
            print(f"Error in LLMDocumentProcessor.get_unique_entities: {e}")
            # Return a minimal valid DataFrame as fallback
            return pd.DataFrame({
                'entity_id': ['fallback_entity'],
                'text': ['Error Processing Entities'],
                'label': ['ERROR']
            })
    
    def get_relationships_df(self) -> pd.DataFrame:
        """
        Get relationships as a DataFrame.
        
        Returns:
            DataFrame of relationships
        """
        if not self.relationships:
            return pd.DataFrame(columns=['source', 'target', 'relation'])
        
        return pd.DataFrame(self.relationships)
    
    def export_to_csv(self, output_dir: str) -> Tuple[str, str]:
        """
        Export entities and relationships to CSV files.
        
        Args:
            output_dir: Directory to save the CSV files
            
        Returns:
            Tuple of (entities_path, relationships_path)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        entities_path = os.path.join(output_dir, 'llm_entities.csv')
        relationships_path = os.path.join(output_dir, 'llm_relationships.csv')
        
        self.get_unique_entities().to_csv(entities_path, index=False)
        self.get_relationships_df().to_csv(relationships_path, index=False)
        
        return entities_path, relationships_path
    
    def enrich_entities_with_embeddings(self, embedding_model: Optional[SentenceTransformer] = None) -> pd.DataFrame:
        """
        Enrich entities with vector embeddings.
        
        Args:
            embedding_model: Optional SentenceTransformer model
            
        Returns:
            DataFrame with entities and their embeddings
        """
        entities_df = self.get_unique_entities()
        
        if embedding_model is None:
            embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Generate embeddings for each entity text
        embeddings = []
        
        for text in tqdm(entities_df['text'], desc="Generating embeddings"):
            embedding = embedding_model.encode(text)
            embeddings.append(embedding.tolist())
        
        entities_df['embedding'] = embeddings
        return entities_df
