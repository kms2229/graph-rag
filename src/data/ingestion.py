"""
Data ingestion module for building knowledge graphs from text documents.
"""
import os
import spacy
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
from tqdm import tqdm

class DocumentProcessor:
    """Process documents to extract entities and relationships for knowledge graph construction."""
    
    def __init__(self, spacy_model: str = "en_core_web_md"):
        """
        Initialize the document processor.
        
        Args:
            spacy_model: The spaCy model to use for NLP tasks
        """
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            print(f"Downloading spaCy model {spacy_model}...")
            os.system(f"python -m spacy download {spacy_model}")
            self.nlp = spacy.load(spacy_model)
        
        # Add custom components if needed
        # self.nlp.add_pipe(...)
    
    def process_text(self, text: str) -> Dict[str, Any]:
        """
        Process a text document to extract entities and relationships.
        
        Args:
            text: The text content to process
            
        Returns:
            Dict containing extracted entities and relationships
        """
        doc = self.nlp(text)
        
        # Extract entities
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start_char": ent.start_char,
                "end_char": ent.end_char
            })
        
        # Extract relationships (basic co-occurrence)
        relationships = []
        for i, ent1 in enumerate(doc.ents):
            for j, ent2 in enumerate(doc.ents):
                if i != j:
                    # Simple co-occurrence within a window
                    if abs(ent1.start - ent2.start) < 10:  # Within 10 tokens
                        relationships.append({
                            "source": ent1.text,
                            "source_type": ent1.label_,
                            "target": ent2.text,
                            "target_type": ent2.label_,
                            "relation": "co-occurs_with"
                        })
        
        # Extract subject-verb-object relationships
        for sent in doc.sents:
            for token in sent:
                if token.dep_ == "ROOT" and token.pos_ == "VERB":
                    verb = token.text
                    
                    # Find subject
                    subject = None
                    for child in token.children:
                        if child.dep_ in ["nsubj", "nsubjpass"]:
                            subject = child
                            break
                    
                    # Find object
                    obj = None
                    for child in token.children:
                        if child.dep_ in ["dobj", "pobj"]:
                            obj = child
                            break
                    
                    if subject and obj:
                        relationships.append({
                            "source": subject.text,
                            "source_type": "SUBJECT",
                            "target": obj.text,
                            "target_type": "OBJECT",
                            "relation": verb
                        })
        
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
        
        return self.process_text(text)
    
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
            for file in tqdm(files, desc="Processing files"):
                if file.endswith(('.txt', '.md', '.csv')):
                    file_path = os.path.join(root, file)
                    try:
                        result = self.process_file(file_path)
                        result['source_file'] = file_path
                        results.append(result)
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
        
        return results


class KnowledgeGraphBuilder:
    """Build knowledge graph data from processed documents."""
    
    def __init__(self):
        """Initialize the knowledge graph builder."""
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
            print(f"Error in get_unique_entities: {e}")
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
        
        entities_path = os.path.join(output_dir, 'entities.csv')
        relationships_path = os.path.join(output_dir, 'relationships.csv')
        
        self.get_unique_entities().to_csv(entities_path, index=False)
        self.get_relationships_df().to_csv(relationships_path, index=False)
        
        return entities_path, relationships_path
