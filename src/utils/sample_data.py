"""
Sample data generation for the knowledge graph RAG system.
"""
import os
import argparse
import random
from typing import List, Dict, Any, Optional, Tuple

def generate_ai_documents(output_dir: str, num_docs: int = 5) -> List[str]:
    """
    Generate sample AI-related documents.
    
    Args:
        output_dir: Directory to save the documents
        num_docs: Number of documents to generate
        
    Returns:
        List of file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define topics and related entities
    topics = {
        "machine_learning": {
            "entities": ["Machine Learning", "Neural Networks", "Deep Learning", "Supervised Learning", 
                         "Unsupervised Learning", "Reinforcement Learning", "Decision Trees", "Random Forests",
                         "Support Vector Machines", "Gradient Boosting"],
            "companies": ["Google", "Microsoft", "Amazon", "Facebook", "OpenAI", "DeepMind"],
            "applications": ["Image Recognition", "Speech Recognition", "Natural Language Processing",
                            "Recommendation Systems", "Fraud Detection", "Autonomous Vehicles"]
        },
        "nlp": {
            "entities": ["Natural Language Processing", "NLP", "Language Models", "Transformers", 
                         "BERT", "GPT", "T5", "Word Embeddings", "Named Entity Recognition", "Sentiment Analysis"],
            "companies": ["OpenAI", "Hugging Face", "Google", "Microsoft", "Meta AI", "Anthropic"],
            "applications": ["Machine Translation", "Question Answering", "Text Summarization",
                            "Chatbots", "Content Generation", "Information Extraction"]
        },
        "knowledge_graphs": {
            "entities": ["Knowledge Graph", "Graph Database", "Ontology", "Entity", "Relationship",
                         "Triple", "RDF", "SPARQL", "Graph Neural Networks", "Node Embeddings"],
            "companies": ["Neo4j", "Google", "Microsoft", "Amazon", "Oracle", "TigerGraph"],
            "applications": ["Search Enhancement", "Recommendation Systems", "Fraud Detection",
                            "Drug Discovery", "Knowledge Management", "Question Answering"]
        },
        "computer_vision": {
            "entities": ["Computer Vision", "Image Recognition", "Object Detection", "Image Segmentation",
                         "Convolutional Neural Networks", "CNN", "Feature Extraction", "YOLO", "R-CNN", "GANs"],
            "companies": ["NVIDIA", "Google", "Microsoft", "Tesla", "OpenAI", "Scale AI"],
            "applications": ["Autonomous Vehicles", "Medical Imaging", "Surveillance", "Augmented Reality",
                            "Face Recognition", "Product Visual Search"]
        },
        "rag": {
            "entities": ["Retrieval-Augmented Generation", "RAG", "Vector Database", "Semantic Search",
                         "Information Retrieval", "Document Embedding", "Query Expansion", "Reranking"],
            "companies": ["Pinecone", "Weaviate", "OpenAI", "Anthropic", "Cohere", "Langchain"],
            "applications": ["Question Answering", "Chatbots", "Search Engines", "Content Generation",
                            "Knowledge Management", "Customer Support"]
        }
    }
    
    # Templates for document generation
    templates = [
        "{topic} is a rapidly evolving field in artificial intelligence. {entity1} and {entity2} are key concepts in this area. "
        "Companies like {company1} and {company2} are leading the development of {topic} technologies. "
        "Common applications include {application1} and {application2}.",
        
        "In recent years, {topic} has transformed various industries. {entity1} is particularly important for solving complex problems. "
        "{company1} has developed advanced {topic} solutions that leverage {entity2}. "
        "These technologies are widely used for {application1} and can also be applied to {application2}.",
        
        "The field of {topic} continues to advance at a remarkable pace. Researchers at {company1} and {company2} "
        "have made significant breakthroughs in {entity1}. These innovations have enabled new applications such as "
        "{application1}. {entity2} is another promising approach that is gaining traction in the industry.",
        
        "{company1} and {company2} are competing to develop the most advanced {topic} systems. "
        "Their technologies rely heavily on {entity1} and {entity2} to achieve state-of-the-art performance. "
        "The most common applications of these systems include {application1} and {application2}.",
        
        "Researchers are exploring new approaches to {topic} that go beyond traditional methods. "
        "{entity1} has shown promising results in recent studies. {company1} has integrated this approach into "
        "their products for {application1}. Meanwhile, {company2} is focusing on {entity2} for solving {application2} challenges."
    ]
    
    file_paths = []
    
    for i in range(num_docs):
        # Select a random topic and template
        topic_key = random.choice(list(topics.keys()))
        topic_data = topics[topic_key]
        template = random.choice(templates)
        
        # Select random entities, companies, and applications
        entities = random.sample(topic_data["entities"], min(2, len(topic_data["entities"])))
        companies = random.sample(topic_data["companies"], min(2, len(topic_data["companies"])))
        applications = random.sample(topic_data["applications"], min(2, len(topic_data["applications"])))
        
        # Format the document
        topic_name = topic_key.replace("_", " ").title()
        document = template.format(
            topic=topic_name,
            entity1=entities[0],
            entity2=entities[1] if len(entities) > 1 else entities[0],
            company1=companies[0],
            company2=companies[1] if len(companies) > 1 else companies[0],
            application1=applications[0],
            application2=applications[1] if len(applications) > 1 else applications[0]
        )
        
        # Add some additional sentences for variety
        additional_sentences = [
            f"{random.choice(companies)} is investing heavily in {random.choice(entities)}.",
            f"{random.choice(applications)} is one of the most promising applications of {topic_name}.",
            f"The combination of {entities[0]} and {entities[1] if len(entities) > 1 else random.choice(topic_data['entities'])} is particularly powerful.",
            f"Researchers at {random.choice(companies)} have published several papers on {random.choice(entities)}.",
            f"{random.choice(entities)} can significantly improve the performance of {random.choice(applications)} systems."
        ]
        
        # Add 2-3 random additional sentences
        for _ in range(random.randint(2, 3)):
            document += " " + random.choice(additional_sentences)
        
        # Save the document
        file_name = f"{topic_key}_{i+1}.txt"
        file_path = os.path.join(output_dir, file_name)
        
        with open(file_path, "w") as f:
            f.write(document)
        
        file_paths.append(file_path)
    
    return file_paths

def generate_structured_knowledge_graph(output_dir: str) -> Tuple[str, str]:
    """
    Generate a structured knowledge graph with predefined entities and relationships.
    
    Args:
        output_dir: Directory to save the knowledge graph data
        
    Returns:
        Tuple of (entities_path, relationships_path)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define entities with types
    entities = [
        {"entity_id": "machine_learning", "text": "Machine Learning", "label": "TECHNOLOGY"},
        {"entity_id": "deep_learning", "text": "Deep Learning", "label": "TECHNOLOGY"},
        {"entity_id": "neural_networks", "text": "Neural Networks", "label": "TECHNOLOGY"},
        {"entity_id": "nlp", "text": "Natural Language Processing", "label": "TECHNOLOGY"},
        {"entity_id": "computer_vision", "text": "Computer Vision", "label": "TECHNOLOGY"},
        {"entity_id": "reinforcement_learning", "text": "Reinforcement Learning", "label": "TECHNOLOGY"},
        {"entity_id": "knowledge_graphs", "text": "Knowledge Graphs", "label": "TECHNOLOGY"},
        {"entity_id": "rag", "text": "Retrieval-Augmented Generation", "label": "TECHNOLOGY"},
        
        {"entity_id": "google", "text": "Google", "label": "ORGANIZATION"},
        {"entity_id": "microsoft", "text": "Microsoft", "label": "ORGANIZATION"},
        {"entity_id": "openai", "text": "OpenAI", "label": "ORGANIZATION"},
        {"entity_id": "meta", "text": "Meta", "label": "ORGANIZATION"},
        {"entity_id": "deepmind", "text": "DeepMind", "label": "ORGANIZATION"},
        {"entity_id": "anthropic", "text": "Anthropic", "label": "ORGANIZATION"},
        {"entity_id": "hugging_face", "text": "Hugging Face", "label": "ORGANIZATION"},
        
        {"entity_id": "gpt4", "text": "GPT-4", "label": "MODEL"},
        {"entity_id": "bert", "text": "BERT", "label": "MODEL"},
        {"entity_id": "t5", "text": "T5", "label": "MODEL"},
        {"entity_id": "llama", "text": "LLaMA", "label": "MODEL"},
        {"entity_id": "claude", "text": "Claude", "label": "MODEL"},
        {"entity_id": "gemini", "text": "Gemini", "label": "MODEL"},
        
        {"entity_id": "image_recognition", "text": "Image Recognition", "label": "APPLICATION"},
        {"entity_id": "machine_translation", "text": "Machine Translation", "label": "APPLICATION"},
        {"entity_id": "question_answering", "text": "Question Answering", "label": "APPLICATION"},
        {"entity_id": "recommendation_systems", "text": "Recommendation Systems", "label": "APPLICATION"},
        {"entity_id": "autonomous_vehicles", "text": "Autonomous Vehicles", "label": "APPLICATION"}
    ]
    
    # Define relationships
    relationships = [
        {"source": "deep_learning", "target": "machine_learning", "relation": "SUBSET_OF"},
        {"source": "neural_networks", "target": "machine_learning", "relation": "TECHNIQUE_IN"},
        {"source": "reinforcement_learning", "target": "machine_learning", "relation": "TYPE_OF"},
        
        {"source": "nlp", "target": "machine_learning", "relation": "APPLICATION_OF"},
        {"source": "computer_vision", "target": "machine_learning", "relation": "APPLICATION_OF"},
        {"source": "knowledge_graphs", "target": "nlp", "relation": "RELATED_TO"},
        {"source": "rag", "target": "knowledge_graphs", "relation": "USES"},
        {"source": "rag", "target": "nlp", "relation": "USES"},
        
        {"source": "google", "target": "deepmind", "relation": "OWNS"},
        {"source": "google", "target": "gemini", "relation": "DEVELOPED"},
        {"source": "microsoft", "target": "openai", "relation": "INVESTED_IN"},
        {"source": "openai", "target": "gpt4", "relation": "DEVELOPED"},
        {"source": "meta", "target": "llama", "relation": "DEVELOPED"},
        {"source": "anthropic", "target": "claude", "relation": "DEVELOPED"},
        {"source": "google", "target": "bert", "relation": "DEVELOPED"},
        {"source": "google", "target": "t5", "relation": "DEVELOPED"},
        {"source": "hugging_face", "target": "bert", "relation": "PROVIDES"},
        
        {"source": "computer_vision", "target": "image_recognition", "relation": "ENABLES"},
        {"source": "nlp", "target": "machine_translation", "relation": "ENABLES"},
        {"source": "nlp", "target": "question_answering", "relation": "ENABLES"},
        {"source": "knowledge_graphs", "target": "recommendation_systems", "relation": "IMPROVES"},
        {"source": "computer_vision", "target": "autonomous_vehicles", "relation": "ENABLES"},
        
        {"source": "gpt4", "target": "question_answering", "relation": "USED_FOR"},
        {"source": "bert", "target": "nlp", "relation": "USED_IN"},
        {"source": "t5", "target": "machine_translation", "relation": "USED_FOR"},
        {"source": "gemini", "target": "image_recognition", "relation": "CAPABLE_OF"},
        {"source": "llama", "target": "nlp", "relation": "DESIGNED_FOR"}
    ]
    
    # Save entities and relationships to CSV
    import pandas as pd
    
    entities_df = pd.DataFrame(entities)
    relationships_df = pd.DataFrame(relationships)
    
    entities_path = os.path.join(output_dir, "sample_entities.csv")
    relationships_path = os.path.join(output_dir, "sample_relationships.csv")
    
    entities_df.to_csv(entities_path, index=False)
    relationships_df.to_csv(relationships_path, index=False)
    
    return entities_path, relationships_path

def main():
    parser = argparse.ArgumentParser(description="Generate sample data for knowledge graph RAG")
    parser.add_argument("--output_dir", default="../data/sample", help="Output directory for sample data")
    parser.add_argument("--num_docs", type=int, default=10, help="Number of documents to generate")
    parser.add_argument("--structured", action="store_true", help="Generate structured knowledge graph data")
    
    args = parser.parse_args()
    
    if args.structured:
        print("Generating structured knowledge graph data...")
        entities_path, relationships_path = generate_structured_knowledge_graph(args.output_dir)
        print(f"Generated structured knowledge graph data:")
        print(f"  - Entities: {entities_path}")
        print(f"  - Relationships: {relationships_path}")
    else:
        print(f"Generating {args.num_docs} sample documents...")
        file_paths = generate_ai_documents(args.output_dir, args.num_docs)
        print(f"Generated {len(file_paths)} sample documents in {args.output_dir}")

if __name__ == "__main__":
    main()
