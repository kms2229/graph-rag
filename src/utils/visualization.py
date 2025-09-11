"""
Visualization utilities for knowledge graphs.
"""
import os
import json
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network


def create_networkx_graph(
    entities_df: pd.DataFrame, 
    relationships_df: pd.DataFrame
) -> nx.DiGraph:
    """
    Create a NetworkX graph from entities and relationships.
    
    Args:
        entities_df: DataFrame with entity data
        relationships_df: DataFrame with relationship data
        
    Returns:
        NetworkX DiGraph
    """
    G = nx.DiGraph()
    
    # Add nodes
    for _, row in entities_df.iterrows():
        G.add_node(
            row['entity_id'],
            label=row['text'],
            type=row['label']
        )
    
    # Add edges
    for _, row in relationships_df.iterrows():
        G.add_edge(
            row['source'],
            row['target'],
            label=row['relation']
        )
    
    return G


def visualize_graph(
    G: nx.DiGraph,
    output_path: str = "knowledge_graph.html",
    height: str = "750px",
    width: str = "100%",
    notebook: bool = False,
    show_buttons: bool = True,
    node_size_field: Optional[str] = None,
    edge_weight_field: Optional[str] = None,
    node_color_field: str = "type"
) -> None:
    """
    Visualize a NetworkX graph using PyVis.
    
    Args:
        G: NetworkX graph
        output_path: Path to save the HTML visualization
        height: Height of the visualization
        width: Width of the visualization
        notebook: Whether to display in a Jupyter notebook
        show_buttons: Whether to show control buttons
        node_size_field: Node attribute to use for sizing nodes
        edge_weight_field: Edge attribute to use for edge weights
        node_color_field: Node attribute to use for coloring nodes
    """
    # Create PyVis network
    net = Network(height=height, width=width, notebook=notebook, directed=True)
    
    # Get unique node types for coloring
    node_types = set()
    for _, node_data in G.nodes(data=True):
        if node_color_field in node_data:
            node_types.add(node_data[node_color_field])
    
    # Generate color map
    import matplotlib.cm as cm
    import numpy as np
    
    colors = cm.rainbow(np.linspace(0, 1, len(node_types)))
    color_map = {}
    
    for i, node_type in enumerate(node_types):
        r, g, b, _ = colors[i]
        color_map[node_type] = f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
    
    # Add nodes
    for node_id, node_data in G.nodes(data=True):
        # Get node size
        size = None
        if node_size_field and node_size_field in node_data:
            size = node_data[node_size_field]
        
        # Get node color
        color = "#CCCCCC"  # Default color
        if node_color_field in node_data:
            color = color_map.get(node_data[node_color_field], "#CCCCCC")
        
        # Get node label
        label = node_data.get('label', node_id)
        
        # Create title with all attributes
        title = "<br>".join([f"{k}: {v}" for k, v in node_data.items()])
        
        # Add node to network
        net.add_node(node_id, label=label, title=title, color=color, size=size)
    
    # Add edges
    for source, target, edge_data in G.edges(data=True):
        # Get edge weight
        weight = None
        if edge_weight_field and edge_weight_field in edge_data:
            weight = edge_data[edge_weight_field]
        
        # Get edge label
        label = edge_data.get('label', '')
        
        # Create title with all attributes
        title = "<br>".join([f"{k}: {v}" for k, v in edge_data.items()])
        
        # Add edge to network
        net.add_edge(source, target, title=title, label=label, width=weight)
    
    # Configure physics
    net.barnes_hut(
        gravity=-80000,
        central_gravity=0.3,
        spring_length=250,
        spring_strength=0.001,
        damping=0.09,
        overlap=0
    )
    
    # Show buttons if requested
    if show_buttons:
        net.show_buttons(filter_=['physics'])
    
    # Save or display
    net.save_graph(output_path)
    print(f"Graph visualization saved to {output_path}")


def visualize_neo4j_results(
    results: Dict[str, Any],
    output_path: str = "neo4j_results.html",
    height: str = "750px",
    width: str = "100%",
    notebook: bool = False
) -> None:
    """
    Visualize results from Neo4j query.
    
    Args:
        results: Dict with nodes and relationships from Neo4j
        output_path: Path to save the HTML visualization
        height: Height of the visualization
        width: Width of the visualization
        notebook: Whether to display in a Jupyter notebook
    """
    # Create PyVis network
    net = Network(height=height, width=width, notebook=notebook, directed=True)
    
    # Get unique node labels for coloring
    node_labels = set()
    for node in results.get('nodes', []):
        for label in node.get('labels', []):
            node_labels.add(label)
    
    # Generate color map
    import matplotlib.cm as cm
    import numpy as np
    
    colors = cm.rainbow(np.linspace(0, 1, len(node_labels)))
    color_map = {}
    
    for i, label in enumerate(node_labels):
        r, g, b, _ = colors[i]
        color_map[label] = f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
    
    # Add nodes
    for node in results.get('nodes', []):
        # Get node color based on first label
        color = "#CCCCCC"  # Default color
        if 'labels' in node and node['labels']:
            color = color_map.get(node['labels'][0], "#CCCCCC")
        
        # Get node label
        label = node.get('text', str(node.get('id', '')))
        
        # Create title with all attributes
        title = "<br>".join([f"{k}: {v}" for k, v in node.items() 
                            if k not in ['id', 'labels', 'embedding']])
        
        # Add node to network
        net.add_node(node['id'], label=label, title=title, color=color)
    
    # Add edges
    for rel in results.get('relationships', []):
        source = rel.get('source')
        target = rel.get('target')
        rel_type = rel.get('type', '')
        
        if source is not None and target is not None:
            net.add_edge(source, target, label=rel_type)
    
    # Configure physics
    net.barnes_hut(
        gravity=-80000,
        central_gravity=0.3,
        spring_length=250,
        spring_strength=0.001,
        damping=0.09,
        overlap=0
    )
    
    # Show buttons
    net.show_buttons(filter_=['physics'])
    
    # Save or display
    net.save_graph(output_path)
    print(f"Graph visualization saved to {output_path}")


def export_graph_to_json(
    G: nx.DiGraph,
    output_path: str
) -> None:
    """
    Export a NetworkX graph to JSON format.
    
    Args:
        G: NetworkX graph
        output_path: Path to save the JSON file
    """
    # Convert graph to data structure
    data = {
        "nodes": [],
        "links": []
    }
    
    # Add nodes
    for node_id, node_data in G.nodes(data=True):
        node_entry = {"id": node_id}
        node_entry.update(node_data)
        data["nodes"].append(node_entry)
    
    # Add edges
    for source, target, edge_data in G.edges(data=True):
        edge_entry = {
            "source": source,
            "target": target
        }
        edge_entry.update(edge_data)
        data["links"].append(edge_entry)
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Graph exported to {output_path}")


def plot_graph_statistics(G: nx.DiGraph) -> None:
    """
    Plot statistics about the graph.
    
    Args:
        G: NetworkX graph
    """
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Node degree distribution
    plt.subplot(2, 2, 1)
    degrees = [d for _, d in G.degree()]
    plt.hist(degrees, bins=20)
    plt.title('Node Degree Distribution')
    plt.xlabel('Degree')
    plt.ylabel('Count')
    
    # Plot 2: Node types distribution
    plt.subplot(2, 2, 2)
    node_types = {}
    for _, data in G.nodes(data=True):
        node_type = data.get('type', 'Unknown')
        node_types[node_type] = node_types.get(node_type, 0) + 1
    
    plt.bar(node_types.keys(), node_types.values())
    plt.title('Node Types Distribution')
    plt.xlabel('Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    
    # Plot 3: Edge types distribution
    plt.subplot(2, 2, 3)
    edge_types = {}
    for _, _, data in G.edges(data=True):
        edge_type = data.get('label', 'Unknown')
        edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
    
    plt.bar(edge_types.keys(), edge_types.values())
    plt.title('Edge Types Distribution')
    plt.xlabel('Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    
    # Plot 4: Connected components
    plt.subplot(2, 2, 4)
    
    # Convert to undirected for connected components
    G_undirected = G.to_undirected()
    components = list(nx.connected_components(G_undirected))
    component_sizes = [len(c) for c in components]
    
    plt.bar(range(len(component_sizes)), sorted(component_sizes, reverse=True))
    plt.title('Connected Component Sizes')
    plt.xlabel('Component Index')
    plt.ylabel('Size')
    
    plt.tight_layout()
    plt.show()
