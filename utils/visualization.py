import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
import numpy as np
from pathlib import Path
import networkx as nx

class AttentionVisualizer:
    def __init__(self, save_dir: str = "visualizations"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_attention_weights(self,
                             attention_weights: torch.Tensor,
                             tokens: List[str],
                             layer_idx: int,
                             head_idx: Optional[int] = None,
                             save_path: Optional[str] = None):
        """Plot attention weight heatmap"""
        plt.figure(figsize=(10, 8))
        
        # Average across heads if head_idx is not specified
        if head_idx is None:
            weights = attention_weights.mean(dim=0)
            title = f"Average Attention Weights (Layer {layer_idx})"
        else:
            weights = attention_weights[head_idx]
            title = f"Attention Weights (Layer {layer_idx}, Head {head_idx})"
        
        # Create heatmap
        sns.heatmap(
            weights.cpu().numpy(),
            xticklabels=tokens,
            yticklabels=tokens,
            cmap='viridis',
            annot=True,
            fmt='.2f'
        )
        
        plt.title(title)
        plt.xlabel("Key Tokens")
        plt.ylabel("Query Tokens")
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        
        if save_path:
            plt.savefig(self.save_dir / save_path, bbox_inches='tight')
        plt.close()
    
    def plot_attention_flow(self,
                          attention_weights: List[torch.Tensor],
                          tokens: List[str],
                          save_path: Optional[str] = None):
        """Plot attention flow across layers"""
        num_layers = len(attention_weights)
        fig, axes = plt.subplots(1, num_layers, figsize=(5*num_layers, 6))
        
        for i, weights in enumerate(attention_weights):
            avg_weights = weights.mean(dim=0)
            sns.heatmap(
                avg_weights.cpu().numpy(),
                ax=axes[i],
                xticklabels=tokens if i == 0 else [],
                yticklabels=tokens,
                cmap='viridis'
            )
            axes[i].set_title(f"Layer {i+1}")
            
        plt.tight_layout()
        if save_path:
            plt.savefig(self.save_dir / save_path, bbox_inches='tight')
        plt.close()

class LogicalTreeVisualizer:
    def __init__(self, save_dir: str = "visualizations"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_logical_tree(self,
                         tree: Dict,
                         save_path: Optional[str] = None,
                         highlight_nodes: Optional[List[str]] = None):
        """Visualize logical reasoning tree"""
        G = nx.DiGraph()
        pos = {}
        labels = {}
        
        def add_nodes(node: Dict, parent_id: Optional[str] = None, level: int = 0, pos_x: float = 0):
            if isinstance(node, str):
                node_id = f"leaf_{len(G.nodes)}"
                G.add_node(node_id)
                labels[node_id] = node
                pos[node_id] = (pos_x, -level)
                if parent_id:
                    G.add_edge(parent_id, node_id)
                return 1
            
            node_id = f"op_{len(G.nodes)}"
            G.add_node(node_id)
            labels[node_id] = node['operation']
            
            if parent_id:
                G.add_edge(parent_id, node_id)
            
            width = 0
            for arg in node['arguments']:
                w = add_nodes(arg, node_id, level + 1, pos_x + width)
                width += w
                
            pos[node_id] = (pos_x + width/2 - 0.5, -level)
            return width
        
        add_nodes(tree)
        
        plt.figure(figsize=(12, 8))
        nx.draw(
            G,
            pos=pos,
            labels=labels,
            with_labels=True,
            node_color='lightblue',
            node_size=2000,
            arrowsize=20,
            font_size=10,
            font_weight='bold'
        )
        
        if highlight_nodes:
            nx.draw_networkx_nodes(
                G,
                pos=pos,
                nodelist=[n for n in G.nodes if labels[n] in highlight_nodes],
                node_color='lightgreen',
                node_size=2000
            )
        
        plt.title("Logical Reasoning Tree")
        if save_path:
            plt.savefig(self.save_dir / save_path, bbox_inches='tight')
        plt.close()
    
    def compare_trees(self,
                     pred_tree: Dict,
                     true_tree: Dict,
                     save_path: Optional[str] = None):
        """Compare predicted and true logical trees"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot predicted tree
        self.plot_logical_tree(pred_tree)
        plt.sca(ax1)
        plt.title("Predicted Tree")
        
        # Plot true tree
        self.plot_logical_tree(true_tree)
        plt.sca(ax2)
        plt.title("True Tree")
        
        plt.tight_layout()
        if save_path:
            plt.savefig(self.save_dir / save_path, bbox_inches='tight')
        plt.close() 