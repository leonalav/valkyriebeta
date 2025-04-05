from typing import List, Dict, Any, Optional
import torch
import numpy as np
from dataclasses import dataclass
from sklearn.metrics import precision_recall_fscore_support
from nltk.translate.bleu_score import corpus_bleu
from scipy import stats

@dataclass
class LogicalMetrics:
    """Metrics for logical reasoning evaluation"""
    accuracy: float
    consistency: float
    validity: float
    completeness: float
    structural_similarity: float
    confidence: Optional[float] = None
    metadata: Optional[Dict] = None

class MetricsCalculator:
    """Calculate evaluation metrics for logical reasoning"""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Reset all metrics"""
        self.total_examples = 0
        self.correct_predictions = 0
        self.total_loss = 0.0
        self.logical_errors = 0
        self.tree_errors = 0
        self.predictions = []
        self.true_labels = []
        self.predicted_sentences = []
        self.reference_sentences = []
        
    def update(self, 
               predictions: torch.Tensor,
               labels: torch.Tensor,
               loss: Optional[float] = None,
               logical_tree_pred: Optional[Dict] = None,
               logical_tree_true: Optional[Dict] = None,
               predicted_sentences: Optional[List[str]] = None,
               reference_sentences: Optional[List[List[str]]] = None):
        """Update metrics with batch results"""
        # Update basic metrics
        self.total_examples += labels.size(0)
        self.correct_predictions += (predictions == labels).sum().item()
        
        if loss is not None:
            self.total_loss += loss
            
        # Store predictions and labels for precision/recall calculation
        self.predictions.extend(predictions.cpu().numpy())
        self.true_labels.extend(labels.cpu().numpy())
        
        # Store sentences for BLEU score calculation
        if predicted_sentences and reference_sentences:
            self.predicted_sentences.extend(predicted_sentences)
            self.reference_sentences.extend(reference_sentences)
        
        # Check logical consistency if trees are provided
        if logical_tree_pred and logical_tree_true:
            self.logical_errors += self._count_logical_errors(
                logical_tree_pred, logical_tree_true
            )
            self.tree_errors += self._count_tree_structure_errors(
                logical_tree_pred, logical_tree_true
            )
    
    def compute(self) -> LogicalMetrics:
        """Compute final metrics"""
        accuracy = self.correct_predictions / self.total_examples
        
        # Calculate precision, recall, and F1 score
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.true_labels,
            self.predictions,
            average='weighted'
        )
        
        # Calculate logical consistency
        logical_consistency = 1.0
        if self.logical_errors > 0:
            logical_consistency = 1.0 - (self.logical_errors / self.total_examples)
            
        # Calculate tree accuracy if applicable
        tree_accuracy = None
        if hasattr(self, 'tree_errors'):
            tree_accuracy = 1.0 - (self.tree_errors / self.total_examples)
            
        # Calculate perplexity if loss is available
        perplexity = None
        if self.total_loss > 0:
            perplexity = torch.exp(torch.tensor(self.total_loss / self.total_examples)).item()
        
        # Calculate BLEU score
        bleu = None
        if self.predicted_sentences and self.reference_sentences:
            bleu = corpus_bleu(self.reference_sentences, self.predicted_sentences)
        
        return LogicalMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            logical_consistency=logical_consistency,
            tree_accuracy=tree_accuracy,
            perplexity=perplexity,
            bleu=bleu
        )
    
    def _count_logical_errors(self, pred_tree: Dict, true_tree: Dict) -> int:
        """Count logical consistency errors in predicted tree"""
        def check_consistency(node):
            if not isinstance(node, dict):
                return 0
                
            errors = 0
            operation = node.get('operation')
            arguments = node.get('arguments', [])
            
            # Check operation validity
            if operation == 'AND':
                if not all(arguments):
                    errors += 1
            elif operation == 'OR':
                if not any(arguments):
                    errors += 1
            elif operation == 'NOT':
                if len(arguments) != 1:
                    errors += 1
                    
            # Recursively check children
            for arg in arguments:
                errors += check_consistency(arg)
                
            return errors
            
        return check_consistency(pred_tree)
    
    def _count_tree_structure_errors(self, pred_tree: Dict, true_tree: Dict) -> int:
        """Count structural errors between predicted and true trees"""
        def compare_trees(pred, true):
            if not isinstance(pred, dict) or not isinstance(true, dict):
                return int(pred != true)
                
            errors = 0
            
            # Compare operations
            if pred.get('operation') != true.get('operation'):
                errors += 1
                
            # Compare arguments recursively
            pred_args = pred.get('arguments', [])
            true_args = true.get('arguments', [])
            
            if len(pred_args) != len(true_args):
                errors += 1
            else:
                for p_arg, t_arg in zip(pred_args, true_args):
                    errors += compare_trees(p_arg, t_arg)
                    
            return errors
            
        return compare_trees(pred_tree, true_tree)
    
    def calculate_tree_metrics(self,
                             predicted_trees: List[Dict],
                             target_trees: List[Dict]) -> LogicalMetrics:
        """Calculate metrics for logical reasoning trees"""
        num_samples = len(predicted_trees)
        
        # Calculate accuracy
        accuracy = sum(self._tree_equals(p, t) 
                      for p, t in zip(predicted_trees, target_trees)) / num_samples
        
        # Calculate logical consistency
        consistency = sum(self._is_consistent(t) 
                         for t in predicted_trees) / num_samples
        
        # Calculate logical validity
        validity = sum(self._is_valid(p, t)
                      for p, t in zip(predicted_trees, target_trees)) / num_samples
        
        # Calculate completeness
        completeness = sum(self._is_complete(p)
                          for p in predicted_trees) / num_samples
        
        # Calculate structural similarity
        similarities = [self._tree_similarity(p, t)
                       for p, t in zip(predicted_trees, target_trees)]
        structural_similarity = sum(similarities) / num_samples
        
        # Calculate confidence if available
        confidence = None
        if all(hasattr(t, 'confidence') for t in predicted_trees):
            confidence = sum(t.confidence for t in predicted_trees) / num_samples
            
        return LogicalMetrics(
            accuracy=accuracy,
            consistency=consistency,
            validity=validity,
            completeness=completeness,
            structural_similarity=structural_similarity,
            confidence=confidence
        )

    def _tree_equals(self, tree1: Dict, tree2: Dict) -> bool:
        """Check if two logical trees are equivalent"""
        if not (tree1 and tree2):
            return False
            
        if tree1['operation'] != tree2['operation']:
            return False
            
        if len(tree1['arguments']) != len(tree2['arguments']):
            return False
            
        return all(self._tree_equals(a1, a2) 
                  for a1, a2 in zip(tree1['arguments'], tree2['arguments']))
                  
    def _is_consistent(self, tree: Dict) -> bool:
        """Check if a logical tree is internally consistent"""
        # Check operation validity
        if not tree.get('operation') in {'AND', 'OR', 'NOT', 'IMPLIES', 'IFF'}:
            return False
            
        # Check arguments validity
        args = tree.get('arguments', [])
        if tree['operation'] == 'NOT' and len(args) != 1:
            return False
        elif tree['operation'] in {'AND', 'OR', 'IMPLIES', 'IFF'} and len(args) != 2:
            return False
            
        # Recursively check arguments
        return all(self._is_consistent(arg) for arg in args if isinstance(arg, dict))
        
    def _is_valid(self, predicted: Dict, target: Dict) -> bool:
        """Check if predicted tree is logically valid given target"""
        # Implement logical validation rules
        pass
        
    def _is_complete(self, tree: Dict) -> bool:
        """Check if logical tree is complete"""
        # Implement completeness validation
        pass
        
    def _tree_similarity(self, tree1: Dict, tree2: Dict) -> float:
        """Calculate structural similarity between trees"""
        # Implement tree similarity metric
        pass