import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
from tqdm import tqdm
import logging
from .metrics import MetricsCalculator, LogicalMetrics
from data.dataset import LogicalReasoningDataset

class LogicalReasoningEvaluator:
    def __init__(self, 
                 model: nn.Module,
                 tokenizer: Any,
                 device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.metrics = MetricsCalculator()
        self.logger = logging.getLogger(__name__)
        
    def evaluate(self, 
                 eval_dataset: LogicalReasoningDataset,
                 batch_size: int = 16) -> LogicalMetrics:
        """Evaluate model on dataset"""
        self.model.eval()
        self.metrics.reset()
        
        dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                
                # Get predictions
                predictions = outputs.logits.argmax(dim=-1) if hasattr(outputs, 'logits') else outputs.argmax(dim=-1)
                
                # Update metrics
                self.metrics.update(
                    predictions=predictions,
                    labels=batch['labels'],
                    loss=outputs.loss.item() if hasattr(outputs, 'loss') else None,
                    logical_tree_pred=batch.get('logical_tree'),
                    logical_tree_true=batch.get('labels_tree')
                )
        
        # Compute and return final metrics
        return self.metrics.compute()
    
    def evaluate_example(self, 
                        text: str,
                        true_tree: Optional[Dict] = None) -> Dict[str, Any]:
        self.model.eval()
        
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            # Forward pass
            outputs = self.model(**inputs)
            
            # Get predictions
            predictions = outputs.logits.argmax(dim=-1) if hasattr(outputs, 'logits') else outputs.argmax(dim=-1)
            
            # Decode prediction
            predicted_text = self.tokenizer.decode(predictions[0], skip_special_tokens=True)
            
            result = {
                'input_text': text,
                'predicted_text': predicted_text,
                'confidence': torch.softmax(outputs.logits, dim=-1).max().item()
                if hasattr(outputs, 'logits') else None
            }
            
            # Evaluate logical tree if provided
            if true_tree is not None and hasattr(outputs, 'tree_predictions'):
                tree_pred = outputs.tree_predictions[0]
                tree_errors = self.metrics._count_tree_structure_errors(tree_pred, true_tree)
                result['tree_accuracy'] = 1.0 - (tree_errors / len(true_tree))
                
            return result
    
    def analyze_errors(self, 
                      eval_dataset: LogicalReasoningDataset,
                      num_examples: int = 10) -> List[Dict[str, Any]]:
        self.model.eval()
        error_cases = []
        
        dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=1,
            shuffle=True
        )
        
        with torch.no_grad():
            for batch in dataloader:
                if len(error_cases) >= num_examples:
                    break
                    
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                predictions = outputs.logits.argmax(dim=-1) if hasattr(outputs, 'logits') else outputs.argmax(dim=-1)
                
                # Check for errors
                if not torch.all(predictions == batch['labels']):
                    error_case = {
                        'input_text': self.tokenizer.decode(batch['input_ids'][0], skip_special_tokens=True),
                        'predicted': self.tokenizer.decode(predictions[0], skip_special_tokens=True),
                        'true': self.tokenizer.decode(batch['labels'][0], skip_special_tokens=True),
                        'logical_tree': batch.get('logical_tree')
                    }
                    error_cases.append(error_case)
        
        return error_cases 