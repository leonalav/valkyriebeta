import torch
import torch.nn as nn
from typing import Dict, Optional, List, Union, Tuple
from .rag import EnhancedRAG, RAGConfig
from .rag_utils import KnowledgeBankManager, KnowledgeEntry
import logging
from tqdm import tqdm
import wandb

logger = logging.getLogger(__name__)

class RAGTrainer:
    """Trainer for RAG-enhanced models"""
    
    def __init__(
        self,
        model: nn.Module,
        rag_config: RAGConfig,
        knowledge_manager: KnowledgeBankManager,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        use_wandb: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model
        self.rag_config = rag_config
        self.knowledge_manager = knowledge_manager
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.use_wandb = use_wandb
        self.device = device
        
        # Training metrics
        self.metrics = {
            "train_loss": [],
            "val_loss": [],
            "retrieval_accuracy": [],
            "knowledge_usage": []
        }
        
    def train_epoch(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        epoch: int,
        val_dataloader: Optional[torch.utils.data.DataLoader] = None
    ) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_retrieval_accuracy = 0
        total_knowledge_usage = 0
        num_batches = 0
        
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch}"):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs["loss"]
            
            # Compute retrieval metrics
            if "retrieval_scores" in outputs:
                retrieval_accuracy = self._compute_retrieval_accuracy(
                    outputs["retrieval_scores"],
                    batch.get("knowledge_labels")
                )
                total_retrieval_accuracy += retrieval_accuracy
                
                knowledge_usage = outputs["retrieval_scores"].mean().item()
                total_knowledge_usage += knowledge_usage
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Log batch metrics
            if self.use_wandb:
                wandb.log({
                    "batch_loss": loss.item(),
                    "batch_retrieval_accuracy": retrieval_accuracy if "retrieval_scores" in outputs else 0,
                    "batch_knowledge_usage": knowledge_usage if "retrieval_scores" in outputs else 0
                })
        
        # Compute epoch metrics
        epoch_metrics = {
            "train_loss": total_loss / num_batches,
            "retrieval_accuracy": total_retrieval_accuracy / num_batches if "retrieval_scores" in outputs else 0,
            "knowledge_usage": total_knowledge_usage / num_batches if "retrieval_scores" in outputs else 0
        }
        
        # Validation
        if val_dataloader is not None:
            val_metrics = self.evaluate(val_dataloader)
            epoch_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
        
        # Update importance scores in knowledge bank
        if hasattr(outputs, "retrieval_scores") and batch.get("knowledge_indices") is not None:
            self.knowledge_manager.update_importance(
                batch["knowledge_indices"].tolist(),
                outputs["retrieval_scores"].max(dim=1)[0].tolist()
            )
            
        # Log epoch metrics
        self.metrics["train_loss"].append(epoch_metrics["train_loss"])
        self.metrics["retrieval_accuracy"].append(epoch_metrics["retrieval_accuracy"])
        self.metrics["knowledge_usage"].append(epoch_metrics["knowledge_usage"])
        
        if self.use_wandb:
            wandb.log(epoch_metrics)
        
        return epoch_metrics
    
    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0
        total_retrieval_accuracy = 0
        total_knowledge_usage = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                
                # Update metrics
                total_loss += outputs["loss"].item()
                
                if "retrieval_scores" in outputs:
                    retrieval_accuracy = self._compute_retrieval_accuracy(
                        outputs["retrieval_scores"],
                        batch.get("knowledge_labels")
                    )
                    total_retrieval_accuracy += retrieval_accuracy
                    
                    knowledge_usage = outputs["retrieval_scores"].mean().item()
                    total_knowledge_usage += knowledge_usage
                
                num_batches += 1
        
        # Compute metrics
        metrics = {
            "loss": total_loss / num_batches,
            "retrieval_accuracy": total_retrieval_accuracy / num_batches if "retrieval_scores" in outputs else 0,
            "knowledge_usage": total_knowledge_usage / num_batches if "retrieval_scores" in outputs else 0
        }
        
        # Update validation metrics
        self.metrics["val_loss"].append(metrics["loss"])
        
        return metrics
    
    def _compute_retrieval_accuracy(
        self,
        retrieval_scores: torch.Tensor,
        knowledge_labels: Optional[torch.Tensor]
    ) -> float:
        """Compute accuracy of knowledge retrieval"""
        if knowledge_labels is None:
            return 0.0
            
        # Get top retrieved items
        top_retrievals = retrieval_scores.argmax(dim=-1)
        
        # Compute accuracy
        correct = (top_retrievals == knowledge_labels).float().mean()
        return correct.item()
    
    def save_checkpoint(self, path: str, epoch: int) -> None:
        """Save training checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": self.metrics
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
            
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path: str) -> int:
        """Load training checkpoint"""
        checkpoint = torch.load(path)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            
        self.metrics = checkpoint["metrics"]
        
        return checkpoint["epoch"]