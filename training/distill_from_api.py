import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
import logging
from tqdm import tqdm
import wandb
from ..model.api_distillation import APITeacherModel, APIKnowledgeDistillation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APIDistillationTrainer:
    """Trainer for API-based knowledge distillation"""
    
    def __init__(
        self,
        student_model: torch.nn.Module,
        tokenizer,
        api_key: str,
        site_url: str,
        site_name: str,
        config: Dict,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.student_model = student_model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.config = config
        
        # Initialize teacher model
        self.teacher_model = APITeacherModel(
            api_key=api_key,
            site_url=site_url,
            site_name=site_name,
            model=config.get("teacher_model", "deepseek/deepseek-r1:free"),
            temperature=config.get("temperature", 0.85),
            top_p=config.get("top_p", 1.0)
        )
        
        # Initialize distillation module
        self.distillation = APIKnowledgeDistillation(
            config=config,
            teacher_model=self.teacher_model,
            tokenizer=tokenizer,
            hidden_size=config.hidden_size,
            temperature=config.get("distillation_temperature", 2.0)
        ).to(device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=config.get("learning_rate", 1e-4),
            weight_decay=config.get("weight_decay", 0.01)
        )
        
        # Initialize learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get("num_epochs", 10)
        )
        
        # Initialize wandb for tracking
        if config.get("use_wandb", False):
            wandb.init(
                project=config.get("wandb_project", "api_distillation"),
                config=config
            )
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        alpha: float = 0.5
    ) -> Dict[str, float]:
        """Train for one epoch"""
        self.student_model.train()
        total_loss = 0
        total_distill_loss = 0
        total_student_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in progress_bar:
            # Get input texts and move tensors to device
            input_texts = batch["text"]
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            
            # Get student model outputs
            student_outputs = self.student_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            student_logits = student_outputs.logits
            
            # Perform distillation
            distill_outputs = self.distillation(
                student_logits=student_logits,
                input_texts=input_texts,
                alpha=alpha
            )
            
            loss = distill_outputs["loss"]
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), 1.0)
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            total_distill_loss += distill_outputs["distillation_loss"].item()
            total_student_loss += distill_outputs["student_loss"].item()
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": loss.item(),
                "distill_loss": distill_outputs["distillation_loss"].item(),
                "student_loss": distill_outputs["student_loss"].item()
            })
        
        # Calculate average losses
        avg_loss = total_loss / len(train_loader)
        avg_distill_loss = total_distill_loss / len(train_loader)
        avg_student_loss = total_student_loss / len(train_loader)
        
        return {
            "loss": avg_loss,
            "distillation_loss": avg_distill_loss,
            "student_loss": avg_student_loss
        }
    
    def train(
        self,
        train_loader: DataLoader,
        num_epochs: int,
        alpha: float = 0.5,
        eval_loader: Optional[DataLoader] = None
    ):
        """Full training loop with optional evaluation"""
        best_loss = float("inf")
        
        for epoch in range(num_epochs):
            # Training
            train_metrics = self.train_epoch(train_loader, epoch, alpha)
            
            # Evaluation
            if eval_loader is not None:
                eval_metrics = self.evaluate(eval_loader)
                logger.info(f"Epoch {epoch} - Eval Loss: {eval_metrics['loss']:.4f}")
                
                # Save best model
                if eval_metrics["loss"] < best_loss:
                    best_loss = eval_metrics["loss"]
                    torch.save(
                        self.student_model.state_dict(),
                        f"{self.config.get('output_dir', '.')}/best_model.pt"
                    )
            
            # Update learning rate
            self.scheduler.step()
            
            # Log metrics
            if self.config.get("use_wandb", False):
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "train_distill_loss": train_metrics["distillation_loss"],
                    "train_student_loss": train_metrics["student_loss"],
                    "learning_rate": self.scheduler.get_last_lr()[0]
                })
                if eval_loader is not None:
                    wandb.log({
                        "eval_loss": eval_metrics["loss"]
                    })
            
            logger.info(
                f"Epoch {epoch} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Distill Loss: {train_metrics['distillation_loss']:.4f}, "
                f"Student Loss: {train_metrics['student_loss']:.4f}"
            )
    
    @torch.no_grad()
    def evaluate(self, eval_loader: DataLoader) -> Dict[str, float]:
        """Evaluate the model"""
        self.student_model.eval()
        total_loss = 0
        
        for batch in tqdm(eval_loader, desc="Evaluating"):
            input_texts = batch["text"]
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            
            student_outputs = self.student_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            distill_outputs = self.distillation(
                student_logits=student_outputs.logits,
                input_texts=input_texts
            )
            
            total_loss += distill_outputs["loss"].item()
        
        return {"loss": total_loss / len(eval_loader)}
