import torch
from typing import Dict, Any, Optional
from ..config.model_config import ModelConfig
from ..utils.monitoring import ModelMonitor
from ..utils.resource_manager import ResourceManager
from ..security.validator import ModelValidator
from ..utils.checkpoint_manager import CheckpointManager
from ..utils.customization import ModelCustomizer
from ..utils.feedback_collector import FeedbackCollector
from ..utils.compliance_checker import ComplianceChecker
from ..training.distributed_handler import DistributedHandler

class ModelPipeline:
    def __init__(self, config: ModelConfig):
        self.config = config
        
        # Initialize components
        self.monitor = ModelMonitor()
        self.resource_manager = ResourceManager(config.resource_config)
        self.validator = ModelValidator(config.security_config)
        self.checkpoint_manager = CheckpointManager(config.checkpoint_dir)
        self.customizer = ModelCustomizer(config.customization_config)
        self.feedback_collector = FeedbackCollector(config.feedback_dir)
        self.compliance_checker = ComplianceChecker(config.compliance_config)
        
        # Initialize distributed training if config provided
        self.distributed_handler = None
        if config.distributed_config:
            self.distributed_handler = DistributedHandler(config.distributed_config)

    def train(self, model: torch.nn.Module, 
              train_dataloader: torch.utils.data.DataLoader,
              num_epochs: int) -> Dict[str, Any]:
        """Main training loop with all components integrated"""
        
        # Setup distributed training if enabled
        if self.distributed_handler:
            self.distributed_handler.setup()
            model = torch.nn.parallel.DistributedDataParallel(model)

        # Optimize model and create training setup
        model = self.resource_manager.optimize_model(model)
        training_setup = self.customizer.create_fine_tuning_setup(model)
        
        metrics = {}
        try:
            for epoch in range(num_epochs):
                epoch_metrics = self._train_epoch(
                    model, 
                    train_dataloader,
                    training_setup["optimizer"],
                    training_setup["scheduler"]
                )
                
                # Save checkpoint
                if self.distributed_handler is None or self.distributed_handler.is_main_process:
                    self.checkpoint_manager.save_checkpoint(
                        model,
                        training_setup["optimizer"],
                        training_setup["scheduler"],
                        epoch_metrics,
                        epoch
                    )
                
                metrics[f"epoch_{epoch}"] = epoch_metrics
                
        finally:
            if self.distributed_handler:
                self.distributed_handler.cleanup()
                
        return metrics

    def inference(self, model: torch.nn.Module, input_data: Any) -> Dict[str, Any]:
        """Main inference pipeline with all components integrated"""
        
        # Validate input
        is_valid, message = self.validator.validate_input(input_data)
        if not is_valid:
            raise ValueError(message)
            
        # Check compliance
        compliance_results = self.compliance_checker.check_compliance(input_data)
        if compliance_results["contains_pii"]:
            raise ValueError("Input contains PII data")
            
        # Record start time for monitoring
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        try:
            start_time.record()
            
            # Run inference
            with torch.no_grad():
                output = model(input_data)
                
            end_time.record()
            torch.cuda.synchronize()
            
            # Record metrics
            self.monitor.record_inference(start_time.elapsed_time(end_time), True)
            
            # Collect feedback (async)
            self.feedback_collector.store_feedback(
                input_data=input_data,
                output=output,
                feedback_score=1.0  # Default score
            )
            
            return {
                "output": output,
                "metrics": self.monitor.get_resource_usage(),
                "compliance": compliance_results
            }
            
        except Exception as e:
            self.monitor.record_inference(start_time.elapsed_time(end_time), False)
            raise e
        finally:
            self.resource_manager.clear_cache()

    def _train_epoch(self, 
                     model: torch.nn.Module,
                     dataloader: torch.utils.data.DataLoader,
                     optimizer: torch.optim.Optimizer,
                     scheduler: torch.optim.lr_scheduler._LRScheduler) -> Dict[str, float]:
        """Single training epoch with monitoring and resource management"""
        model.train()
        epoch_metrics = {}
        
        for batch in dataloader:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Update metrics
            if self.distributed_handler:
                epoch_metrics = self.distributed_handler.reduce_metrics({"loss": loss})
            else:
                epoch_metrics = {"loss": loss.item()}
                
        return epoch_metrics
