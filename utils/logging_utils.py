import logging
import sys
from typing import Optional, Dict, Any
import json
from pathlib import Path
import wandb
from datetime import datetime
import torch
import os

def setup_logging(args=None, log_dir: str = "logs", log_level=logging.INFO):
    """Setup basic logging configuration for the application.
    
    Args:
        args: Argument namespace that might contain logging specific arguments
        log_dir: Directory to store log files
        log_level: Logging level
        
    Returns:
        Path to the log file
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Determine log file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = getattr(args, 'experiment_name', 'Valkyrie') if args else 'Valkyrie'
    log_file = Path(log_dir) / f"{experiment_name}_{timestamp}.log"
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicate logs
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatters and handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    logging.info(f"Logging initialized. Log file: {log_file}")
    return str(log_file)

class LoggerManager:
    def __init__(self, 
                 experiment_name: str,
                 log_dir: str = "logs",
                 use_wandb: bool = True,
                 wandb_project: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None):
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.use_wandb = use_wandb
        self.config = config or {}
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize W&B if enabled
        if use_wandb and wandb_project:
            try:
                wandb.init(
                    project=wandb_project,
                    name=experiment_name,
                    config=config,
                    dir=str(self.log_dir)
                )
            except Exception as e:
                self.logger.warning(f"Failed to initialize W&B: {e}")
                self.use_wandb = False
    
    def _setup_logging(self):
        """Setup logging configuration"""
        self.logger = logging.getLogger(self.experiment_name)
        self.logger.setLevel(logging.INFO)
        
        # Create formatters and handlers
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler(
            self.log_dir / f"{self.experiment_name}_{datetime.now():%Y%m%d_%H%M%S}.log"
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to all configured outputs"""
        # Log to file
        metrics_str = " - ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Step {step}: {metrics_str}" if step else metrics_str)
        
        # Log to W&B
        if self.use_wandb:
            wandb.log(metrics, step=step)
    
    def log_model_graph(self, model: torch.nn.Module, input_size: tuple):
        """Log model architecture graph"""
        try:
            if self.use_wandb:
                wandb.watch(model, log="all")
        except Exception as e:
            self.logger.warning(f"Failed to log model graph: {e}")
    
    def log_config(self):
        """Log configuration parameters"""
        self.logger.info("Configuration:")
        self.logger.info(json.dumps(self.config, indent=2))
    
    def log_error(self, error: Exception, additional_info: Optional[Dict] = None):
        """Log error with additional context"""
        error_msg = f"Error: {str(error)}"
        if additional_info:
            error_msg += f"\nAdditional Info: {json.dumps(additional_info, indent=2)}"
        self.logger.error(error_msg)
    
    def close(self):
        """Cleanup logging resources"""
        if self.use_wandb:
            wandb.finish()
        
        # Close file handlers
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler) 