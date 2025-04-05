import torch
from config.model_config import ModelConfig
from model.model import LogicalReasoningTransformer
from training.trainer import LogicalReasoningTrainer
from utils.optimization import apply_model_optimizations
from transformers import AutoTokenizer
import logging

def main():
    # Initialize logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = ModelConfig()
    
    # Initialize tokenizer with Gemma-specific handling
    tokenizer = AutoTokenizer.from_pretrained('google/gemma-2b-27b-it')
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize model
    model = LogicalReasoningTransformer(config)
    
    # Apply optimizations
    model = apply_model_optimizations(model, config)
    
    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Initialize trainer
    trainer = LogicalReasoningTrainer(
        model=model,
        config=config,
        train_dataset=None,  # Add your training dataset here
        val_dataset=None,    # Add your validation dataset here
        tokenizer=tokenizer
    )
    
    # Start training
    logger.info("Starting training...")
    trainer.train(num_epochs=config.num_epochs)
    
    # Save final model
    torch.save(model.state_dict(), "final_model.pt")
    logger.info("Training completed!")

if __name__ == "__main__":
    main()