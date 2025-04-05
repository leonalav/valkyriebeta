import logging
from typing import Optional, Callable
from functools import wraps

class ProductionErrorHandler:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def handle_training_error(self, error: Exception, context: dict):
        """Handle training-specific errors"""
        self.logger.error(f"Training error: {str(error)}", exc_info=True, extra=context)
        
    def handle_inference_error(self, error: Exception, context: dict):
        """Handle inference-specific errors"""
        self.logger.error(f"Inference error: {str(error)}", exc_info=True, extra=context)

def production_error_handler(func: Callable):
    """Decorator for production error handling"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            handler = ProductionErrorHandler()
            context = {'function': func.__name__, 'args': args, 'kwargs': kwargs}
            handler.handle_training_error(e, context)
            raise
    return wrapper
