import logging
import logging.config
import json
from pathlib import Path
from typing import Optional, Dict, Any
import os
from logging.handlers import RotatingFileHandler
import socket

class StructuredLogFormatter(logging.Formatter):
    """
    Formatter that emits logs in JSON format with standardized fields.
    
    Adds:
    - Timestamp in ISO8601 format
    - Hostname
    - Service name (from env or default)
    - Standard log severity levels
    - Contextual metadata
    """
    
    def __init__(self):
        super().__init__()
        self.hostname = socket.gethostname()
        self.service_name = os.getenv('SERVICE_NAME', 'valkyrie-llm')
        
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record into JSON"""
        log_data = {
            'timestamp': self.formatTime(record, '%Y-%m-%dT%H:%M:%S.%fZ'),
            'level': record.levelname,
            'service': self.service_name,
            'host': self.hostname,
            'logger': record.name,
            'message': record.getMessage(),
            'correlation_id': getattr(record, 'correlation_id', None),
            'thread': record.thread,
            'process': record.process,
        }
        
        # Include exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
            
        # Include any extra context
        if hasattr(record, 'context'):
            log_data.update(record.context)
            
        return json.dumps(log_data)

def configure_logging(
    log_level: str = 'INFO',
    log_file: Optional[str] = None,
    rotate_logs: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    external_logging_config: Optional[Dict[str, Any]] = None,
    elk_config: Optional[Dict[str, Any]] = None
) -> logging.Logger:
    """
    Configure application logging with structured format and optional file output.
    
    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        rotate_logs: Whether to enable log rotation
        max_bytes: Max log file size before rotation (if rotation enabled)
        backup_count: Number of backup logs to keep (if rotation enabled)
        external_logging_config: Additional logging configuration for external services
        
    Returns:
        Root logger instance
    """
    # Clear any existing handlers
    logging.root.handlers = []
    
    # Create formatter
    formatter = StructuredLogFormatter()
    
    # Configure console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    handlers = [console_handler]
    
    # Configure file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        if rotate_logs:
            file_handler = RotatingFileHandler(
                filename=log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
        else:
            file_handler = logging.FileHandler(
                filename=log_file,
                encoding='utf-8'
            )
            
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configure external logging if specified
    if external_logging_config:
        if external_logging_config.get('cloudwatch'):
            try:
                from watchtower import CloudWatchLogHandler
                cw_handler = CloudWatchLogHandler(
                    log_group=external_logging_config['cloudwatch']['log_group'],
                    stream_name=external_logging_config['cloudwatch'].get('stream_name', 'default'),
                    create_log_group=True
                )
                cw_handler.setFormatter(formatter)
                handlers.append(cw_handler)
            except ImportError:
                logging.warning("CloudWatch logging requested but watchtower not installed")
                
    # Configure ELK logging if specified
    if elk_config:
        try:
            from python_logstash import LogstashHandler
            elk_handler = LogstashHandler(
                host=elk_config['host'],
                port=elk_config['port'],
                version=1,
                tags=elk_config.get('tags', []),
                ssl_enable=elk_config.get('ssl', False)
            )
            elk_handler.setFormatter(formatter)
            handlers.append(elk_handler)
        except ImportError:
            logging.warning("ELK logging requested but python-logstash not installed")
    
    # Apply configuration
    logging.basicConfig(
        level=log_level.upper(),
        handlers=handlers,
        force=True
    )
    
    # Get root logger
    logger = logging.getLogger()
    logger.info("Logging configured", extra={
        'context': {
            'log_level': log_level,
            'log_file': log_file,
            'rotate_logs': rotate_logs,
            'external_logging': bool(external_logging_config)
        }
    })
    
    return logger

def get_logger(name: str = None) -> logging.Logger:
    """
    Get a configured logger instance with structured logging support.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)

# Default configuration when imported
default_logger = configure_logging()