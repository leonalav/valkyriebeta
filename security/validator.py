from typing import Any, Dict, List, Optional, Union
import re
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("security.validator")

@dataclass
class ValidationResult:
    """Result of an input validation check"""
    is_valid: bool
    message: str = ""
    details: Optional[Dict[str, Any]] = None

class InputValidator:
    """Validates user input for LLM inference to ensure security and proper formatting"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the validator with optional configuration
        
        Args:
            config_path: Path to JSON configuration file with validation rules
        """
        # Default configuration
        self.config = {
            "max_input_length": 32768,  # Maximum input length in characters
            "min_input_length": 1,      # Minimum input length in characters
            "enable_profanity_filter": True,
            "enable_prompt_injection_filter": True,
            "block_scripts": True,
            "max_repetitions": 20,      # Maximum number of repeated characters
            "check_language": True      # Whether to check for supported languages
        }
        
        # Load configuration from file if provided
        if config_path:
            self._load_config(config_path)
            
        # Load filter lists based on configuration
        self._load_filter_lists()
        
        logger.info(f"Input validator initialized with config: {self.config}")
        
    def _load_config(self, config_path: str) -> None:
        """Load configuration from JSON file
        
        Args:
            config_path: Path to JSON configuration file
        """
        try:
            with open(config_path, 'r') as f:
                file_config = json.load(f)
                
            # Update configuration with file values
            for key, value in file_config.items():
                if key in self.config:
                    self.config[key] = value
                    
            logger.info(f"Loaded validator configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading validator configuration: {str(e)}")
            
    def _load_filter_lists(self) -> None:
        """Load filter word lists and patterns based on configuration"""
        # Initialize filter lists
        self.profanity_patterns = []
        self.injection_patterns = []
        self.script_patterns = []
        self.supported_language_patterns = []
        
        # Basic profanity filter if enabled
        if self.config["enable_profanity_filter"]:
            # This is a very basic list - in production you would use a comprehensive library
            # or service for content moderation
            self.profanity_patterns = [
                re.compile(r'\b(fuck|shit|ass|damn|bitch)\b', re.IGNORECASE),
                # Add more patterns as needed
            ]
            
        # Prompt injection patterns if enabled
        if self.config["enable_prompt_injection_filter"]:
            self.injection_patterns = [
                re.compile(r'ignore previous instructions', re.IGNORECASE),
                re.compile(r'ignore all instructions', re.IGNORECASE),
                re.compile(r'disregard (previous|prior|earlier) instructions', re.IGNORECASE),
                re.compile(r'forget (previous|prior|earlier) instructions', re.IGNORECASE),
                re.compile(r'you are now', re.IGNORECASE),
                # Add more patterns as needed
            ]
            
        # Script patterns if blocking scripts
        if self.config["block_scripts"]:
            self.script_patterns = [
                re.compile(r'<script.*?>.*?</script>', re.IGNORECASE | re.DOTALL),
                re.compile(r'<\?php.*?\?>', re.IGNORECASE | re.DOTALL),
                re.compile(r'import os.*?;.*?os\.(system|exec|spawn|popen)', re.DOTALL),
                re.compile(r'require\([\'"].*?[\'"]', re.IGNORECASE),
                # Add more patterns as needed
            ]
            
        # Language detection patterns
        if self.config["check_language"]:
            # These are very basic patterns - in production use a proper language detection library
            self.supported_language_patterns = [
                ("english", re.compile(r'^[a-zA-Z\s\.,\?\!]+$')),
                ("contains_latin", re.compile(r'[a-zA-Z]')),
                # Add more language patterns as needed
            ]
    
    def validate_input(self, text: str) -> ValidationResult:
        """Validate user input for security and proper formatting
        
        Args:
            text: The input text to validate
            
        Returns:
            ValidationResult indicating if the input is valid and any error details
        """
        # Check for empty input
        if not text or not text.strip():
            return ValidationResult(
                is_valid=False,
                message="Input text cannot be empty",
                details={"error": "empty_input"}
            )
            
        # Check input length
        if len(text) < self.config["min_input_length"]:
            return ValidationResult(
                is_valid=False,
                message=f"Input text too short (minimum {self.config['min_input_length']} characters)",
                details={"error": "input_too_short", "length": len(text)}
            )
            
        if len(text) > self.config["max_input_length"]:
            return ValidationResult(
                is_valid=False,
                message=f"Input text too long (maximum {self.config['max_input_length']} characters)",
                details={"error": "input_too_long", "length": len(text)}
            )
            
        # Check for excessive repetition
        max_repetitions = self.config["max_repetitions"]
        repetition_pattern = re.compile(r'(.)\1{' + str(max_repetitions) + ',}')
        if repetition_pattern.search(text):
            return ValidationResult(
                is_valid=False,
                message=f"Input contains excessive repetition",
                details={"error": "excessive_repetition"}
            )
            
        # Check profanity if enabled
        if self.config["enable_profanity_filter"]:
            for pattern in self.profanity_patterns:
                match = pattern.search(text)
                if match:
                    return ValidationResult(
                        is_valid=False,
                        message="Input contains prohibited content",
                        details={"error": "profanity"}
                    )
                    
        # Check for prompt injection if enabled
        if self.config["enable_prompt_injection_filter"]:
            for pattern in self.injection_patterns:
                match = pattern.search(text)
                if match:
                    return ValidationResult(
                        is_valid=False,
                        message="Input contains potential prompt injection",
                        details={"error": "prompt_injection", "pattern": match.group(0)}
                    )
                    
        # Check for scripts if enabled
        if self.config["block_scripts"]:
            for pattern in self.script_patterns:
                match = pattern.search(text)
                if match:
                    return ValidationResult(
                        is_valid=False,
                        message="Input contains script or code that is not allowed",
                        details={"error": "script_detected", "pattern": match.group(0)}
                    )
                    
        # Check for supported language if enabled
        if self.config["check_language"] and len(self.supported_language_patterns) > 0:
            language_supported = False
            for language, pattern in self.supported_language_patterns:
                if pattern.search(text):
                    language_supported = True
                    break
                    
            if not language_supported:
                return ValidationResult(
                    is_valid=False,
                    message="Input language not supported",
                    details={"error": "unsupported_language"}
                )
                
        # If all checks pass, the input is valid
        return ValidationResult(
            is_valid=True,
            message="Input is valid",
            details={"length": len(text)}
        )
        
class SecurityConfig:
    """Configuration for security settings in the model server"""
    
    def __init__(
        self, 
        max_input_length: int = 32768,
        enable_profanity_filter: bool = True,
        enable_prompt_injection_filter: bool = True,
        block_scripts: bool = True,
        max_tokens_per_request: int = 2048,
        max_requests_per_minute: int = 60
    ):
        """Initialize security configuration
        
        Args:
            max_input_length: Maximum allowed input text length
            enable_profanity_filter: Whether to enable profanity filtering
            enable_prompt_injection_filter: Whether to enable prompt injection filtering
            block_scripts: Whether to block script content in input
            max_tokens_per_request: Maximum number of tokens allowed per request
            max_requests_per_minute: Maximum number of requests allowed per minute
        """
        self.max_input_length = max_input_length
        self.enable_profanity_filter = enable_profanity_filter
        self.enable_prompt_injection_filter = enable_prompt_injection_filter
        self.block_scripts = block_scripts
        self.max_tokens_per_request = max_tokens_per_request
        self.max_requests_per_minute = max_requests_per_minute
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary
        
        Returns:
            Dictionary representation of the configuration
        """
        return {
            "max_input_length": self.max_input_length,
            "enable_profanity_filter": self.enable_profanity_filter,
            "enable_prompt_injection_filter": self.enable_prompt_injection_filter,
            "block_scripts": self.block_scripts,
            "max_tokens_per_request": self.max_tokens_per_request,
            "max_requests_per_minute": self.max_requests_per_minute
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SecurityConfig":
        """Create from dictionary
        
        Args:
            data: Dictionary with configuration values
            
        Returns:
            SecurityConfig instance
        """
        return cls(
            max_input_length=data.get("max_input_length", 32768),
            enable_profanity_filter=data.get("enable_profanity_filter", True),
            enable_prompt_injection_filter=data.get("enable_prompt_injection_filter", True),
            block_scripts=data.get("block_scripts", True),
            max_tokens_per_request=data.get("max_tokens_per_request", 2048),
            max_requests_per_minute=data.get("max_requests_per_minute", 60)
        )
