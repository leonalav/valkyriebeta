from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Tuple, Optional, ClassVar, Type
import json
import os
import logging
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)

@dataclass
class BaseConfig:
    """Base configuration class that all other configs inherit from"""
    
    # Class variables
    CONFIG_VERSION: ClassVar[str] = "1.0.0"
    
    # Instance variables
    config_name: str = "base"
    config_version: str = "1.0.0"
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate configuration parameters.
        Returns a tuple of (is_valid, error_messages).
        """
        valid = True
        errors = []
        
        # Basic validation for version format
        if not self._is_valid_version_format(self.config_version):
            valid = False
            errors.append(f"Invalid config_version format: {self.config_version}. Expected format: X.Y.Z")
            
        return valid, errors
    
    def _is_valid_version_format(self, version: str) -> bool:
        """Check if version string follows semantic versioning format"""
        import re
        pattern = r'^\d+\.\d+\.\d+$'
        return bool(re.match(pattern, version))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return asdict(self)
    
    def to_json(self, file_path: Optional[str] = None) -> Optional[str]:
        """
        Convert configuration to JSON.
        If file_path is provided, save to file and return None.
        Otherwise, return JSON string.
        """
        config_dict = self.to_dict()
        
        if file_path:
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            logger.info(f"Saved configuration to {file_path}")
            return None
        else:
            return json.dumps(config_dict, indent=2)
    
    def to_yaml(self, file_path: Optional[str] = None) -> Optional[str]:
        """
        Convert configuration to YAML.
        If file_path is provided, save to file and return None.
        Otherwise, return YAML string.
        """
        config_dict = self.to_dict()
        
        if file_path:
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            with open(file_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
            logger.info(f"Saved configuration to {file_path}")
            return None
        else:
            return yaml.dump(config_dict, default_flow_style=False)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BaseConfig':
        """Create configuration from dictionary"""
        # Filter out keys that are not in the dataclass
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        
        return cls(**filtered_dict)
    
    @classmethod
    def from_json(cls, json_str_or_path: str) -> 'BaseConfig':
        """
        Create configuration from JSON string or file path.
        Automatically detects whether input is a file path or JSON string.
        """
        if os.path.exists(json_str_or_path):
            # Input is a file path
            with open(json_str_or_path, 'r') as f:
                config_dict = json.load(f)
        else:
            # Input is a JSON string
            config_dict = json.loads(json_str_or_path)
            
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_yaml(cls, yaml_str_or_path: str) -> 'BaseConfig':
        """
        Create configuration from YAML string or file path.
        Automatically detects whether input is a file path or YAML string.
        """
        if os.path.exists(yaml_str_or_path):
            # Input is a file path
            with open(yaml_str_or_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            # Input is a YAML string
            config_dict = yaml.safe_load(yaml_str_or_path)
            
        return cls.from_dict(config_dict)
    
    def merge_with(self, other: 'BaseConfig') -> 'BaseConfig':
        """
        Merge this configuration with another configuration.
        Values from the other configuration take precedence.
        """
        self_dict = self.to_dict()
        other_dict = other.to_dict()
        
        # Merge dictionaries
        merged_dict = {**self_dict, **other_dict}
        
        return self.__class__.from_dict(merged_dict)
    
    def __str__(self) -> str:
        """String representation of configuration"""
        return f"{self.__class__.__name__}({self.to_json()})"
    
    def __repr__(self) -> str:
        """Detailed string representation of configuration"""
        return self.__str__()
