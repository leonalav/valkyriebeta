import os
from typing import Dict, List, Optional
from dataclasses import dataclass
import re

@dataclass
class SecurityValidationResult:
    """Results of security validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]

class SecurityValidator:
    """Validates security settings and configurations."""
    
    @staticmethod
    def validate_paths(paths: List[str]) -> SecurityValidationResult:
        """Validate file paths for security issues"""
        errors = []
        warnings = []
        
        # Get the project directory
        try:
            project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        except Exception:
            project_dir = os.getcwd()
            warnings.append("Could not determine project directory, using current working directory")
        
        for path in paths:
            # Check for parent directory traversal
            if ".." in path:
                errors.append(f"Path '{path}' contains parent directory traversal")
                
            # Check for absolute paths
            if os.path.isabs(path):
                abs_path = path
            else:
                abs_path = os.path.abspath(os.path.join(project_dir, path))
                
            # Verify the path is within the project directory
            if not abs_path.startswith(project_dir):
                errors.append(f"Path '{path}' is outside of the project directory")
                
            # Check for symbolic links
            if os.path.exists(abs_path) and os.path.islink(abs_path):
                target = os.path.realpath(abs_path)
                if not target.startswith(project_dir):
                    errors.append(f"Path '{path}' is a symbolic link pointing outside the project directory")
                    
            # Check for suspicious path segments
            suspicious_patterns = ['/tmp/', '/var/tmp/', '/dev/', '/proc/', '/sys/']
            for pattern in suspicious_patterns:
                if pattern in abs_path:
                    warnings.append(f"Path '{path}' contains suspicious pattern '{pattern}'")
                    
            # Check for executable paths
            if os.path.exists(abs_path) and os.access(abs_path, os.X_OK):
                warnings.append(f"Path '{path}' is executable")
                
        return SecurityValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    @staticmethod
    def validate_config(config: Dict) -> SecurityValidationResult:
        """Validate configuration for security issues."""
        errors = []
        warnings = []
        
        # Check for sensitive keys in config
        sensitive_keys = ['password', 'secret', 'token', 'key']
        for key in config:
            if any(sens in key.lower() for sens in sensitive_keys):
                warnings.append(f"Config contains potentially sensitive key: {key}")
                
        return SecurityValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
