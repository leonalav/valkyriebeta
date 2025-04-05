from dataclasses import dataclass, field
from typing import List, Dict, Optional
import re

@dataclass
class SecurityConfig:
    """Security configuration for the model"""
    
    # Input validation
    max_input_length: int = 32768
    max_tokens_per_request: int = 4096
    input_validation_regex: str = r'^[\w\s\.,\-\'\"]+$'
    blocked_patterns: List[str] = field(default_factory=lambda: [
        r'(?i)(?:delete|drop|truncate)\s+(?:table|database)',
        r'(?i)exec(?:ute)?\s*\(',
        r'(?i)system\s*\(',
        r'<script[^>]*>.*?</script>'
    ])

    # Rate limiting
    requests_per_minute: int = 60
    tokens_per_minute: int = 100000
    concurrent_requests: int = 10

    # Authentication
    require_api_key: bool = True
    api_key_header: str = "X-API-Key"
    jwt_secret: Optional[str] = None
    token_expiry_minutes: int = 60

    # Output filtering
    enable_output_filtering: bool = True
    max_output_length: int = 4096
    filter_profanity: bool = True
    filter_personal_info: bool = True
    
    # Model security
    enable_model_validation: bool = True
    model_hash_verification: bool = True
    allowed_model_sources: List[str] = field(default_factory=lambda: [
        "huggingface.co",
        "openai.com"
    ])

    # Monitoring
    log_all_requests: bool = True
    log_filtered_outputs: bool = True
    alert_on_violations: bool = True
    max_violation_alerts: int = 100

    # Add missing security features
    enable_rate_limiting: bool = True
    max_request_size: int = 1024 * 1024  # 1MB
    allowed_hosts: List[str] = field(default_factory=lambda: ['localhost'])
    ssl_verify: bool = True
    
    # Add security headers
    security_headers: Dict[str, str] = field(default_factory=lambda: {
        'X-Frame-Options': 'DENY',
        'X-Content-Type-Options': 'nosniff',
        'X-XSS-Protection': '1; mode=block',
        'Content-Security-Policy': "default-src 'self'"
    })

    def validate_input(self, input_text: str) -> tuple[bool, str]:
        """Validate input text against security rules"""
        if len(input_text) > self.max_input_length:
            return False, "Input exceeds maximum length"
            
        if not re.match(self.input_validation_regex, input_text):
            return False, "Input contains invalid characters"
            
        for pattern in self.blocked_patterns:
            if re.search(pattern, input_text):
                return False, "Input contains blocked pattern"
                
        return True, "Input validation successful"

    def get_rate_limit_config(self) -> Dict:
        """Get rate limiting configuration"""
        return {
            'requests_per_minute': self.requests_per_minute,
            'tokens_per_minute': self.tokens_per_minute,
            'concurrent_requests': self.concurrent_requests
        }
