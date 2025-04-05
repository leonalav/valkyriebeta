from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime

@dataclass
class ComplianceConfig:
    """Configuration for regulatory compliance and data governance"""
    
    # Data protection
    enable_data_encryption: bool = True
    encryption_algorithm: str = "AES-256"
    key_rotation_days: int = 90
    data_retention_days: int = 30
    
    # Privacy compliance
    gdpr_compliance: bool = True
    ccpa_compliance: bool = True
    hipaa_compliance: bool = False
    pii_detection: bool = True
    pii_handling: str = "mask"  # Options: mask, remove, encrypt
    
    # Audit logging
    enable_audit_logging: bool = True
    audit_log_retention_days: int = 365
    log_access_attempts: bool = True
    log_data_modifications: bool = True
    
    # Model governance
    model_versioning: bool = True
    version_retention_count: int = 3
    require_model_approval: bool = True
    approved_model_versions: List[str] = field(default_factory=list)
    
    # Data governance
    data_classification_levels: Dict[str, Dict] = field(default_factory=lambda: {
        'public': {'access_level': 0, 'encryption': False},
        'internal': {'access_level': 1, 'encryption': True},
        'confidential': {'access_level': 2, 'encryption': True},
        'restricted': {'access_level': 3, 'encryption': True}
    })

    def validate_compliance(self, data_classification: str) -> tuple[bool, str]:
        """Validate compliance requirements for given data classification"""
        if data_classification not in self.data_classification_levels:
            return False, f"Invalid data classification: {data_classification}"
            
        return True, "Compliance requirements met"

    def get_retention_policy(self, data_type: str) -> Dict:
        """Get retention policy for specific data type"""
        return {
            'retention_days': self.data_retention_days,
            'encryption_required': self.enable_data_encryption,
            'audit_logging': self.enable_audit_logging
        }
