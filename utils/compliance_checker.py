from typing import List, Dict, Any
from dataclasses import dataclass
import re
import logging

@dataclass
class ComplianceConfig:
    check_pii: bool = True
    check_profanity: bool = True
    allowed_languages: List[str] = None
    data_retention_days: int = 30

class ComplianceChecker:
    def __init__(self, config: ComplianceConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.pii_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{16}\b',  # Credit card
        ]

    def check_compliance(self, data: Any) -> Dict[str, bool]:
        results = {
            "contains_pii": False,
            "contains_profanity": False,
            "valid_language": True
        }

        if self.config.check_pii:
            results["contains_pii"] = self._check_pii(str(data))

        return results

    def _check_pii(self, text: str) -> bool:
        for pattern in self.pii_patterns:
            if re.search(pattern, text):
                self.logger.warning("PII detected in input")
                return True
        return False
