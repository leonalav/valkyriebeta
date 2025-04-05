import json
import hashlib
from typing import Any

def question_hash(question: str) -> str:
    """Create a hash for a question string"""
    return hashlib.md5(question.encode()).hexdigest()

def jdump(obj: Any, path: str) -> None:
    """Dump an object to a JSON file"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def jload(path: str) -> Any:
    """Load an object from a JSON file"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Add aliases for compatibility
save_json = jdump
load_json = jload 