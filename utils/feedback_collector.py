from typing import Dict, Any, List
import json
from datetime import datetime
from pathlib import Path

class FeedbackCollector:
    def __init__(self, feedback_dir: str):
        self.feedback_dir = Path(feedback_dir)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)

    def store_feedback(self, 
                      input_data: Any,
                      output: Any,
                      feedback_score: float,
                      user_comments: str = None):
        feedback = {
            "timestamp": datetime.now().isoformat(),
            "input": str(input_data),
            "output": str(output),
            "score": feedback_score,
            "comments": user_comments
        }
        
        feedback_file = self.feedback_dir / f"feedback_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(feedback_file, 'w') as f:
            json.dump(feedback, f, indent=2)

    def analyze_feedback(self) -> Dict[str, float]:
        scores = []
        for feedback_file in self.feedback_dir.glob("*.json"):
            with open(feedback_file) as f:
                feedback = json.load(f)
                scores.append(feedback["score"])

        return {
            "average_score": sum(scores) / len(scores) if scores else 0,
            "total_feedback": len(scores)
        }
