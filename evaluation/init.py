from typing import List, Dict, Any
import numpy as np
from dataclasses import dataclass
import re

@dataclass
class ReasoningMetrics:
    logical_consistency: float
    argument_strength: float
    conclusion_validity: float
    inference_accuracy: float

class LogicalReasoningEvaluator:
    def __init__(self, reasoning_rules: Dict[str, Any] = None):
        self.reasoning_rules = reasoning_rules or {}
        self.evaluation_history = []

    def evaluate_reasoning(self, 
                         premises: List[str], 
                         conclusion: str,
                         intermediate_steps: List[str] = None) -> ReasoningMetrics:
        consistency = self._check_logical_consistency(premises, conclusion)
        strength = self._evaluate_argument_strength(premises)
        validity = self._check_conclusion_validity(premises, conclusion)
        accuracy = self._measure_inference_accuracy(premises, intermediate_steps, conclusion)
        
        metrics = ReasoningMetrics(
            logical_consistency=consistency,
            argument_strength=strength,
            conclusion_validity=validity,
            inference_accuracy=accuracy
        )
        self.evaluation_history.append(metrics)
        return metrics

    def _check_logical_consistency(self, premises: List[str], conclusion: str) -> float:
        """
        Check for logical consistency between premises and conclusion.
        
        Args:
            premises: List of premise statements
            conclusion: The conclusion statement
        
        Returns:
            Score between 0.0 and 1.0 indicating logical consistency
        """
        # Initialize score
        consistency_score = 1.0
        
        # Check for empty premises
        if not premises:
            return 0.0
        
        # Check for contradictions within premises
        contradiction_penalty = 0.0
        premise_pairs = [(premises[i], premises[j]) for i in range(len(premises)) for j in range(i+1, len(premises))]
        
        for p1, p2 in premise_pairs:
            # Normalize text for comparison
            p1_tokens = set(p1.lower().split())
            p2_tokens = set(p2.lower().split())
            
            # Look for direct contradictions (simplistic approach)
            negation_words = {"not", "never", "no", "isn't", "aren't", "doesn't", "don't"}
            
            # Check if one premise contains negation words while the other doesn't
            # on the same topic (approximated by word overlap)
            common_words = p1_tokens.intersection(p2_tokens)
            if common_words and (p1_tokens.intersection(negation_words) != p2_tokens.intersection(negation_words)):
                # The more words in common, the more likely a contradiction
                contradiction_penalty += 0.2 * (len(common_words) / max(len(p1_tokens), len(p2_tokens)))
        
        # Check conclusion against premises
        conclusion_tokens = set(conclusion.lower().split())
        
        # At least some words from the conclusion should appear in premises
        premise_words = set()
        for p in premises:
            premise_words.update(p.lower().split())
        
        conclusion_coverage = len(conclusion_tokens.intersection(premise_words)) / max(len(conclusion_tokens), 1)
        if conclusion_coverage < 0.3:
            consistency_score -= 0.3
        
        # Apply contradiction penalty
        consistency_score = max(0.0, consistency_score - min(contradiction_penalty, 0.8))
        
        # Apply reasoning rules if provided
        if hasattr(self, 'reasoning_rules') and self.reasoning_rules:
            rule_scores = []
            for rule_name, rule_func in self.reasoning_rules.items():
                if rule_name.startswith('consistency'):
                    try:
                        rule_score = rule_func(premises, conclusion)
                        rule_scores.append(rule_score)
                    except:
                        continue
            
            if rule_scores:
                # Average the rule scores
                rule_consistency = sum(rule_scores) / len(rule_scores)
                # Blend with the heuristic score
                consistency_score = 0.6 * consistency_score + 0.4 * rule_consistency
        
        return consistency_score

    def _evaluate_argument_strength(self, premises: List[str]) -> float:
        """
        Evaluate the strength of the argument based on the premises.
        
        Args:
            premises: List of premise statements
        
        Returns:
            Score between 0.0 and 1.0 indicating argument strength
        """
        # Initialize score
        strength_score = 0.0
        
        # Check for empty premises
        if not premises:
            return 0.0
        
        # More premises tend to make stronger arguments (up to a point)
        num_premises = len(premises)
        if num_premises <= 5:
            strength_score += 0.1 * num_premises
        else:
            strength_score += 0.5 + 0.05 * min(num_premises - 5, 5)  # Diminishing returns
        
        # Evaluate precision and specificity
        precision_score = 0.0
        for premise in premises:
            # Check for specific language patterns indicating precision
            if re.search(r'\b\d+(\.\d+)?%?\b', premise):  # Contains numbers/percentages
                precision_score += 0.1
            
            # Check for specific qualifiers and limiting words
            specificity_words = {"specifically", "particularly", "exactly", "precisely", "definitely"}
            if any(word in premise.lower() for word in specificity_words):
                precision_score += 0.05
            
            # Check for citations or references
            if re.search(r'\(\d{4}\)|\[[^\]]+\]', premise):
                precision_score += 0.15
        
        # Normalize precision score
        precision_score = min(precision_score, 0.5)
        
        # Check premise diversity (unique words across premises)
        all_words = set()
        total_words = 0
        for premise in premises:
            words = premise.lower().split()
            all_words.update(words)
            total_words += len(words)
        
        # Higher lexical diversity suggests stronger arguments
        diversity_score = min(len(all_words) / max(total_words, 1), 1.0) * 0.3
        
        # Apply reasoning rules if provided
        if hasattr(self, 'reasoning_rules') and self.reasoning_rules:
            rule_scores = []
            for rule_name, rule_func in self.reasoning_rules.items():
                if rule_name.startswith('strength'):
                    try:
                        rule_score = rule_func(premises)
                        rule_scores.append(rule_score)
                    except:
                        continue
            
            if rule_scores:
                # Average the rule scores
                rule_strength = sum(rule_scores) / len(rule_scores)
                
                # Combine scores
                strength_score = (strength_score + precision_score + diversity_score) / 3
                # Blend with rule-based score
                strength_score = 0.7 * strength_score + 0.3 * rule_strength
                return strength_score
        
        # Final strength score
        strength_score = (strength_score + precision_score + diversity_score) / 3
        return strength_score

    def _check_conclusion_validity(self, premises: List[str], conclusion: str) -> float:
        """
        Verify if the conclusion logically follows from the premises.
        
        Args:
            premises: List of premise statements
            conclusion: The conclusion statement
        
        Returns:
            Score between 0.0 and 1.0 indicating conclusion validity
        """
        # Initialize score
        validity_score = 0.5  # Start at middle ground
        
        # Check for empty premises
        if not premises:
            return 0.0
        
        # Combine premises into a single text for analysis
        combined_premises = " ".join(premises).lower()
        conclusion_text = conclusion.lower()
        
        # Check for conclusion restating a premise
        for premise in premises:
            premise_lower = premise.lower()
            similarity = self._calculate_text_similarity(premise_lower, conclusion_text)
            if similarity > 0.8:
                # High similarity means conclusion is just restating a premise
                # This is valid but not a strong inference
                validity_score = 0.6
                break
        
        # Check if conclusion contains words not in any premise
        premise_words = set(combined_premises.split())
        conclusion_words = set(conclusion_text.split())
        
        # Remove common stopwords
        stopwords = {"the", "a", "an", "in", "on", "at", "to", "of", "for", "and", "or", "but", "is", "are", "was", "were"}
        premise_words = premise_words - stopwords
        conclusion_words = conclusion_words - stopwords
        
        # New concepts in conclusion (not in premises) reduce validity
        new_words = conclusion_words - premise_words
        new_word_ratio = len(new_words) / max(len(conclusion_words), 1)
        
        # If too many new concepts, reduce validity
        if new_word_ratio > 0.4:
            validity_score -= 0.3 * min(new_word_ratio, 1.0)
        
        # Check for logical operators in conclusion
        logical_operators = {"therefore", "thus", "hence", "consequently", "as a result", "so"}
        if any(op in conclusion_text for op in logical_operators):
            validity_score += 0.1  # Proper logical connection
        
        # Check for conclusion being longer than all premises combined
        if len(conclusion_text) > len(combined_premises) * 0.8:
            # Overly elaborate conclusion likely introduces new content
            validity_score -= 0.2
        
        # Apply reasoning rules if provided
        if hasattr(self, 'reasoning_rules') and self.reasoning_rules:
            rule_scores = []
            for rule_name, rule_func in self.reasoning_rules.items():
                if rule_name.startswith('validity'):
                    try:
                        rule_score = rule_func(premises, conclusion)
                        rule_scores.append(rule_score)
                    except:
                        continue
            
            if rule_scores:
                # Average the rule scores
                rule_validity = sum(rule_scores) / len(rule_scores)
                # Blend with the heuristic score
                validity_score = 0.5 * validity_score + 0.5 * rule_validity
        
        # Ensure score is within bounds
        return max(0.0, min(validity_score, 1.0))

    def _measure_inference_accuracy(self, premises: List[str], 
                                  steps: List[str], 
                                  conclusion: str) -> float:
        """
        Measure the accuracy of reasoning steps from premises to conclusion.
        
        Args:
            premises: List of premise statements
            steps: List of intermediate reasoning steps
            conclusion: The conclusion statement
        
        Returns:
            Score between 0.0 and 1.0 indicating inference accuracy
        """
        # Check for empty steps
        if not steps:
            # If no steps provided, base score on premises and conclusion only
            return self._check_conclusion_validity(premises, conclusion)
        
        # Start with full score
        accuracy_score = 1.0
        
        # Coherence check: steps should form a logical sequence
        step_coherence_scores = []
        prev_step = premises[-1] if premises else ""
        
        for i, step in enumerate(steps):
            if i == 0 and premises:
                # First step should connect with premises
                premise_connections = [self._calculate_text_similarity(p, step) for p in premises]
                connection_score = max(premise_connections)
                step_coherence_scores.append(connection_score)
            elif i > 0:
                # Check connection with previous step
                connection_score = self._calculate_text_similarity(prev_step, step)
                step_coherence_scores.append(connection_score)
            
            prev_step = step
        
        # Check final step connects to conclusion
        if steps:
            conclusion_connection = self._calculate_text_similarity(steps[-1], conclusion)
            step_coherence_scores.append(conclusion_connection)
        
        # Average coherence score
        if step_coherence_scores:
            coherence_score = sum(step_coherence_scores) / len(step_coherence_scores)
            accuracy_score = coherence_score
        
        # Check for logical fallacies in steps
        fallacy_penalty = 0.0
        fallacy_patterns = {
            "appeal to authority": [r"\b(expert|authority|professor|doctor)\b", r"\bsaid\b"],
            "circular reasoning": [r"\bprove\b.*\bbecause\b.*\bproven\b"],
            "ad hominem": [r"\b(stupid|idiot|fool|dumb)\b"],
            "hasty generalization": [r"\b(all|every|always|never)\b"]
        }
        
        for step in steps:
            for fallacy, patterns in fallacy_patterns.items():
                if all(re.search(pattern, step, re.IGNORECASE) for pattern in patterns):
                    fallacy_penalty += 0.1
        
        # Apply fallacy penalty
        accuracy_score = max(0.0, accuracy_score - min(fallacy_penalty, 0.5))
        
        # Check step progression
        if len(steps) >= 2:
            # Each step should build on previous ones
            information_growth = []
            
            # Create sets of significant words for each step
            step_word_sets = []
            for step in steps:
                words = set(step.lower().split()) - {"the", "a", "an", "in", "on", "at", "to", "of", "for"}
                step_word_sets.append(words)
            
            # Check for information growth between consecutive steps
            for i in range(1, len(step_word_sets)):
                new_info = len(step_word_sets[i] - step_word_sets[i-1])
                relative_growth = new_info / max(len(step_word_sets[i-1]), 1)
                
                # Some growth is good, too much may indicate logic jumps
                if 0.1 <= relative_growth <= 0.4:
                    information_growth.append(0.1)
                elif relative_growth > 0.4:
                    information_growth.append(-0.1)  # Penalty for too big jumps
                else:
                    information_growth.append(0.0)  # No growth
            
            # Apply information growth score adjustment
            if information_growth:
                growth_score = sum(information_growth)
                accuracy_score = min(1.0, accuracy_score + growth_score)
        
        # Apply reasoning rules if provided
        if hasattr(self, 'reasoning_rules') and self.reasoning_rules:
            rule_scores = []
            for rule_name, rule_func in self.reasoning_rules.items():
                if rule_name.startswith('accuracy'):
                    try:
                        rule_score = rule_func(premises, steps, conclusion)
                        rule_scores.append(rule_score)
                    except:
                        continue
            
            if rule_scores:
                # Average the rule scores
                rule_accuracy = sum(rule_scores) / len(rule_scores)
                # Blend with the heuristic score
                accuracy_score = 0.6 * accuracy_score + 0.4 * rule_accuracy
        
        # Ensure score is within bounds
        return max(0.0, min(accuracy_score, 1.0))
        
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two text snippets.
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Convert to lowercase and tokenize
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        
        # Remove common stopwords
        stopwords = {"the", "a", "an", "in", "on", "at", "to", "of", "for", "and", "or", "but"}
        tokens1 = tokens1 - stopwords
        tokens2 = tokens2 - stopwords
        
        # Calculate Jaccard similarity
        if not tokens1 or not tokens2:
            return 0.0
            
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        
        return intersection / union

class MetricsCollector:
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
        self.metadata: Dict[str, Any] = {}

    def add_metric(self, name: str, value: float, metadata: Dict[str, Any] = None):
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
        
        if metadata:
            if name not in self.metadata:
                self.metadata[name] = []
            self.metadata[name].append(metadata)

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        summary = {}
        for name, values in self.metrics.items():
            summary[name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'count': len(values)
            }
        return summary

    def reset(self):
        self.metrics.clear()
        self.metadata.clear()
