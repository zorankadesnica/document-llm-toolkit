"""
Contrastive Evaluation Module for Pronoun Translation

Implements challenge set evaluation following ContraPro/ContraCAT methodology.
Tests whether models can correctly identify the right pronoun translation
given context containing the antecedent.

Mathematical Framework:
- Score each candidate translation using model log-probability
- Model succeeds if P(y+) > P(y-) for all contrastive pairs
- Adversarial robustness: test if small perturbations flip predictions

Reference: 
- Müller et al. (2018) "A Large-Scale Test Set for Context-Dependent Pronoun Translation"
- Stojanovski et al. (2020) "ContraCAT: Contrastive Coreference Analytical Templates"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import numpy as np
from collections import defaultdict


class PronounGender(Enum):
    """German pronoun genders for English-German evaluation."""
    MASCULINE = "er"      # he/it (masc)
    FEMININE = "sie"      # she/it (fem)
    NEUTER = "es"         # it (neut)
    PLURAL = "sie"        # they


@dataclass
class ContrastivePair:
    """
    A contrastive evaluation example.
    
    Contains source sentence with pronoun, context with antecedent,
    correct translation, and contrastive (incorrect) translations.
    """
    source: str
    context: str
    reference: str                    # Correct translation
    contrastive: List[str]            # Incorrect translations
    antecedent: str                   # The noun the pronoun refers to
    antecedent_translation: str       # German translation of antecedent
    correct_pronoun: str              # Correct German pronoun
    source_pronoun: str = "it"        # English pronoun being evaluated
    
    def __post_init__(self):
        self.num_contrastive = len(self.contrastive)


@dataclass
class ContrastiveResult:
    """Result of contrastive evaluation on a single example."""
    correct: bool
    reference_score: float
    contrastive_scores: List[float]
    margin: float                     # score(ref) - max(score(contrastive))
    pronoun_type: str
    antecedent_distance: int = 0      # Sentences between pronoun and antecedent


@dataclass
class ContrastiveEvaluationResult:
    """Aggregated results from contrastive evaluation."""
    accuracy: float
    results_by_pronoun: Dict[str, float]
    results_by_distance: Dict[int, float]
    average_margin: float
    total_examples: int
    detailed_results: List[ContrastiveResult] = field(default_factory=list)


class ContrastiveScorer:
    """
    Scores translations using model log-probabilities.
    
    Mathematical formulation:
        score(y | x, c) = Σ_t log P(y_t | y_<t, x, c)
        
    The model should assign higher score to correct translation:
        score(y+) > score(y-) for all contrastive pairs
    """
    
    def __init__(self, model: nn.Module, tokenizer: Any):
        """
        Args:
            model: Sequence-to-sequence model with score() method
            tokenizer: Tokenizer for encoding text
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
    
    @torch.no_grad()
    def score_translation(
        self,
        source: str,
        target: str,
        context: Optional[str] = None
    ) -> float:
        """
        Compute log-probability score for a translation.
        
        Args:
            source: Source sentence
            target: Target translation
            context: Optional context sentence
            
        Returns:
            Log-probability score (higher = more likely)
        """
        # Encode inputs
        if context:
            source_input = f"{context} ||| {source}"
        else:
            source_input = source
        
        source_ids = self.tokenizer.encode(source_input, return_tensors="pt")
        target_ids = self.tokenizer.encode(target, return_tensors="pt")
        
        # Get model scores
        outputs = self.model(
            input_ids=source_ids,
            decoder_input_ids=target_ids[:, :-1],
            labels=target_ids[:, 1:]
        )
        
        # Negative cross-entropy = log probability
        log_prob = -outputs.loss.item() * (target_ids.size(1) - 1)
        
        return log_prob
    
    def evaluate_pair(
        self,
        pair: ContrastivePair
    ) -> ContrastiveResult:
        """
        Evaluate a single contrastive pair.
        
        Args:
            pair: ContrastivePair with reference and contrastive translations
            
        Returns:
            ContrastiveResult with scores and correctness
        """
        # Score reference translation
        ref_score = self.score_translation(
            pair.source, pair.reference, pair.context
        )
        
        # Score contrastive translations
        contrastive_scores = [
            self.score_translation(pair.source, cont, pair.context)
            for cont in pair.contrastive
        ]
        
        # Check if reference has highest score
        max_contrastive = max(contrastive_scores)
        correct = ref_score > max_contrastive
        margin = ref_score - max_contrastive
        
        return ContrastiveResult(
            correct=correct,
            reference_score=ref_score,
            contrastive_scores=contrastive_scores,
            margin=margin,
            pronoun_type=pair.correct_pronoun
        )


class ContrastiveEvaluator:
    """
    Full contrastive evaluation pipeline.
    
    Evaluates model on challenge set and computes:
    - Overall accuracy
    - Accuracy by pronoun type
    - Accuracy by antecedent distance
    - Margin statistics
    
    Also supports adversarial robustness testing.
    """
    
    def __init__(self, scorer: ContrastiveScorer):
        self.scorer = scorer
    
    def evaluate(
        self,
        pairs: List[ContrastivePair]
    ) -> ContrastiveEvaluationResult:
        """
        Evaluate model on list of contrastive pairs.
        
        Mathematical formulation:
            Accuracy = (1/N) Σ_i 1[score(y_i+) > max_j score(y_ij-)]
        
        Args:
            pairs: List of ContrastivePair examples
            
        Returns:
            ContrastiveEvaluationResult with detailed metrics
        """
        results = []
        by_pronoun = defaultdict(list)
        by_distance = defaultdict(list)
        
        for pair in pairs:
            result = self.scorer.evaluate_pair(pair)
            results.append(result)
            
            by_pronoun[result.pronoun_type].append(result.correct)
            by_distance[result.antecedent_distance].append(result.correct)
        
        # Compute aggregated metrics
        accuracy = sum(r.correct for r in results) / len(results)
        
        results_by_pronoun = {
            pronoun: sum(corrects) / len(corrects)
            for pronoun, corrects in by_pronoun.items()
        }
        
        results_by_distance = {
            dist: sum(corrects) / len(corrects)
            for dist, corrects in by_distance.items()
        }
        
        average_margin = sum(r.margin for r in results) / len(results)
        
        return ContrastiveEvaluationResult(
            accuracy=accuracy,
            results_by_pronoun=results_by_pronoun,
            results_by_distance=results_by_distance,
            average_margin=average_margin,
            total_examples=len(results),
            detailed_results=results
        )


class AdversarialPerturbation:
    """
    Adversarial perturbation generator for testing robustness.
    
    Key insight: If model relies on spurious correlations rather than
    true coreference understanding, small perturbations should flip predictions.
    
    Types of perturbations:
    1. Lexical: Replace context words with synonyms
    2. Distractor: Add irrelevant nouns with different genders
    3. Order: Reorder context sentences
    """
    
    @staticmethod
    def add_distractor(
        pair: ContrastivePair,
        distractor_noun: str,
        distractor_gender: str
    ) -> ContrastivePair:
        """
        Add a distractor noun to context that could confuse the model.
        
        If the model uses simple heuristics (e.g., "closest noun"),
        it will incorrectly select the distractor as antecedent.
        
        Args:
            pair: Original contrastive pair
            distractor_noun: Noun to add (e.g., "the book")
            distractor_gender: Gender of distractor in German
            
        Returns:
            New pair with modified context
        """
        # Insert distractor between antecedent and pronoun
        modified_context = f"{pair.context} {distractor_noun} was also there."
        
        return ContrastivePair(
            source=pair.source,
            context=modified_context,
            reference=pair.reference,
            contrastive=pair.contrastive,
            antecedent=pair.antecedent,
            antecedent_translation=pair.antecedent_translation,
            correct_pronoun=pair.correct_pronoun,
            source_pronoun=pair.source_pronoun
        )
    
    @staticmethod
    def replace_irrelevant_word(
        pair: ContrastivePair,
        original: str,
        replacement: str
    ) -> ContrastivePair:
        """
        Replace an irrelevant word in context.
        
        This should NOT change the correct answer, but might if model
        relies on spurious correlations.
        
        Args:
            pair: Original pair
            original: Word to replace (should be irrelevant to coreference)
            replacement: Replacement word
            
        Returns:
            Modified pair
        """
        modified_context = pair.context.replace(original, replacement)
        
        return ContrastivePair(
            source=pair.source,
            context=modified_context,
            reference=pair.reference,
            contrastive=pair.contrastive,
            antecedent=pair.antecedent,
            antecedent_translation=pair.antecedent_translation,
            correct_pronoun=pair.correct_pronoun,
            source_pronoun=pair.source_pronoun
        )


class AdversarialEvaluator:
    """
    Evaluates adversarial robustness of contrastive evaluation.
    
    A robust model should:
    1. Maintain correct predictions under irrelevant perturbations
    2. Only change predictions when semantically relevant changes occur
    
    Metrics:
    - Flip rate: % of predictions that change under perturbation
    - Robustness: % of correct predictions maintained after perturbation
    """
    
    def __init__(self, evaluator: ContrastiveEvaluator):
        self.evaluator = evaluator
    
    def evaluate_robustness(
        self,
        original_pairs: List[ContrastivePair],
        perturbed_pairs: List[ContrastivePair]
    ) -> Dict[str, float]:
        """
        Compare model performance on original vs perturbed examples.
        
        Args:
            original_pairs: Original contrastive pairs
            perturbed_pairs: Pairs with adversarial perturbations
            
        Returns:
            Dict with robustness metrics
        """
        orig_results = self.evaluator.evaluate(original_pairs)
        pert_results = self.evaluator.evaluate(perturbed_pairs)
        
        # Count flips
        flips = sum(
            o.correct != p.correct
            for o, p in zip(orig_results.detailed_results, 
                           pert_results.detailed_results)
        )
        
        # Count maintained correct predictions
        maintained = sum(
            o.correct and p.correct
            for o, p in zip(orig_results.detailed_results,
                           pert_results.detailed_results)
        )
        
        originally_correct = sum(r.correct for r in orig_results.detailed_results)
        
        return {
            "original_accuracy": orig_results.accuracy,
            "perturbed_accuracy": pert_results.accuracy,
            "flip_rate": flips / len(original_pairs),
            "robustness": maintained / originally_correct if originally_correct > 0 else 0,
            "accuracy_drop": orig_results.accuracy - pert_results.accuracy
        }


class CoreferenceStepEvaluator:
    """
    Evaluate specific steps of coreference resolution pipeline.
    
    Following ContraCAT methodology, we test each step independently:
    1. Pronoun identification
    2. Antecedent identification  
    3. Feature extraction (gender/number)
    4. Pronoun generation
    
    This helps diagnose WHERE the model fails, not just IF it fails.
    """
    
    def __init__(self, model: nn.Module, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer
    
    def evaluate_antecedent_attention(
        self,
        pair: ContrastivePair,
        attention_weights: torch.Tensor
    ) -> Dict[str, float]:
        """
        Check if model attends to correct antecedent when generating pronoun.
        
        Args:
            pair: Contrastive pair with antecedent information
            attention_weights: [num_heads, target_len, source_len]
            
        Returns:
            Metrics about antecedent attention
        """
        # Find antecedent position in context
        context_tokens = self.tokenizer.tokenize(pair.context)
        antecedent_tokens = self.tokenizer.tokenize(pair.antecedent)
        
        # Find where antecedent appears
        antecedent_positions = []
        for i in range(len(context_tokens) - len(antecedent_tokens) + 1):
            if context_tokens[i:i+len(antecedent_tokens)] == antecedent_tokens:
                antecedent_positions.extend(range(i, i + len(antecedent_tokens)))
        
        if not antecedent_positions:
            return {"antecedent_found": False}
        
        # Find pronoun position in target
        target_tokens = self.tokenizer.tokenize(pair.reference)
        pronoun_position = None
        for i, token in enumerate(target_tokens):
            if token.lower() in ["er", "sie", "es"]:
                pronoun_position = i
                break
        
        if pronoun_position is None:
            return {"pronoun_found": False}
        
        # Compute attention to antecedent
        # Average over heads
        avg_attention = attention_weights.mean(dim=0)  # [target_len, source_len]
        
        # Attention from pronoun to antecedent positions
        pronoun_attention = avg_attention[pronoun_position]
        antecedent_attention = pronoun_attention[antecedent_positions].sum().item()
        
        # Compare to attention to other nouns (potential distractors)
        total_attention = pronoun_attention.sum().item()
        
        return {
            "antecedent_found": True,
            "pronoun_found": True,
            "antecedent_attention": antecedent_attention,
            "antecedent_attention_ratio": antecedent_attention / total_attention,
            "num_antecedent_tokens": len(antecedent_positions)
        }


def create_contrapro_example(
    source: str,
    context: str,
    antecedent: str,
    antecedent_translation: str,
    correct_gender: PronounGender
) -> ContrastivePair:
    """
    Create a ContraPro-style contrastive example.
    
    Args:
        source: Source sentence with "it"
        context: Context containing antecedent
        antecedent: English antecedent
        antecedent_translation: German translation of antecedent
        correct_gender: Correct gender for pronoun
        
    Returns:
        ContrastivePair ready for evaluation
    """
    # Define pronoun options
    pronouns = {
        PronounGender.MASCULINE: "er",
        PronounGender.FEMININE: "sie",
        PronounGender.NEUTER: "es"
    }
    
    correct_pronoun = pronouns[correct_gender]
    
    # Create reference and contrastive translations
    # (In practice, these would come from parallel data or templates)
    reference_template = source.replace("it", correct_pronoun.capitalize())
    
    contrastive = []
    for gender, pronoun in pronouns.items():
        if gender != correct_gender:
            contrastive.append(source.replace("it", pronoun.capitalize()))
    
    return ContrastivePair(
        source=source,
        context=context,
        reference=reference_template,
        contrastive=contrastive,
        antecedent=antecedent,
        antecedent_translation=antecedent_translation,
        correct_pronoun=correct_pronoun
    )


# Example usage and testing
if __name__ == "__main__":
    # Create example contrastive pairs
    pair1 = ContrastivePair(
        source="It presents a problem.",
        context="Let me summarize the novel for you.",
        reference="Er präsentiert ein Problem.",
        contrastive=[
            "Sie präsentiert ein Problem.",
            "Es präsentiert ein Problem."
        ],
        antecedent="novel",
        antecedent_translation="Roman",
        correct_pronoun="er"
    )
    
    pair2 = ContrastivePair(
        source="It was very expensive.",
        context="I bought a new car yesterday.",
        reference="Es war sehr teuer.",
        contrastive=[
            "Er war sehr teuer.",
            "Sie war sehr teuer."
        ],
        antecedent="car",
        antecedent_translation="Auto",
        correct_pronoun="es"
    )
    
    print("Contrastive Evaluation Examples")
    print("=" * 50)
    print(f"Example 1:")
    print(f"  Context: {pair1.context}")
    print(f"  Source: {pair1.source}")
    print(f"  Correct: {pair1.reference} (pronoun: {pair1.correct_pronoun})")
    print(f"  Contrastive: {pair1.contrastive}")
    print()
    print(f"Example 2:")
    print(f"  Context: {pair2.context}")
    print(f"  Source: {pair2.source}")
    print(f"  Correct: {pair2.reference} (pronoun: {pair2.correct_pronoun})")
    print(f"  Contrastive: {pair2.contrastive}")
    
    # Demonstrate adversarial perturbation
    print("\n" + "=" * 50)
    print("Adversarial Perturbation Example")
    print("=" * 50)
    
    perturbed = AdversarialPerturbation.add_distractor(
        pair1,
        distractor_noun="The newspaper",
        distractor_gender="feminine"
    )
    
    print(f"Original context: {pair1.context}")
    print(f"Perturbed context: {perturbed.context}")
    print(f"If model uses 'closest noun' heuristic, it might incorrectly choose 'sie'")