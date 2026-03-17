"""
Document Coherence and Cohesion Metrics

Implements metrics for evaluating document-level consistency and coherence,
addressing the limitation of BLEU for discourse evaluation.

Key Insight:
BLEU cannot reliably measure discourse improvements because:
1. Pronouns are infrequent tokens (~2-5% of text)
2. Expected BLEU change ≈ f_p × Δp_1 ≈ 0.03 × 0.5 = 0.015 (within noise)
3. N-grams don't capture long-range dependencies

This module provides dedicated coherence metrics:
- Lexical cohesion (word repetition consistency)
- Entity consistency (same entity → same translation)
- Semantic coherence (LSA-based sentence similarity)
- Discourse connector analysis

Reference: 
- Wong & Kit (2012) "Extending Machine Translation Evaluation Metrics"
- Lapata & Barzilay (2005) "Automatic Evaluation of Text Coherence"
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import re


@dataclass
class CoherenceResult:
    """Result from coherence evaluation."""
    metric_name: str
    score: float
    details: Dict = field(default_factory=dict)


@dataclass
class DocumentCoherenceResult:
    """Aggregated coherence results for a document."""
    lexical_cohesion: float
    entity_consistency: float
    semantic_coherence: float
    overall_score: float
    sentence_scores: List[float] = field(default_factory=list)
    details: Dict = field(default_factory=dict)


class LexicalCohesionMetric:
    """
    Measures lexical cohesion via repeated content words.
    
    Mathematical formulation (Wong & Kit, 2012):
        LC = (# repeated content words) / (# total content words)
        
    Higher score = more consistent word usage across sentences.
    
    For translation:
    - Source has repeated word W in sentences i and j
    - Both translations should use same/similar target word
    """
    
    def __init__(self, stopwords: Optional[Set[str]] = None):
        self.stopwords = stopwords or self._default_stopwords()
    
    def _default_stopwords(self) -> Set[str]:
        """Common English stopwords."""
        return {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
            'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
            'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'under', 'again', 'further', 'then', 'once',
            'here', 'there', 'when', 'where', 'why', 'how', 'all',
            'each', 'few', 'more', 'most', 'other', 'some', 'such',
            'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
            'too', 'very', 's', 't', 'just', 'don', 'now', 'and', 'but',
            'or', 'because', 'until', 'while', 'if', 'this', 'that',
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }
    
    def _extract_content_words(self, text: str) -> List[str]:
        """Extract content words (non-stopwords, alphabetic only)."""
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        return [w for w in words if w not in self.stopwords and len(w) > 2]
    
    def compute(self, sentences: List[str]) -> CoherenceResult:
        """
        Compute lexical cohesion score for document.
        
        Args:
            sentences: List of sentences in the document
            
        Returns:
            CoherenceResult with cohesion score
        """
        if len(sentences) < 2:
            return CoherenceResult(
                metric_name="lexical_cohesion",
                score=1.0,
                details={"reason": "single sentence"}
            )
        
        all_content_words = []
        for sent in sentences:
            words = self._extract_content_words(sent)
            all_content_words.extend(words)
        
        if len(all_content_words) == 0:
            return CoherenceResult(
                metric_name="lexical_cohesion",
                score=0.0,
                details={"reason": "no content words"}
            )
        
        # Count word frequencies
        word_counts = defaultdict(int)
        for word in all_content_words:
            word_counts[word] += 1
        
        # Repeated words = words appearing more than once
        repeated_count = sum(
            count for word, count in word_counts.items() 
            if count > 1
        )
        
        total_count = len(all_content_words)
        
        # Lexical cohesion score
        score = repeated_count / total_count
        
        # Find most repeated words for analysis
        top_repeated = sorted(
            [(w, c) for w, c in word_counts.items() if c > 1],
            key=lambda x: -x[1]
        )[:10]
        
        return CoherenceResult(
            metric_name="lexical_cohesion",
            score=score,
            details={
                "total_content_words": total_count,
                "repeated_words": repeated_count,
                "unique_words": len(word_counts),
                "top_repeated": top_repeated
            }
        )


class EntityConsistencyMetric:
    """
    Measures consistency of entity translations across sentences.
    
    Key idea: Same entity should have consistent translation throughout document.
    
    Example (inconsistent):
        Sentence 1: "The bank approved the loan" → "Die Bank genehmigte den Kredit"
        Sentence 5: "The bank was closed" → "Das Ufer war geschlossen"
        (Wrong! "bank" should be "Bank" in both, not "Ufer")
    
    Mathematical formulation:
        For entity e appearing in sentences {s_i}:
            consistency(e) = (# consistent translations) / (# occurrences - 1)
        
        EntityConsistency = mean over all repeated entities
    """
    
    def __init__(self):
        self.entity_patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Proper nouns
            r'\bthe\s+[a-z]+\b',                      # Definite NPs
        ]
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract potential entities from text."""
        entities = []
        for pattern in self.entity_patterns:
            matches = re.findall(pattern, text)
            entities.extend([m.lower() for m in matches])
        return entities
    
    def compute(
        self,
        source_sentences: List[str],
        target_sentences: List[str],
        alignments: Optional[List[Dict[str, str]]] = None
    ) -> CoherenceResult:
        """
        Compute entity consistency between source and target.
        
        Args:
            source_sentences: Source language sentences
            target_sentences: Target language translations
            alignments: Optional word alignments
            
        Returns:
            CoherenceResult with consistency score
        """
        # Track entity translations
        entity_translations = defaultdict(list)
        
        for i, (src, tgt) in enumerate(zip(source_sentences, target_sentences)):
            src_entities = self._extract_entities(src)
            tgt_words = tgt.lower().split()
            
            for entity in src_entities:
                # Simple heuristic: look for similar-length words
                # In practice, use word alignments
                entity_translations[entity].append((i, tgt))
        
        # Compute consistency for repeated entities
        consistency_scores = []
        inconsistent_entities = []
        
        for entity, translations in entity_translations.items():
            if len(translations) < 2:
                continue
            
            # Check if translations are consistent
            # Simple heuristic: look for same key words
            first_tgt = translations[0][1].lower()
            consistent_count = 0
            
            for sent_idx, tgt in translations[1:]:
                # Simple overlap check
                first_words = set(first_tgt.split())
                tgt_words = set(tgt.lower().split())
                overlap = len(first_words & tgt_words) / max(len(first_words), 1)
                
                if overlap > 0.3:  # Threshold for "consistent"
                    consistent_count += 1
            
            consistency = consistent_count / (len(translations) - 1)
            consistency_scores.append(consistency)
            
            if consistency < 0.5:
                inconsistent_entities.append(entity)
        
        if not consistency_scores:
            overall_score = 1.0
        else:
            overall_score = np.mean(consistency_scores)
        
        return CoherenceResult(
            metric_name="entity_consistency",
            score=overall_score,
            details={
                "num_repeated_entities": len(consistency_scores),
                "inconsistent_entities": inconsistent_entities,
                "per_entity_scores": consistency_scores
            }
        )


class SemanticCoherenceMetric:
    """
    Measures semantic coherence using word overlap between consecutive sentences.
    
    Based on the principle that coherent documents have topical continuity,
    which manifests as shared content words between adjacent sentences.
    
    Mathematical formulation:
        overlap(s_i, s_{i+1}) = |words_i ∩ words_{i+1}| / |words_i ∪ words_{i+1}|
        coherence = (1/(n-1)) Σ_{i=1}^{n-1} overlap(s_i, s_{i+1})
    
    High coherence = consistent topic/entity references
    Low coherence = abrupt topic changes with no shared words
    """
    
    def __init__(self, embedding_model: Optional[object] = None):
        """
        Args:
            embedding_model: Model with encode() method for sentence embeddings
                           If None, uses word overlap similarity
        """
        self.embedding_model = embedding_model
        self._stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'can', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
            'from', 'as', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'under', 'again', 'further',
            'then', 'once', 'here', 'there', 'when', 'where', 'why',
            'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some',
            'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
            'than', 'too', 'very', 'just', 'and', 'but', 'or', 'because',
            'until', 'while', 'if', 'this', 'that', 'these', 'those',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which',
            'who', 'whom', 'its', 'his', 'her', 'their', 'my', 'your',
            'however', 'also', 'about', 'out', 'up', 'down', 'over'
        }
    
    def _extract_content_words(self, text: str) -> Set[str]:
        """Extract content words (non-stopwords) from text."""
        words = set()
        for word in text.lower().split():
            # Remove punctuation
            clean_word = ''.join(c for c in word if c.isalnum())
            
            # Handle possessives: "bank's" -> "banks" -> "bank"
            if clean_word.endswith('s') and len(clean_word) > 3:
                # Add both the word and its stem (without trailing s)
                stem = clean_word[:-1]
                if stem not in self._stopwords and len(stem) > 2:
                    words.add(stem)
            
            if clean_word and clean_word not in self._stopwords and len(clean_word) > 2:
                words.add(clean_word)
        return words
    
    def _word_overlap_similarity(self, sent1: str, sent2: str) -> float:
        """
        Compute Jaccard-like word overlap between two sentences.
        
        Returns:
            Overlap score in [0, 1]
        """
        words1 = self._extract_content_words(sent1)
        words2 = self._extract_content_words(sent2)
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def compute(self, sentences: List[str]) -> CoherenceResult:
        """
        Compute semantic coherence score.
        
        Args:
            sentences: List of sentences in document
            
        Returns:
            CoherenceResult with coherence score
        """
        if len(sentences) < 2:
            return CoherenceResult(
                metric_name="semantic_coherence",
                score=1.0,
                details={"reason": "single sentence"}
            )
        
        # Compute word overlap between consecutive sentences
        similarities = []
        for i in range(len(sentences) - 1):
            sim = self._word_overlap_similarity(sentences[i], sentences[i + 1])
            similarities.append(sim)
        
        # Average coherence
        avg_coherence = np.mean(similarities) if similarities else 0.0
        
        # Find low-coherence transitions (potential problems)
        low_coherence_indices = [
            i for i, sim in enumerate(similarities) if sim < 0.1
        ]
        
        return CoherenceResult(
            metric_name="semantic_coherence",
            score=float(avg_coherence),
            details={
                "consecutive_similarities": similarities,
                "min_similarity": float(min(similarities)) if similarities else 0.0,
                "max_similarity": float(max(similarities)) if similarities else 0.0,
                "low_coherence_transitions": low_coherence_indices
            }
        )


class DiscourseConnectorMetric:
    """
    Analyzes discourse connector usage.
    
    Discourse connectors (therefore, however, because, etc.) signal
    logical relationships between sentences. Their proper translation
    is important for coherence.
    """
    
    CONNECTORS = {
        'causal': ['because', 'therefore', 'thus', 'hence', 'so', 'since'],
        'contrastive': ['however', 'but', 'although', 'though', 'yet', 'still'],
        'additive': ['also', 'moreover', 'furthermore', 'additionally', 'besides'],
        'temporal': ['then', 'next', 'finally', 'first', 'second', 'meanwhile'],
        'elaborative': ['specifically', 'namely', 'for example', 'that is']
    }
    
    def compute(self, sentences: List[str]) -> CoherenceResult:
        """
        Analyze discourse connector usage.
        
        Args:
            sentences: List of sentences
            
        Returns:
            Analysis of connector usage
        """
        connector_counts = defaultdict(int)
        connector_positions = defaultdict(list)
        
        for i, sent in enumerate(sentences):
            sent_lower = sent.lower()
            for category, connectors in self.CONNECTORS.items():
                for conn in connectors:
                    if conn in sent_lower:
                        connector_counts[category] += 1
                        connector_positions[category].append(i)
        
        total_connectors = sum(connector_counts.values())
        connector_density = total_connectors / len(sentences) if sentences else 0
        
        return CoherenceResult(
            metric_name="discourse_connectors",
            score=connector_density,  # Higher = more explicit discourse structure
            details={
                "counts_by_type": dict(connector_counts),
                "total_connectors": total_connectors,
                "density": connector_density,
                "positions": dict(connector_positions)
            }
        )


class DocumentCoherenceEvaluator:
    """
    Complete document coherence evaluation combining multiple metrics.
    
    Addresses BLEU's limitations for discourse by providing:
    1. Lexical cohesion (consistent word usage)
    2. Entity consistency (same entity → same translation)
    3. Semantic coherence (smooth topic flow)
    4. Discourse connector analysis
    
    Usage example:
        evaluator = DocumentCoherenceEvaluator()
        result = evaluator.evaluate(generated_sentences, reference_sentences)
        print(f"Overall coherence: {result.overall_score}")
    """
    
    def __init__(
        self,
        embedding_model: Optional[object] = None,
        weights: Optional[Dict[str, float]] = None
    ):
        self.lexical = LexicalCohesionMetric()
        self.entity = EntityConsistencyMetric()
        self.semantic = SemanticCoherenceMetric(embedding_model)
        self.discourse = DiscourseConnectorMetric()
        
        self.weights = weights or {
            'lexical': 0.25,
            'entity': 0.30,
            'semantic': 0.35,
            'discourse': 0.10
        }
    
    def evaluate(
        self,
        generated: List[str],
        reference: Optional[List[str]] = None,
        source: Optional[List[str]] = None
    ) -> DocumentCoherenceResult:
        """
        Evaluate document coherence.
        
        Args:
            generated: Generated/translated sentences
            reference: Optional reference sentences (for comparison)
            source: Optional source sentences (for entity consistency)
            
        Returns:
            DocumentCoherenceResult with all metrics
        """
        # Compute individual metrics
        lexical_result = self.lexical.compute(generated)
        semantic_result = self.semantic.compute(generated)
        discourse_result = self.discourse.compute(generated)
        
        if source is not None:
            entity_result = self.entity.compute(source, generated)
        else:
            entity_result = CoherenceResult(
                metric_name="entity_consistency",
                score=1.0,
                details={"reason": "no source provided"}
            )
        
        # Weighted overall score
        overall = (
            self.weights['lexical'] * lexical_result.score +
            self.weights['entity'] * entity_result.score +
            self.weights['semantic'] * semantic_result.score +
            self.weights['discourse'] * min(discourse_result.score, 1.0)  # Cap at 1
        )
        
        # Per-sentence semantic scores
        sentence_scores = semantic_result.details.get('consecutive_similarities', [])
        
        return DocumentCoherenceResult(
            lexical_cohesion=lexical_result.score,
            entity_consistency=entity_result.score,
            semantic_coherence=semantic_result.score,
            overall_score=overall,
            sentence_scores=sentence_scores,
            details={
                'lexical_details': lexical_result.details,
                'entity_details': entity_result.details,
                'semantic_details': semantic_result.details,
                'discourse_details': discourse_result.details
            }
        )
    
    def compare_to_reference(
        self,
        generated: List[str],
        reference: List[str]
    ) -> Dict[str, float]:
        """
        Compare coherence of generated vs reference.
        
        Args:
            generated: Generated sentences
            reference: Reference sentences
            
        Returns:
            Comparison metrics
        """
        gen_result = self.evaluate(generated)
        ref_result = self.evaluate(reference)
        
        return {
            'generated_coherence': gen_result.overall_score,
            'reference_coherence': ref_result.overall_score,
            'coherence_gap': ref_result.overall_score - gen_result.overall_score,
            'lexical_gap': ref_result.lexical_cohesion - gen_result.lexical_cohesion,
            'semantic_gap': ref_result.semantic_coherence - gen_result.semantic_coherence
        }


# Example usage
if __name__ == "__main__":
    # Example document
    sentences = [
        "The bank announced record profits today.",
        "The institution has been growing steadily.",
        "However, some analysts remain skeptical.",
        "The bank's CEO addressed these concerns.",
        "She emphasized the strong fundamentals.",
    ]
    
    evaluator = DocumentCoherenceEvaluator()
    result = evaluator.evaluate(sentences)
    
    print("Document Coherence Evaluation")
    print("=" * 50)
    print(f"Lexical Cohesion: {result.lexical_cohesion:.3f}")
    print(f"Entity Consistency: {result.entity_consistency:.3f}")
    print(f"Semantic Coherence: {result.semantic_coherence:.3f}")
    print(f"Overall Score: {result.overall_score:.3f}")
    print()
    print("Details:")
    print(f"  Top repeated words: {result.details['lexical_details'].get('top_repeated', [])[:5]}")
    print(f"  Consecutive similarities: {[f'{s:.3f}' for s in result.sentence_scores]}")
    
    # Demonstrate BLEU limitation
    print("\n" + "=" * 50)
    print("Why BLEU fails for discourse:")
    print("=" * 50)
    print("""
    BLEU problem: Pronouns are ~3% of tokens
    
    If we fix ALL pronoun errors:
        ΔPronoun_accuracy = 0.5 (50% → 100%)
        ΔBLEU ≈ 0.03 × 0.5 = 0.015
        
    This 1.5% change is within BLEU's noise range!
    Hence, discourse improvements are invisible to BLEU.
    """)