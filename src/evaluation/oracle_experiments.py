"""
Oracle Experiments for Discourse Phenomena Evaluation

Implements oracle experimental setup for evaluating how well
models can use contextual signals for discourse phenomena.

Key Insight:
Oracles provide UPPER BOUNDS on improvements from discourse modeling.
If oracle signals improve BLEU by Δ, then perfect modeling of that
phenomenon could improve BLEU by at most Δ.

Oracle Types:
1. Previous target sentence: Gold standard previous translation
2. Coreference (pronoun): Perfect gender/number information
3. Coherence (repeated words): Perfect word sense disambiguation

Advantages over challenge sets:
- Automatically created for any domain/language
- Stronger signals enable clearer analysis
- Can measure upper bound of potential improvement

Reference: Stojanovski & Fraser (2018) "Coreference and Coherence in
           Neural Machine Translation: A Study Using Oracle Experiments"
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from enum import Enum
from collections import defaultdict


class OracleType(Enum):
    """Types of oracle signals."""
    PREVIOUS_TARGET = "previous_target"
    COREFERENCE = "coreference"
    COHERENCE = "coherence"
    COMBINED = "combined"


@dataclass
class OracleSample:
    """A single oracle-annotated training/test sample."""
    source: str                       # Original source sentence
    target: str                       # Target translation
    oracle_source: str                # Source with oracle annotations
    oracle_context: str               # Oracle context (prepended to source)
    oracle_type: OracleType
    
    # Metadata
    pronoun_positions: List[int] = field(default_factory=list)
    repeated_words: List[str] = field(default_factory=list)
    antecedent: Optional[str] = None
    target_pronoun: Optional[str] = None


@dataclass
class OracleDataset:
    """Dataset with oracle annotations."""
    samples: List[OracleSample]
    oracle_type: OracleType
    statistics: Dict = field(default_factory=dict)


class PronounOracleCreator:
    """
    Creates coreference (pronoun) oracle annotations.
    
    Methodology:
    1. Find pronouns in source sentence
    2. Get corresponding target pronouns
    3. Mark source pronouns with XPRONOUN
    4. Add target pronouns to oracle context
    
    Example:
        Source: "It presents a problem."
        Target: "Er präsentiert ein Problem."
        Oracle source: "XPRONOUN It presents a problem."
        Oracle context: "er"
        
    The model learns to use the oracle context (German pronoun)
    to generate the correct translation.
    
    This establishes an UPPER BOUND: if oracle gives +5 BLEU,
    perfect coreference resolution could give at most +5 BLEU.
    """
    
    # Pronouns to track (English → German mappings matter)
    ENGLISH_PRONOUNS = {
        'it', 'he', 'she', 'they', 'them', 'his', 'her', 'its'
    }
    
    GERMAN_PRONOUNS = {
        'er', 'sie', 'es', 'ihn', 'ihm', 'ihr', 'ihnen', 
        'sein', 'seine', 'seinen', 'seiner'
    }
    
    PRONOUN_MARKER = "XPRONOUN"
    CONTEXT_SEPARATOR = " !@#$ "
    
    def __init__(self):
        self.stats = defaultdict(int)
    
    def find_pronouns(self, text: str, pronoun_set: Set[str]) -> List[Tuple[int, str]]:
        """Find pronouns and their positions in text."""
        words = text.lower().split()
        pronouns = []
        for i, word in enumerate(words):
            # Remove punctuation for matching
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word in pronoun_set:
                pronouns.append((i, clean_word))
        return pronouns
    
    def create_oracle_sample(
        self,
        source: str,
        target: str,
        previous_source: Optional[str] = None,
        previous_target: Optional[str] = None
    ) -> Optional[OracleSample]:
        """
        Create oracle sample if both source and target have pronouns.
        
        Args:
            source: Current source sentence
            target: Current target sentence
            previous_source: Previous source (for fair oracle)
            previous_target: Previous target (for noisy oracle)
            
        Returns:
            OracleSample if applicable, None otherwise
        """
        # Find pronouns
        src_pronouns = self.find_pronouns(source, self.ENGLISH_PRONOUNS)
        tgt_pronouns = self.find_pronouns(target, self.GERMAN_PRONOUNS)
        
        if not src_pronouns or not tgt_pronouns:
            return None
        
        # Mark source pronouns
        words = source.split()
        for pos, _ in src_pronouns:
            if pos < len(words):
                words[pos] = f"{self.PRONOUN_MARKER} {words[pos]}"
        
        oracle_source = " ".join(words)
        
        # Create oracle context from target pronouns
        oracle_context = " ".join([p for _, p in tgt_pronouns])
        
        self.stats['samples_created'] += 1
        self.stats['pronouns_marked'] += len(src_pronouns)
        
        return OracleSample(
            source=source,
            target=target,
            oracle_source=oracle_source,
            oracle_context=oracle_context,
            oracle_type=OracleType.COREFERENCE,
            pronoun_positions=[p for p, _ in src_pronouns],
            target_pronoun=tgt_pronouns[0][1] if tgt_pronouns else None
        )
    
    def create_fair_oracle(
        self,
        source: str,
        target: str,
        previous_source: str,
        antecedent: str
    ) -> Optional[OracleSample]:
        """
        Create fair oracle using source-side antecedent.
        
        In fair oracle, we don't use target-side information.
        Instead, we use coreference resolution on source side
        to identify the antecedent.
        
        This is more realistic but provides weaker signal.
        """
        src_pronouns = self.find_pronouns(source, self.ENGLISH_PRONOUNS)
        
        if not src_pronouns:
            return None
        
        # Mark pronouns
        words = source.split()
        for pos, _ in src_pronouns:
            if pos < len(words):
                words[pos] = f"{self.PRONOUN_MARKER} {words[pos]}"
        
        oracle_source = " ".join(words)
        
        # Use antecedent as context (source-side only)
        oracle_context = antecedent
        
        return OracleSample(
            source=source,
            target=target,
            oracle_source=oracle_source,
            oracle_context=oracle_context,
            oracle_type=OracleType.COREFERENCE,
            antecedent=antecedent
        )
    
    def create_noisy_oracle(
        self,
        source: str,
        target: str,
        previous_target: str
    ) -> Optional[OracleSample]:
        """
        Create noisy oracle with previous target + pronouns.
        
        Context = previous_target + target_pronouns
        
        This adds "noise" (extra information) to test model robustness.
        A good model should still focus on the relevant pronouns.
        """
        sample = self.create_oracle_sample(source, target)
        
        if sample is None:
            return None
        
        # Prepend previous target to oracle context
        noisy_context = f"{previous_target} {sample.oracle_context}"
        
        return OracleSample(
            source=sample.source,
            target=sample.target,
            oracle_source=sample.oracle_source,
            oracle_context=noisy_context,
            oracle_type=OracleType.COREFERENCE,
            pronoun_positions=sample.pronoun_positions,
            target_pronoun=sample.target_pronoun
        )


class CoherenceOracleCreator:
    """
    Creates coherence (repeated words) oracle annotations.
    
    Methodology:
    1. Find words repeated in consecutive sentences
    2. Mark repeated source words with XREP
    3. Add target translations of repeated words to context
    
    Example:
        Prev source: "...go right to the source."
        Prev target: "...direkt zum Ursprung."
        
        Source: "God the source is pretty."
        Target: "Mann, so ein hübscher Ursprung."
        
        Oracle source: "God the XREP source is pretty."
        Oracle context: "Ursprung"
        
    This helps with polysemous words:
    - "source" → "Quelle" (fountain) or "Ursprung" (origin)
    - Oracle tells model which translation was used previously
    """
    
    REP_MARKER = "XREP"
    CONTEXT_SEPARATOR = " !@#$ "
    
    def __init__(self, stopwords: Optional[Set[str]] = None):
        self.stopwords = stopwords or self._default_stopwords()
        self.stats = defaultdict(int)
    
    def _default_stopwords(self) -> Set[str]:
        return {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'to', 'of', 'in',
            'for', 'on', 'with', 'at', 'by', 'from', 'and', 'or', 'but',
            'if', 'this', 'that', 'it', 'i', 'you', 'he', 'she', 'we', 'they'
        }
    
    def find_repeated_words(
        self,
        current: str,
        previous: str
    ) -> List[str]:
        """Find content words appearing in both sentences."""
        current_words = set(
            w.lower() for w in re.findall(r'\b\w+\b', current)
            if w.lower() not in self.stopwords and len(w) > 2
        )
        previous_words = set(
            w.lower() for w in re.findall(r'\b\w+\b', previous)
            if w.lower() not in self.stopwords and len(w) > 2
        )
        
        return list(current_words & previous_words)
    
    def create_oracle_sample(
        self,
        source: str,
        target: str,
        previous_source: str,
        previous_target: str
    ) -> Optional[OracleSample]:
        """
        Create coherence oracle sample.
        
        Requires:
        - Repeated words in source sentences
        - Corresponding repeated words in target sentences
        """
        # Find repeated words on both sides
        src_repeated = self.find_repeated_words(source, previous_source)
        tgt_repeated = self.find_repeated_words(target, previous_target)
        
        if not src_repeated or not tgt_repeated:
            return None
        
        # Mark repeated source words
        oracle_source = source
        for word in src_repeated:
            # Case-insensitive replacement
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            oracle_source = pattern.sub(f"{self.REP_MARKER} {word}", oracle_source, count=1)
        
        # Oracle context = repeated target words
        oracle_context = " ".join(tgt_repeated)
        
        self.stats['samples_created'] += 1
        self.stats['repeated_words'] += len(src_repeated)
        
        return OracleSample(
            source=source,
            target=target,
            oracle_source=oracle_source,
            oracle_context=oracle_context,
            oracle_type=OracleType.COHERENCE,
            repeated_words=src_repeated
        )


class PreviousSentenceOracle:
    """
    Creates previous target sentence oracle.
    
    This is the simplest oracle: use gold-standard previous translation.
    
    Useful for:
    1. Online post-editing (previous translation IS available)
    2. Establishing upper bound for context use
    3. Testing model's ability to use context
    
    Modes:
    - Gold: Use reference previous target (oracle)
    - Translated: Use MT-translated previous source (fair)
    - Source: Use previous source sentence (source-side only)
    """
    
    CONTEXT_SEPARATOR = " !@#$ "
    
    def create_gold_oracle(
        self,
        source: str,
        target: str,
        previous_target: str
    ) -> OracleSample:
        """
        Create oracle with gold previous target.
        
        This is the STRONGEST signal and establishes upper bound.
        """
        return OracleSample(
            source=source,
            target=target,
            oracle_source=source,
            oracle_context=previous_target,
            oracle_type=OracleType.PREVIOUS_TARGET
        )
    
    def create_fair_oracle(
        self,
        source: str,
        target: str,
        previous_source: str,
        translation_fn
    ) -> OracleSample:
        """
        Create fair oracle using translated previous source.
        
        More realistic: uses MT output, not reference.
        Tests whether model can use imperfect context.
        """
        translated_prev = translation_fn(previous_source)
        
        return OracleSample(
            source=source,
            target=target,
            oracle_source=source,
            oracle_context=translated_prev,
            oracle_type=OracleType.PREVIOUS_TARGET
        )


class OracleDatasetBuilder:
    """
    Builds complete oracle datasets from parallel corpora.
    
    Usage:
        builder = OracleDatasetBuilder()
        pronoun_data = builder.build_pronoun_oracle(sentences)
        coherence_data = builder.build_coherence_oracle(sentences)
    """
    
    def __init__(self):
        self.pronoun_creator = PronounOracleCreator()
        self.coherence_creator = CoherenceOracleCreator()
        self.prev_sentence_oracle = PreviousSentenceOracle()
    
    def build_pronoun_oracle(
        self,
        source_sentences: List[str],
        target_sentences: List[str]
    ) -> OracleDataset:
        """Build pronoun (coreference) oracle dataset."""
        samples = []
        
        for i in range(len(source_sentences)):
            sample = self.pronoun_creator.create_oracle_sample(
                source_sentences[i],
                target_sentences[i]
            )
            if sample is not None:
                samples.append(sample)
        
        return OracleDataset(
            samples=samples,
            oracle_type=OracleType.COREFERENCE,
            statistics=dict(self.pronoun_creator.stats)
        )
    
    def build_coherence_oracle(
        self,
        source_sentences: List[str],
        target_sentences: List[str]
    ) -> OracleDataset:
        """Build coherence (repeated words) oracle dataset."""
        samples = []
        
        for i in range(1, len(source_sentences)):
            sample = self.coherence_creator.create_oracle_sample(
                source=source_sentences[i],
                target=target_sentences[i],
                previous_source=source_sentences[i-1],
                previous_target=target_sentences[i-1]
            )
            if sample is not None:
                samples.append(sample)
        
        return OracleDataset(
            samples=samples,
            oracle_type=OracleType.COHERENCE,
            statistics=dict(self.coherence_creator.stats)
        )
    
    def build_previous_target_oracle(
        self,
        source_sentences: List[str],
        target_sentences: List[str]
    ) -> OracleDataset:
        """Build previous target sentence oracle dataset."""
        samples = []
        
        for i in range(1, len(source_sentences)):
            sample = self.prev_sentence_oracle.create_gold_oracle(
                source=source_sentences[i],
                target=target_sentences[i],
                previous_target=target_sentences[i-1]
            )
            samples.append(sample)
        
        return OracleDataset(
            samples=samples,
            oracle_type=OracleType.PREVIOUS_TARGET,
            statistics={
                'total_samples': len(samples),
                'context_coverage': 1.0  # All samples have context
            }
        )
    
    def format_for_training(self, sample: OracleSample) -> Tuple[str, str]:
        """
        Format oracle sample for model training.
        
        Returns: (input_text, target_text)
        
        Input format: "{oracle_context} !@#$ {oracle_source}"
        """
        if sample.oracle_context:
            input_text = f"{sample.oracle_context}{PronounOracleCreator.CONTEXT_SEPARATOR}{sample.oracle_source}"
        else:
            input_text = sample.source
        
        return input_text, sample.target


def analyze_oracle_impact(
    baseline_bleu: float,
    oracle_bleu: float,
    phenomenon: str
) -> Dict:
    """
    Analyze the impact of oracle signals.
    
    Returns:
        Analysis of what the oracle results tell us.
    """
    delta = oracle_bleu - baseline_bleu
    relative_gain = (delta / baseline_bleu) * 100
    
    analysis = {
        "baseline_bleu": baseline_bleu,
        "oracle_bleu": oracle_bleu,
        "absolute_gain": delta,
        "relative_gain_pct": relative_gain,
        "phenomenon": phenomenon
    }
    
    # Interpretation
    if delta > 5:
        analysis["interpretation"] = (
            f"LARGE potential for {phenomenon} modeling. "
            f"Current models significantly under-utilize this signal."
        )
    elif delta > 2:
        analysis["interpretation"] = (
            f"MODERATE potential for {phenomenon} modeling. "
            f"Worth investing in context-aware approaches."
        )
    elif delta > 0.5:
        analysis["interpretation"] = (
            f"SMALL but measurable potential for {phenomenon}. "
            f"May be worth modeling if computationally cheap."
        )
    else:
        analysis["interpretation"] = (
            f"MINIMAL additional signal from {phenomenon}. "
            f"Baseline may already capture this adequately."
        )
    
    return analysis


# Example usage and demonstration
if __name__ == "__main__":
    print("Oracle Experiments Demonstration")
    print("=" * 60)
    
    # Example parallel sentences
    source_sentences = [
        "Let me summarize the novel for you.",
        "It presents a problem.",
        "The problem is quite complex.",
        "We need to solve it quickly."
    ]
    
    target_sentences = [
        "Ich fasse den Roman für dich zusammen.",
        "Er präsentiert ein Problem.",
        "Das Problem ist ziemlich komplex.",
        "Wir müssen es schnell lösen."
    ]
    
    # Build oracle datasets
    builder = OracleDatasetBuilder()
    
    # Pronoun oracle
    pronoun_data = builder.build_pronoun_oracle(source_sentences, target_sentences)
    print(f"\nPronoun Oracle:")
    print(f"  Samples created: {len(pronoun_data.samples)}")
    for sample in pronoun_data.samples:
        print(f"  Source: {sample.source}")
        print(f"  Oracle: {sample.oracle_context} !@#$ {sample.oracle_source}")
        print()
    
    # Coherence oracle
    coherence_data = builder.build_coherence_oracle(source_sentences, target_sentences)
    print(f"\nCoherence Oracle:")
    print(f"  Samples created: {len(coherence_data.samples)}")
    for sample in coherence_data.samples:
        if sample.repeated_words:
            print(f"  Source: {sample.source}")
            print(f"  Repeated: {sample.repeated_words}")
            print(f"  Oracle: {sample.oracle_context}")
            print()
    
    # Previous target oracle
    prev_data = builder.build_previous_target_oracle(source_sentences, target_sentences)
    print(f"\nPrevious Target Oracle:")
    print(f"  Samples created: {len(prev_data.samples)}")
    for i, sample in enumerate(prev_data.samples):
        print(f"  [{i}] Context: {sample.oracle_context[:50]}...")
    
    # Impact analysis (simulated)
    print("\n" + "=" * 60)
    print("Oracle Impact Analysis (Example Results)")
    print("=" * 60)
    
    analyses = [
        analyze_oracle_impact(28.57, 35.59, "coreference (pronoun)"),
        analyze_oracle_impact(28.57, 30.46, "coherence (repeated words)"),
        analyze_oracle_impact(28.57, 30.35, "previous target sentence")
    ]
    
    for analysis in analyses:
        print(f"\n{analysis['phenomenon'].upper()}")
        print(f"  Baseline: {analysis['baseline_bleu']:.2f} BLEU")
        print(f"  Oracle:   {analysis['oracle_bleu']:.2f} BLEU")
        print(f"  Gain:     +{analysis['absolute_gain']:.2f} BLEU ({analysis['relative_gain_pct']:.1f}%)")
        print(f"  {analysis['interpretation']}")
    
    print("\n" + "=" * 60)
    print("KEY INSIGHTS FROM ORACLE EXPERIMENTS")
    print("=" * 60)
    print("""
    1. COREFERENCE has highest potential (+7 BLEU in best case)
       - Models significantly under-utilize gender/number information
       - Context-aware models should focus here
       
    2. COHERENCE has moderate potential (+2 BLEU)
       - Repeated word disambiguation helps
       - But harder to extract signal automatically
       
    3. PREVIOUS TARGET provides consistent gains (+1.5-2 BLEU)
       - Useful for post-editing scenarios
       - Easy to implement
       
    4. NOISY oracles test ROBUSTNESS
       - Good model should extract relevant signal from noise
       - Concatenation models handle noise better
       
    5. Oracle experiments are LANGUAGE/DOMAIN AGNOSTIC
       - No manual challenge set creation needed
       - Can be applied to any parallel corpus
    """)