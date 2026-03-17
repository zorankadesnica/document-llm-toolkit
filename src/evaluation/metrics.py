"""
Evaluation Metrics Module

Comprehensive evaluation metrics for assessing LLM-generated content,
including lexical, semantic, and model-based evaluation approaches.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class MetricType(Enum):
    """Types of evaluation metrics."""
    LEXICAL = "lexical"
    SEMANTIC = "semantic"
    MODEL_BASED = "model_based"
    FACTUALITY = "factuality"
    COMPOSITE = "composite"


@dataclass
class MetricResult:
    """Result from a single metric evaluation."""
    metric_name: str
    score: float
    details: dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return f"{self.metric_name}: {self.score:.4f}"


@dataclass
class EvaluationResult:
    """Aggregated results from multiple metrics."""
    metrics: list[MetricResult]
    overall_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Calculate overall score if not provided."""
        if self.overall_score is None and self.metrics:
            self.overall_score = np.mean([m.score for m in self.metrics])
    
    def get_metric(self, name: str) -> MetricResult | None:
        """Get a specific metric result by name."""
        for m in self.metrics:
            if m.metric_name == name:
                return m
        return None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "overall_score": self.overall_score,
            "metrics": {m.metric_name: m.score for m in self.metrics},
            "details": {m.metric_name: m.details for m in self.metrics},
            "metadata": self.metadata,
        }
    
    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = ["Evaluation Results", "=" * 40]
        for m in self.metrics:
            lines.append(f"{m.metric_name}: {m.score:.4f}")
        lines.append("-" * 40)
        lines.append(f"Overall Score: {self.overall_score:.4f}")
        return "\n".join(lines)


class BaseMetric(ABC):
    """Abstract base class for evaluation metrics."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the metric."""
        pass
    
    @property
    @abstractmethod
    def metric_type(self) -> MetricType:
        """Type of the metric."""
        pass
    
    @abstractmethod
    def compute(
        self,
        predictions: list[str],
        references: list[str],
        **kwargs: Any,
    ) -> MetricResult:
        """
        Compute the metric.
        
        Args:
            predictions: Model-generated texts
            references: Reference/ground truth texts
            **kwargs: Additional arguments
            
        Returns:
            MetricResult with score and details
        """
        pass


class RougeMetric(BaseMetric):
    """
    ROUGE (Recall-Oriented Understudy for Gisting Evaluation) metric.
    
    Measures overlap between generated and reference text using
    n-gram matching (ROUGE-N) and longest common subsequence (ROUGE-L).
    """
    
    def __init__(self, rouge_types: list[str] | None = None):
        """
        Initialize ROUGE metric.
        
        Args:
            rouge_types: ROUGE variants to compute (default: rouge1, rouge2, rougeL)
        """
        self.rouge_types = rouge_types or ["rouge1", "rouge2", "rougeL"]
        self._scorer = None
    
    @property
    def name(self) -> str:
        return "ROUGE"
    
    @property
    def metric_type(self) -> MetricType:
        return MetricType.LEXICAL
    
    def _get_scorer(self) -> Any:
        """Lazy load the ROUGE scorer."""
        if self._scorer is None:
            try:
                from rouge_score import rouge_scorer
                self._scorer = rouge_scorer.RougeScorer(
                    self.rouge_types,
                    use_stemmer=True
                )
            except ImportError:
                raise ImportError(
                    "rouge-score package required. "
                    "Install with: pip install rouge-score"
                )
        return self._scorer
    
    def compute(
        self,
        predictions: list[str],
        references: list[str],
        **kwargs: Any,
    ) -> MetricResult:
        """Compute ROUGE scores."""
        scorer = self._get_scorer()
        
        all_scores: dict[str, list[float]] = {rt: [] for rt in self.rouge_types}
        
        for pred, ref in zip(predictions, references):
            scores = scorer.score(ref, pred)
            for rouge_type in self.rouge_types:
                all_scores[rouge_type].append(scores[rouge_type].fmeasure)
        
        avg_scores = {rt: np.mean(scores) for rt, scores in all_scores.items()}
        
        # Use ROUGE-L as the primary score
        primary_score = avg_scores.get("rougeL", list(avg_scores.values())[0])
        
        return MetricResult(
            metric_name=self.name,
            score=float(primary_score),
            details={
                "rouge_scores": avg_scores,
                "num_samples": len(predictions),
            },
        )


class BleuMetric(BaseMetric):
    """
    BLEU (Bilingual Evaluation Understudy) metric.
    
    Measures n-gram precision with a brevity penalty.
    Originally designed for machine translation evaluation.
    """
    
    def __init__(self, max_n: int = 4):
        """
        Initialize BLEU metric.
        
        Args:
            max_n: Maximum n-gram order to consider
        """
        self.max_n = max_n
    
    @property
    def name(self) -> str:
        return "BLEU"
    
    @property
    def metric_type(self) -> MetricType:
        return MetricType.LEXICAL
    
    def compute(
        self,
        predictions: list[str],
        references: list[str],
        **kwargs: Any,
    ) -> MetricResult:
        """Compute BLEU score."""
        try:
            from sacrebleu.metrics import BLEU
            bleu = BLEU(effective_order=True)
        except ImportError:
            raise ImportError(
                "sacrebleu package required. "
                "Install with: pip install sacrebleu"
            )
        
        # sacrebleu expects references as list of lists
        refs_wrapped = [[ref] for ref in references]
        
        # Compute sentence-level BLEU
        sentence_scores = []
        for pred, ref_list in zip(predictions, refs_wrapped):
            score = bleu.sentence_score(pred, ref_list)
            sentence_scores.append(score.score / 100)  # Normalize to 0-1
        
        # Compute corpus-level BLEU
        corpus_score = bleu.corpus_score(predictions, [references])
        
        return MetricResult(
            metric_name=self.name,
            score=corpus_score.score / 100,
            details={
                "corpus_bleu": corpus_score.score / 100,
                "avg_sentence_bleu": float(np.mean(sentence_scores)),
                "precisions": corpus_score.precisions,
                "brevity_penalty": corpus_score.bp,
            },
        )


class BertScoreMetric(BaseMetric):
    """
    BERTScore metric for semantic similarity.
    
    Uses contextual embeddings from BERT to compute
    token-level similarity between texts.
    """
    
    def __init__(
        self,
        model_type: str = "microsoft/deberta-xlarge-mnli",
        device: str | None = None,
    ):
        """
        Initialize BERTScore metric.
        
        Args:
            model_type: Transformer model to use for embeddings
            device: Device to run model on (cuda/cpu)
        """
        self.model_type = model_type
        self.device = device
    
    @property
    def name(self) -> str:
        return "BERTScore"
    
    @property
    def metric_type(self) -> MetricType:
        return MetricType.SEMANTIC
    
    def compute(
        self,
        predictions: list[str],
        references: list[str],
        **kwargs: Any,
    ) -> MetricResult:
        """Compute BERTScore."""
        try:
            from bert_score import score as bert_score
        except ImportError:
            raise ImportError(
                "bert-score package required. "
                "Install with: pip install bert-score"
            )
        
        P, R, F1 = bert_score(
            predictions,
            references,
            model_type=self.model_type,
            device=self.device,
            verbose=False,
        )
        
        return MetricResult(
            metric_name=self.name,
            score=float(F1.mean()),
            details={
                "precision": float(P.mean()),
                "recall": float(R.mean()),
                "f1": float(F1.mean()),
                "precision_std": float(P.std()),
                "recall_std": float(R.std()),
                "f1_std": float(F1.std()),
            },
        )


class SemanticSimilarityMetric(BaseMetric):
    """
    Semantic similarity using sentence embeddings.
    
    Computes cosine similarity between sentence embeddings
    from a sentence transformer model.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize semantic similarity metric.
        
        Args:
            model_name: Sentence transformer model name
        """
        self.model_name = model_name
        self._model = None
    
    @property
    def name(self) -> str:
        return "SemanticSimilarity"
    
    @property
    def metric_type(self) -> MetricType:
        return MetricType.SEMANTIC
    
    def _get_model(self) -> Any:
        """Lazy load the sentence transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers package required. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model
    
    def compute(
        self,
        predictions: list[str],
        references: list[str],
        **kwargs: Any,
    ) -> MetricResult:
        """Compute semantic similarity."""
        from sklearn.metrics.pairwise import cosine_similarity
        
        model = self._get_model()
        
        pred_embeddings = model.encode(predictions)
        ref_embeddings = model.encode(references)
        
        # Compute pairwise cosine similarity
        similarities = []
        for pred_emb, ref_emb in zip(pred_embeddings, ref_embeddings):
            sim = cosine_similarity([pred_emb], [ref_emb])[0][0]
            similarities.append(sim)
        
        return MetricResult(
            metric_name=self.name,
            score=float(np.mean(similarities)),
            details={
                "mean_similarity": float(np.mean(similarities)),
                "std_similarity": float(np.std(similarities)),
                "min_similarity": float(np.min(similarities)),
                "max_similarity": float(np.max(similarities)),
            },
        )


class FactualityMetric(BaseMetric):
    """
    Factuality/faithfulness metric.
    
    Evaluates whether generated content is grounded in
    the source document without hallucination.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize factuality metric.
        
        Args:
            model_name: Model for embedding computation
        """
        self.model_name = model_name
        self._model = None
    
    @property
    def name(self) -> str:
        return "Factuality"
    
    @property
    def metric_type(self) -> MetricType:
        return MetricType.FACTUALITY
    
    def _get_model(self) -> Any:
        """Lazy load the model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers package required. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model
    
    def _sentence_tokenize(self, text: str) -> list[str]:
        """Split text into sentences."""
        try:
            import nltk
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)
            return nltk.sent_tokenize(text)
        except ImportError:
            # Fallback to simple splitting
            import re
            return re.split(r'(?<=[.!?])\s+', text)
    
    def compute(
        self,
        predictions: list[str],
        references: list[str],  # These are source documents
        **kwargs: Any,
    ) -> MetricResult:
        """
        Compute factuality score.
        
        Measures how well each sentence in the prediction
        is grounded in the source document.
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        model = self._get_model()
        
        all_scores = []
        all_details = []
        
        for pred, source in zip(predictions, references):
            # Split into sentences
            pred_sentences = self._sentence_tokenize(pred)
            source_sentences = self._sentence_tokenize(source)
            
            if not pred_sentences or not source_sentences:
                all_scores.append(0.0)
                continue
            
            # Embed sentences
            pred_embeddings = model.encode(pred_sentences)
            source_embeddings = model.encode(source_sentences)
            
            # For each prediction sentence, find max similarity to any source sentence
            sentence_scores = []
            for pred_emb in pred_embeddings:
                similarities = cosine_similarity([pred_emb], source_embeddings)[0]
                max_sim = float(np.max(similarities))
                sentence_scores.append(max_sim)
            
            # Average across all prediction sentences
            doc_score = float(np.mean(sentence_scores))
            all_scores.append(doc_score)
            
            all_details.append({
                "num_pred_sentences": len(pred_sentences),
                "num_source_sentences": len(source_sentences),
                "sentence_scores": sentence_scores,
            })
        
        return MetricResult(
            metric_name=self.name,
            score=float(np.mean(all_scores)),
            details={
                "mean_score": float(np.mean(all_scores)),
                "std_score": float(np.std(all_scores)),
                "per_document_details": all_details,
            },
        )


class MetricSuite:
    """Predefined metric suites for common evaluation scenarios."""
    
    ROUGE = "rouge"
    BLEU = "bleu"
    BERTSCORE = "bertscore"
    SEMANTIC = "semantic"
    FACTUALITY = "factuality"
    
    @classmethod
    def get_summarization_metrics(cls) -> list[BaseMetric]:
        """Get recommended metrics for summarization evaluation."""
        return [
            RougeMetric(),
            BertScoreMetric(),
            SemanticSimilarityMetric(),
        ]
    
    @classmethod
    def get_generation_metrics(cls) -> list[BaseMetric]:
        """Get recommended metrics for general text generation."""
        return [
            BleuMetric(),
            BertScoreMetric(),
            SemanticSimilarityMetric(),
        ]
    
    @classmethod
    def get_faithfulness_metrics(cls) -> list[BaseMetric]:
        """Get metrics for evaluating faithfulness to source."""
        return [
            FactualityMetric(),
            SemanticSimilarityMetric(),
        ]


class MetricRegistry:
    """Registry for managing available metrics."""
    
    _metrics: dict[str, type[BaseMetric]] = {
        "rouge": RougeMetric,
        "bleu": BleuMetric,
        "bertscore": BertScoreMetric,
        "semantic": SemanticSimilarityMetric,
        "factuality": FactualityMetric,
    }
    
    @classmethod
    def register(cls, name: str, metric_class: type[BaseMetric]) -> None:
        """Register a new metric."""
        cls._metrics[name] = metric_class
    
    @classmethod
    def get(cls, name: str, **kwargs: Any) -> BaseMetric:
        """Get a metric instance by name."""
        if name not in cls._metrics:
            raise ValueError(f"Unknown metric: {name}")
        return cls._metrics[name](**kwargs)
    
    @classmethod
    def list_available(cls) -> list[str]:
        """List all available metrics."""
        return list(cls._metrics.keys())
