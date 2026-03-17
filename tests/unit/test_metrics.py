"""
Unit Tests for Evaluation Metrics

Tests for the evaluation metrics module including ROUGE, BLEU,
BERTScore, and semantic similarity metrics.
"""

import numpy as np
import pytest

from src.evaluation.metrics import (
    BaseMetric,
    BleuMetric,
    EvaluationResult,
    FactualityMetric,
    MetricRegistry,
    MetricResult,
    MetricSuite,
    MetricType,
    RougeMetric,
    SemanticSimilarityMetric,
)


class TestMetricResult:
    """Tests for MetricResult dataclass."""
    
    def test_metric_result_creation(self) -> None:
        """Test basic MetricResult creation."""
        result = MetricResult(
            metric_name="test_metric",
            score=0.85,
            details={"key": "value"}
        )
        
        assert result.metric_name == "test_metric"
        assert result.score == 0.85
        assert result.details == {"key": "value"}
    
    def test_metric_result_repr(self) -> None:
        """Test MetricResult string representation."""
        result = MetricResult(metric_name="ROUGE", score=0.4523)
        assert "ROUGE" in repr(result)
        assert "0.4523" in repr(result)


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""
    
    def test_evaluation_result_auto_score(self) -> None:
        """Test automatic overall score calculation."""
        metrics = [
            MetricResult("metric1", 0.8),
            MetricResult("metric2", 0.6),
            MetricResult("metric3", 0.7),
        ]
        
        result = EvaluationResult(metrics=metrics)
        
        assert result.overall_score == pytest.approx(0.7, rel=1e-3)
    
    def test_evaluation_result_explicit_score(self) -> None:
        """Test explicit overall score."""
        metrics = [MetricResult("metric1", 0.8)]
        result = EvaluationResult(metrics=metrics, overall_score=0.9)
        
        assert result.overall_score == 0.9
    
    def test_get_metric(self) -> None:
        """Test getting specific metric by name."""
        metrics = [
            MetricResult("ROUGE", 0.5),
            MetricResult("BLEU", 0.6),
        ]
        result = EvaluationResult(metrics=metrics)
        
        rouge = result.get_metric("ROUGE")
        assert rouge is not None
        assert rouge.score == 0.5
        
        missing = result.get_metric("NONEXISTENT")
        assert missing is None
    
    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        metrics = [
            MetricResult("ROUGE", 0.5, {"detail": "value"}),
        ]
        result = EvaluationResult(metrics=metrics)
        
        d = result.to_dict()
        
        assert "overall_score" in d
        assert "metrics" in d
        assert d["metrics"]["ROUGE"] == 0.5
    
    def test_summary(self) -> None:
        """Test human-readable summary generation."""
        metrics = [
            MetricResult("ROUGE", 0.5),
            MetricResult("BLEU", 0.6),
        ]
        result = EvaluationResult(metrics=metrics)
        
        summary = result.summary()
        
        assert "ROUGE" in summary
        assert "BLEU" in summary
        assert "Overall Score" in summary


class TestRougeMetric:
    """Tests for ROUGE metric."""
    
    @pytest.fixture
    def rouge_metric(self) -> RougeMetric:
        """Create ROUGE metric instance."""
        return RougeMetric(rouge_types=["rouge1", "rouge2", "rougeL"])
    
    def test_metric_properties(self, rouge_metric: RougeMetric) -> None:
        """Test metric name and type."""
        assert rouge_metric.name == "ROUGE"
        assert rouge_metric.metric_type == MetricType.LEXICAL
    
    @pytest.mark.skipif(
        not pytest.importorskip("rouge_score", reason="rouge-score not installed"),
        reason="rouge-score not installed"
    )
    def test_compute_identical_texts(self, rouge_metric: RougeMetric) -> None:
        """Test ROUGE on identical texts (should be perfect score)."""
        predictions = ["The quick brown fox jumps over the lazy dog."]
        references = ["The quick brown fox jumps over the lazy dog."]
        
        result = rouge_metric.compute(predictions, references)
        
        assert result.metric_name == "ROUGE"
        assert result.score == pytest.approx(1.0, rel=0.01)
    
    @pytest.mark.skipif(
        not pytest.importorskip("rouge_score", reason="rouge-score not installed"),
        reason="rouge-score not installed"
    )
    def test_compute_different_texts(self, rouge_metric: RougeMetric) -> None:
        """Test ROUGE on completely different texts."""
        predictions = ["Hello world"]
        references = ["Goodbye universe"]
        
        result = rouge_metric.compute(predictions, references)
        
        assert result.score < 0.5  # Should be low but not necessarily 0


class TestBleuMetric:
    """Tests for BLEU metric."""
    
    @pytest.fixture
    def bleu_metric(self) -> BleuMetric:
        """Create BLEU metric instance."""
        return BleuMetric()
    
    def test_metric_properties(self, bleu_metric: BleuMetric) -> None:
        """Test metric name and type."""
        assert bleu_metric.name == "BLEU"
        assert bleu_metric.metric_type == MetricType.LEXICAL
    
    @pytest.mark.skipif(
        not pytest.importorskip("sacrebleu", reason="sacrebleu not installed"),
        reason="sacrebleu not installed"
    )
    def test_compute_identical_texts(self, bleu_metric: BleuMetric) -> None:
        """Test BLEU on identical texts."""
        predictions = ["The cat sat on the mat."]
        references = ["The cat sat on the mat."]
        
        result = bleu_metric.compute(predictions, references)
        
        assert result.metric_name == "BLEU"
        assert result.score == pytest.approx(1.0, rel=0.01)


class TestMetricRegistry:
    """Tests for MetricRegistry."""
    
    def test_list_available(self) -> None:
        """Test listing available metrics."""
        available = MetricRegistry.list_available()
        
        assert "rouge" in available
        assert "bleu" in available
        assert "bertscore" in available
    
    def test_get_known_metric(self) -> None:
        """Test getting a known metric."""
        metric = MetricRegistry.get("rouge")
        
        assert isinstance(metric, RougeMetric)
    
    def test_get_unknown_metric(self) -> None:
        """Test getting an unknown metric raises error."""
        with pytest.raises(ValueError, match="Unknown metric"):
            MetricRegistry.get("nonexistent_metric")
    
    def test_register_custom_metric(self) -> None:
        """Test registering a custom metric."""
        class CustomMetric(BaseMetric):
            @property
            def name(self) -> str:
                return "Custom"
            
            @property
            def metric_type(self) -> MetricType:
                return MetricType.COMPOSITE
            
            def compute(self, predictions, references, **kwargs):
                return MetricResult("Custom", 0.5)
        
        MetricRegistry.register("custom", CustomMetric)
        
        metric = MetricRegistry.get("custom")
        assert metric.name == "Custom"


class TestMetricSuite:
    """Tests for MetricSuite helper class."""
    
    def test_get_summarization_metrics(self) -> None:
        """Test getting summarization metrics."""
        metrics = MetricSuite.get_summarization_metrics()
        
        assert len(metrics) > 0
        assert all(isinstance(m, BaseMetric) for m in metrics)
    
    def test_get_generation_metrics(self) -> None:
        """Test getting generation metrics."""
        metrics = MetricSuite.get_generation_metrics()
        
        assert len(metrics) > 0
        assert all(isinstance(m, BaseMetric) for m in metrics)
    
    def test_get_faithfulness_metrics(self) -> None:
        """Test getting faithfulness metrics."""
        metrics = MetricSuite.get_faithfulness_metrics()
        
        assert len(metrics) > 0
        assert any("Factuality" in m.name for m in metrics)


class TestFactualityMetric:
    """Tests for Factuality metric."""
    
    @pytest.fixture
    def factuality_metric(self) -> FactualityMetric:
        """Create Factuality metric instance."""
        return FactualityMetric()
    
    def test_metric_properties(self, factuality_metric: FactualityMetric) -> None:
        """Test metric name and type."""
        assert factuality_metric.name == "Factuality"
        assert factuality_metric.metric_type == MetricType.FACTUALITY
    
    def test_sentence_tokenize(self, factuality_metric: FactualityMetric) -> None:
        """Test sentence tokenization."""
        text = "First sentence. Second sentence! Third sentence?"
        sentences = factuality_metric._sentence_tokenize(text)
        
        assert len(sentences) >= 3


# Integration test for full evaluation pipeline
class TestEvaluationIntegration:
    """Integration tests for evaluation pipeline."""
    
    @pytest.mark.skipif(
        not pytest.importorskip("rouge_score", reason="rouge-score not installed"),
        reason="Dependencies not installed"
    )
    def test_full_evaluation_pipeline(self) -> None:
        """Test running multiple metrics together."""
        predictions = [
            "The company reported strong quarterly earnings.",
            "Sales increased by 20% year over year.",
        ]
        references = [
            "The company announced excellent quarterly results.",
            "Revenue grew by 20% compared to last year.",
        ]
        
        # Run multiple metrics
        rouge = RougeMetric()
        
        rouge_result = rouge.compute(predictions, references)
        
        # Combine results
        eval_result = EvaluationResult(metrics=[rouge_result])
        
        assert eval_result.overall_score is not None
        assert eval_result.overall_score > 0
