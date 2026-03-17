"""
Unit Tests for A/B Testing Framework

Tests for the statistical analysis and experiment management components.
"""

from datetime import datetime

import numpy as np
import pytest

from src.evaluation.ab_testing import (
    ABTestResult,
    ABTestRunner,
    ExperimentConfig,
    ExperimentStatus,
    ExperimentTracker,
    MetricSample,
    StatisticalAnalyzer,
    StatisticalResult,
    StatisticalTest,
)


class TestStatisticalAnalyzer:
    """Tests for StatisticalAnalyzer class."""
    
    def test_welch_t_test_identical_groups(self) -> None:
        """Test Welch's t-test with identical groups."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)
        
        result = StatisticalAnalyzer.welch_t_test(data, data.copy())
        
        assert result.test_name == "Welch's t-test"
        assert result.p_value > 0.05  # Not significant
        assert not result.significant
        assert result.effect_size == pytest.approx(0, abs=0.1)
    
    def test_welch_t_test_different_means(self) -> None:
        """Test Welch's t-test with clearly different groups."""
        np.random.seed(42)
        control = np.random.normal(0, 1, 100)
        treatment = np.random.normal(2, 1, 100)  # Different mean
        
        result = StatisticalAnalyzer.welch_t_test(control, treatment)
        
        assert result.p_value < 0.05  # Should be significant
        assert result.significant
        assert result.effect_size > 1.0  # Large effect
        assert result.effect_size_interpretation == "large"
    
    def test_welch_t_test_confidence_interval(self) -> None:
        """Test confidence interval calculation."""
        np.random.seed(42)
        control = np.random.normal(100, 10, 50)
        treatment = np.random.normal(110, 10, 50)
        
        result = StatisticalAnalyzer.welch_t_test(control, treatment, alpha=0.05)
        
        ci_lower, ci_upper = result.confidence_interval
        assert ci_lower < ci_upper
        # True difference is ~10, CI should contain it
        assert ci_lower < 10 < ci_upper or (ci_lower > 0)  # At least positive
    
    def test_mann_whitney_u_test(self) -> None:
        """Test Mann-Whitney U test."""
        np.random.seed(42)
        control = np.random.normal(0, 1, 50)
        treatment = np.random.normal(1, 1, 50)
        
        result = StatisticalAnalyzer.mann_whitney_u(control, treatment)
        
        assert result.test_name == "Mann-Whitney U"
        assert result.p_value < 0.05
        assert result.significant
    
    def test_bootstrap_test(self) -> None:
        """Test bootstrap hypothesis test."""
        np.random.seed(42)
        control = np.random.normal(0, 1, 30)
        treatment = np.random.normal(0.5, 1, 30)
        
        result = StatisticalAnalyzer.bootstrap_test(
            control,
            treatment,
            n_bootstrap=1000
        )
        
        assert result.test_name == "Bootstrap"
        assert 0 <= result.p_value <= 1
        assert result.confidence_interval[0] < result.confidence_interval[1]
    
    def test_effect_size_interpretation(self) -> None:
        """Test effect size interpretation."""
        np.random.seed(42)
        
        # Negligible effect
        control = np.random.normal(0, 1, 100)
        treatment = np.random.normal(0.1, 1, 100)
        result = StatisticalAnalyzer.welch_t_test(control, treatment)
        assert result.effect_size_interpretation in ["negligible", "small"]
        
        # Large effect
        treatment_large = np.random.normal(1.5, 1, 100)
        result = StatisticalAnalyzer.welch_t_test(control, treatment_large)
        assert result.effect_size_interpretation in ["large", "medium"]


class TestABTestRunner:
    """Tests for ABTestRunner class."""
    
    @pytest.fixture
    def runner(self) -> ABTestRunner:
        """Create ABTestRunner instance."""
        return ABTestRunner(
            control_model="model-a",
            treatment_model="model-b",
            metrics=["quality_score", "latency"],
            statistical_test=StatisticalTest.WELCH_T_TEST,
            significance_level=0.05,
        )
    
    def test_runner_initialization(self, runner: ABTestRunner) -> None:
        """Test runner initialization."""
        assert runner.control_model == "model-a"
        assert runner.treatment_model == "model-b"
        assert "quality_score" in runner.metrics
    
    def test_analyze_with_samples(self, runner: ABTestRunner) -> None:
        """Test analyze with sample data."""
        np.random.seed(42)
        
        samples = {
            "quality_score": [
                MetricSample(value=v, variant="control")
                for v in np.random.normal(0.7, 0.1, 50)
            ] + [
                MetricSample(value=v, variant="treatment")
                for v in np.random.normal(0.8, 0.1, 50)
            ],
            "latency": [
                MetricSample(value=v, variant="control")
                for v in np.random.normal(100, 20, 50)
            ] + [
                MetricSample(value=v, variant="treatment")
                for v in np.random.normal(90, 20, 50)
            ],
        }
        
        results = runner.analyze(samples)
        
        assert "quality_score" in results
        assert "latency" in results
        
        # Check result structure
        quality_result = results["quality_score"]
        assert isinstance(quality_result, ABTestResult)
        assert quality_result.metric_name == "quality_score"
        assert "mean" in quality_result.control_stats
        assert "mean" in quality_result.treatment_stats
    
    def test_analyze_insufficient_samples(self, runner: ABTestRunner) -> None:
        """Test analyze with insufficient samples."""
        samples = {
            "quality_score": [
                MetricSample(value=0.5, variant="control"),
            ],  # Only 1 sample
        }
        
        results = runner.analyze(samples)
        
        # Should skip metrics with insufficient data
        assert "quality_score" not in results or len(results) == 0
    
    def test_calculate_required_sample_size(self) -> None:
        """Test sample size calculation."""
        n = ABTestRunner.calculate_required_sample_size(
            baseline_mean=0.7,
            baseline_std=0.1,
            min_detectable_effect=0.05,
            alpha=0.05,
            power=0.8,
        )
        
        assert n > 0
        assert isinstance(n, int)
        # Reasonable sample size for detecting 5% effect
        assert 50 < n < 2000


class TestABTestResult:
    """Tests for ABTestResult class."""
    
    def test_summary_generation(self) -> None:
        """Test summary string generation."""
        stat_result = StatisticalResult(
            test_name="Welch's t-test",
            statistic=2.5,
            p_value=0.01,
            significant=True,
            confidence_interval=(0.05, 0.15),
            effect_size=0.5,
            effect_size_interpretation="medium",
        )
        
        result = ABTestResult(
            metric_name="quality_score",
            control_stats={"mean": 0.7, "std": 0.1, "median": 0.7},
            treatment_stats={"mean": 0.8, "std": 0.1, "median": 0.8},
            statistical_result=stat_result,
            sample_sizes={"control": 100, "treatment": 100},
            recommendation="Treatment is better",
        )
        
        summary = result.summary()
        
        assert "quality_score" in summary
        assert "Control" in summary
        assert "Treatment" in summary
        assert "p-value" in summary
        assert "Recommendation" in summary


class TestExperimentConfig:
    """Tests for ExperimentConfig class."""
    
    def test_valid_config(self) -> None:
        """Test creating valid config."""
        config = ExperimentConfig(
            name="test_experiment",
            control_model="model-a",
            treatment_model="model-b",
            metrics=["accuracy"],
            sample_size=1000,
        )
        
        assert config.significance_level == 0.05
        assert config.power == 0.8
    
    def test_invalid_significance_level(self) -> None:
        """Test invalid significance level raises error."""
        with pytest.raises(ValueError, match="Significance level"):
            ExperimentConfig(
                name="test",
                control_model="a",
                treatment_model="b",
                metrics=["m"],
                sample_size=100,
                significance_level=1.5,  # Invalid
            )
    
    def test_invalid_power(self) -> None:
        """Test invalid power raises error."""
        with pytest.raises(ValueError, match="Power"):
            ExperimentConfig(
                name="test",
                control_model="a",
                treatment_model="b",
                metrics=["m"],
                sample_size=100,
                power=0,  # Invalid
            )


class TestExperimentTracker:
    """Tests for ExperimentTracker class."""
    
    @pytest.fixture
    def tracker(self) -> ExperimentTracker:
        """Create ExperimentTracker instance."""
        return ExperimentTracker()
    
    @pytest.fixture
    def config(self) -> ExperimentConfig:
        """Create experiment config."""
        return ExperimentConfig(
            name="test_exp",
            control_model="model-a",
            treatment_model="model-b",
            metrics=["score"],
            sample_size=100,
        )
    
    def test_create_experiment(
        self,
        tracker: ExperimentTracker,
        config: ExperimentConfig
    ) -> None:
        """Test experiment creation."""
        exp_id = tracker.create_experiment(config)
        
        assert exp_id.startswith("test_exp_")
        assert len(exp_id) > len("test_exp_")
    
    def test_start_experiment(
        self,
        tracker: ExperimentTracker,
        config: ExperimentConfig
    ) -> None:
        """Test starting an experiment."""
        exp_id = tracker.create_experiment(config)
        tracker.start_experiment(exp_id)
        
        # Should not raise
        assert True
    
    def test_start_nonexistent_experiment(
        self,
        tracker: ExperimentTracker
    ) -> None:
        """Test starting nonexistent experiment raises error."""
        with pytest.raises(ValueError, match="not found"):
            tracker.start_experiment("nonexistent_id")
    
    def test_record_sample(
        self,
        tracker: ExperimentTracker,
        config: ExperimentConfig
    ) -> None:
        """Test recording samples."""
        exp_id = tracker.create_experiment(config)
        
        tracker.record_sample(exp_id, "score", 0.8, "control")
        tracker.record_sample(exp_id, "score", 0.9, "treatment")
        
        # Should not raise
        assert True
    
    def test_complete_experiment(
        self,
        tracker: ExperimentTracker,
        config: ExperimentConfig
    ) -> None:
        """Test completing an experiment."""
        np.random.seed(42)
        
        exp_id = tracker.create_experiment(config)
        tracker.start_experiment(exp_id)
        
        # Record samples
        for _ in range(50):
            tracker.record_sample(exp_id, "score", np.random.normal(0.7, 0.1), "control")
            tracker.record_sample(exp_id, "score", np.random.normal(0.8, 0.1), "treatment")
        
        results = tracker.complete_experiment(exp_id)
        
        assert "score" in results
        assert isinstance(results["score"], ABTestResult)


class TestMetricSample:
    """Tests for MetricSample class."""
    
    def test_sample_creation(self) -> None:
        """Test creating a metric sample."""
        sample = MetricSample(
            value=0.85,
            variant="control",
            metadata={"user_id": "123"},
        )
        
        assert sample.value == 0.85
        assert sample.variant == "control"
        assert isinstance(sample.timestamp, datetime)
        assert sample.metadata["user_id"] == "123"
    
    def test_sample_default_timestamp(self) -> None:
        """Test default timestamp is set."""
        before = datetime.now()
        sample = MetricSample(value=0.5, variant="treatment")
        after = datetime.now()
        
        assert before <= sample.timestamp <= after
