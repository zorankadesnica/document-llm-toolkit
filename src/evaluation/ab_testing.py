"""
A/B Testing Framework for LLM Evaluation

Statistical framework for comparing model variants through
controlled experiments with proper statistical analysis.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

import numpy as np
from scipy import stats


class StatisticalTest(Enum):
    """Available statistical tests for A/B analysis."""
    WELCH_T_TEST = "welch_t_test"
    MANN_WHITNEY_U = "mann_whitney_u"
    BOOTSTRAP = "bootstrap"
    PERMUTATION = "permutation"


class ExperimentStatus(Enum):
    """Status of an A/B experiment."""
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    STOPPED = "stopped"


@dataclass
class ExperimentConfig:
    """Configuration for an A/B experiment."""
    name: str
    control_model: str
    treatment_model: str
    metrics: list[str]
    sample_size: int
    significance_level: float = 0.05
    power: float = 0.8
    min_detectable_effect: float = 0.05
    statistical_test: StatisticalTest = StatisticalTest.WELCH_T_TEST
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        if not 0 < self.significance_level < 1:
            raise ValueError("Significance level must be between 0 and 1")
        if not 0 < self.power < 1:
            raise ValueError("Power must be between 0 and 1")


@dataclass
class MetricSample:
    """A single metric observation."""
    value: float
    variant: str  # "control" or "treatment"
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StatisticalResult:
    """Result of a statistical test."""
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    confidence_interval: tuple[float, float]
    effect_size: float
    effect_size_interpretation: str
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_name": self.test_name,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "significant": self.significant,
            "confidence_interval": self.confidence_interval,
            "effect_size": self.effect_size,
            "effect_size_interpretation": self.effect_size_interpretation,
        }


@dataclass
class ABTestResult:
    """Complete results from an A/B test."""
    metric_name: str
    control_stats: dict[str, float]
    treatment_stats: dict[str, float]
    statistical_result: StatisticalResult
    sample_sizes: dict[str, int]
    recommendation: str
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        sig_str = "statistically significant" if self.statistical_result.significant else "not statistically significant"
        return f"""
A/B Test Results for: {self.metric_name}
{'=' * 50}

Control ({self.sample_sizes['control']} samples):
  Mean: {self.control_stats['mean']:.4f}
  Std:  {self.control_stats['std']:.4f}

Treatment ({self.sample_sizes['treatment']} samples):
  Mean: {self.treatment_stats['mean']:.4f}
  Std:  {self.treatment_stats['std']:.4f}

Statistical Analysis:
  Test: {self.statistical_result.test_name}
  p-value: {self.statistical_result.p_value:.4f}
  Effect Size: {self.statistical_result.effect_size:.4f} ({self.statistical_result.effect_size_interpretation})
  Result: {sig_str}

95% CI for difference: [{self.statistical_result.confidence_interval[0]:.4f}, {self.statistical_result.confidence_interval[1]:.4f}]

Recommendation: {self.recommendation}
"""


class StatisticalAnalyzer:
    """Performs statistical analysis for A/B tests."""
    
    @staticmethod
    def welch_t_test(
        control: np.ndarray,
        treatment: np.ndarray,
        alpha: float = 0.05,
    ) -> StatisticalResult:
        """
        Perform Welch's t-test (unequal variances).
        
        Args:
            control: Control group samples
            treatment: Treatment group samples
            alpha: Significance level
            
        Returns:
            StatisticalResult with test outcomes
        """
        statistic, p_value = stats.ttest_ind(
            treatment,
            control,
            equal_var=False,  # Welch's t-test
        )
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(
            (np.var(control) + np.var(treatment)) / 2
        )
        effect_size = (np.mean(treatment) - np.mean(control)) / pooled_std
        
        # Effect size interpretation
        abs_effect = abs(effect_size)
        if abs_effect < 0.2:
            interpretation = "negligible"
        elif abs_effect < 0.5:
            interpretation = "small"
        elif abs_effect < 0.8:
            interpretation = "medium"
        else:
            interpretation = "large"
        
        # Calculate confidence interval for the difference
        mean_diff = np.mean(treatment) - np.mean(control)
        se_diff = np.sqrt(
            np.var(control) / len(control) + 
            np.var(treatment) / len(treatment)
        )
        # Use t-distribution for CI
        df = len(control) + len(treatment) - 2
        t_crit = stats.t.ppf(1 - alpha / 2, df)
        ci_lower = mean_diff - t_crit * se_diff
        ci_upper = mean_diff + t_crit * se_diff
        
        return StatisticalResult(
            test_name="Welch's t-test",
            statistic=float(statistic),
            p_value=float(p_value),
            significant=p_value < alpha,
            confidence_interval=(float(ci_lower), float(ci_upper)),
            effect_size=float(effect_size),
            effect_size_interpretation=interpretation,
        )
    
    @staticmethod
    def mann_whitney_u(
        control: np.ndarray,
        treatment: np.ndarray,
        alpha: float = 0.05,
    ) -> StatisticalResult:
        """
        Perform Mann-Whitney U test (non-parametric).
        
        Args:
            control: Control group samples
            treatment: Treatment group samples
            alpha: Significance level
            
        Returns:
            StatisticalResult with test outcomes
        """
        statistic, p_value = stats.mannwhitneyu(
            treatment,
            control,
            alternative='two-sided',
        )
        
        # Calculate effect size (rank-biserial correlation)
        n1, n2 = len(control), len(treatment)
        effect_size = 1 - (2 * statistic) / (n1 * n2)
        
        abs_effect = abs(effect_size)
        if abs_effect < 0.1:
            interpretation = "negligible"
        elif abs_effect < 0.3:
            interpretation = "small"
        elif abs_effect < 0.5:
            interpretation = "medium"
        else:
            interpretation = "large"
        
        # Bootstrap CI for median difference
        n_bootstrap = 1000
        diffs = []
        for _ in range(n_bootstrap):
            c_sample = np.random.choice(control, len(control), replace=True)
            t_sample = np.random.choice(treatment, len(treatment), replace=True)
            diffs.append(np.median(t_sample) - np.median(c_sample))
        
        ci_lower = np.percentile(diffs, 2.5)
        ci_upper = np.percentile(diffs, 97.5)
        
        return StatisticalResult(
            test_name="Mann-Whitney U",
            statistic=float(statistic),
            p_value=float(p_value),
            significant=p_value < alpha,
            confidence_interval=(float(ci_lower), float(ci_upper)),
            effect_size=float(effect_size),
            effect_size_interpretation=interpretation,
        )
    
    @staticmethod
    def bootstrap_test(
        control: np.ndarray,
        treatment: np.ndarray,
        alpha: float = 0.05,
        n_bootstrap: int = 10000,
        statistic_func: Callable[[np.ndarray], float] = np.mean,
    ) -> StatisticalResult:
        """
        Perform bootstrap hypothesis test.
        
        Args:
            control: Control group samples
            treatment: Treatment group samples
            alpha: Significance level
            n_bootstrap: Number of bootstrap iterations
            statistic_func: Function to compute test statistic
            
        Returns:
            StatisticalResult with test outcomes
        """
        observed_diff = statistic_func(treatment) - statistic_func(control)
        
        # Pool data under null hypothesis
        pooled = np.concatenate([control, treatment])
        n_control = len(control)
        
        # Bootstrap under null
        null_diffs = []
        for _ in range(n_bootstrap):
            np.random.shuffle(pooled)
            bootstrap_control = pooled[:n_control]
            bootstrap_treatment = pooled[n_control:]
            null_diffs.append(
                statistic_func(bootstrap_treatment) - 
                statistic_func(bootstrap_control)
            )
        
        null_diffs = np.array(null_diffs)
        
        # Two-sided p-value
        p_value = np.mean(np.abs(null_diffs) >= np.abs(observed_diff))
        
        # Bootstrap CI for the difference
        boot_diffs = []
        for _ in range(n_bootstrap):
            c_sample = np.random.choice(control, len(control), replace=True)
            t_sample = np.random.choice(treatment, len(treatment), replace=True)
            boot_diffs.append(statistic_func(t_sample) - statistic_func(c_sample))
        
        ci_lower = np.percentile(boot_diffs, 100 * alpha / 2)
        ci_upper = np.percentile(boot_diffs, 100 * (1 - alpha / 2))
        
        # Effect size
        pooled_std = np.std(pooled)
        effect_size = observed_diff / pooled_std if pooled_std > 0 else 0
        
        abs_effect = abs(effect_size)
        if abs_effect < 0.2:
            interpretation = "negligible"
        elif abs_effect < 0.5:
            interpretation = "small"
        elif abs_effect < 0.8:
            interpretation = "medium"
        else:
            interpretation = "large"
        
        return StatisticalResult(
            test_name="Bootstrap",
            statistic=float(observed_diff),
            p_value=float(p_value),
            significant=p_value < alpha,
            confidence_interval=(float(ci_lower), float(ci_upper)),
            effect_size=float(effect_size),
            effect_size_interpretation=interpretation,
        )


class ABTestRunner:
    """
    A/B testing framework for comparing LLM model variants.
    
    Example:
        >>> runner = ABTestRunner(
        ...     control_model="gpt-3.5-turbo",
        ...     treatment_model="gpt-4",
        ...     metrics=["quality_score", "latency_ms"],
        ...     statistical_test=StatisticalTest.WELCH_T_TEST,
        ... )
        >>> results = await runner.run(dataset, sample_size=1000)
        >>> print(runner.analyze(results))
    """
    
    def __init__(
        self,
        control_model: str,
        treatment_model: str,
        metrics: list[str],
        statistical_test: StatisticalTest = StatisticalTest.WELCH_T_TEST,
        significance_level: float = 0.05,
    ):
        """
        Initialize A/B test runner.
        
        Args:
            control_model: Model identifier for control variant
            treatment_model: Model identifier for treatment variant
            metrics: List of metric names to evaluate
            statistical_test: Statistical test to use
            significance_level: Alpha for hypothesis testing
        """
        self.control_model = control_model
        self.treatment_model = treatment_model
        self.metrics = metrics
        self.statistical_test = statistical_test
        self.significance_level = significance_level
        self._analyzer = StatisticalAnalyzer()
    
    def analyze(
        self,
        samples: dict[str, list[MetricSample]],
    ) -> dict[str, ABTestResult]:
        """
        Analyze collected samples and perform statistical tests.
        
        Args:
            samples: Dict mapping metric names to list of samples
            
        Returns:
            Dict mapping metric names to ABTestResult
        """
        results = {}
        
        for metric_name in self.metrics:
            metric_samples = samples.get(metric_name, [])
            
            # Separate by variant
            control_values = np.array([
                s.value for s in metric_samples if s.variant == "control"
            ])
            treatment_values = np.array([
                s.value for s in metric_samples if s.variant == "treatment"
            ])
            
            if len(control_values) < 2 or len(treatment_values) < 2:
                continue
            
            # Perform statistical test
            if self.statistical_test == StatisticalTest.WELCH_T_TEST:
                stat_result = self._analyzer.welch_t_test(
                    control_values,
                    treatment_values,
                    self.significance_level,
                )
            elif self.statistical_test == StatisticalTest.MANN_WHITNEY_U:
                stat_result = self._analyzer.mann_whitney_u(
                    control_values,
                    treatment_values,
                    self.significance_level,
                )
            else:
                stat_result = self._analyzer.bootstrap_test(
                    control_values,
                    treatment_values,
                    self.significance_level,
                )
            
            # Generate recommendation
            if stat_result.significant:
                if stat_result.effect_size > 0:
                    recommendation = f"Treatment ({self.treatment_model}) shows statistically significant improvement. Consider rolling out."
                else:
                    recommendation = f"Treatment ({self.treatment_model}) shows statistically significant degradation. Keep control."
            else:
                recommendation = "No statistically significant difference detected. Need more data or the effect is negligible."
            
            results[metric_name] = ABTestResult(
                metric_name=metric_name,
                control_stats={
                    "mean": float(np.mean(control_values)),
                    "std": float(np.std(control_values)),
                    "median": float(np.median(control_values)),
                },
                treatment_stats={
                    "mean": float(np.mean(treatment_values)),
                    "std": float(np.std(treatment_values)),
                    "median": float(np.median(treatment_values)),
                },
                statistical_result=stat_result,
                sample_sizes={
                    "control": len(control_values),
                    "treatment": len(treatment_values),
                },
                recommendation=recommendation,
            )
        
        return results
    
    @staticmethod
    def calculate_required_sample_size(
        baseline_mean: float,
        baseline_std: float,
        min_detectable_effect: float,
        alpha: float = 0.05,
        power: float = 0.8,
    ) -> int:
        """
        Calculate required sample size per group.
        
        Args:
            baseline_mean: Expected mean of control group
            baseline_std: Expected standard deviation
            min_detectable_effect: Minimum relative effect to detect
            alpha: Significance level
            power: Statistical power
            
        Returns:
            Required sample size per group
        """
        # Effect size in standard deviation units
        effect_size = (baseline_mean * min_detectable_effect) / baseline_std
        
        # Z-scores
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)
        
        # Sample size formula for two-sample t-test
        n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
        
        return int(np.ceil(n))


class ExperimentTracker:
    """Track and manage multiple A/B experiments."""
    
    def __init__(self) -> None:
        """Initialize experiment tracker."""
        self._experiments: dict[str, dict[str, Any]] = {}
        self._samples: dict[str, dict[str, list[MetricSample]]] = {}
    
    def create_experiment(self, config: ExperimentConfig) -> str:
        """
        Create a new experiment.
        
        Args:
            config: Experiment configuration
            
        Returns:
            Experiment ID
        """
        exp_id = f"{config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self._experiments[exp_id] = {
            "config": config,
            "status": ExperimentStatus.PLANNED,
            "created_at": datetime.now(),
            "started_at": None,
            "completed_at": None,
        }
        self._samples[exp_id] = {metric: [] for metric in config.metrics}
        
        return exp_id
    
    def start_experiment(self, exp_id: str) -> None:
        """Start an experiment."""
        if exp_id not in self._experiments:
            raise ValueError(f"Experiment {exp_id} not found")
        
        self._experiments[exp_id]["status"] = ExperimentStatus.RUNNING
        self._experiments[exp_id]["started_at"] = datetime.now()
    
    def record_sample(
        self,
        exp_id: str,
        metric_name: str,
        value: float,
        variant: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Record a metric sample for an experiment.
        
        Args:
            exp_id: Experiment ID
            metric_name: Name of the metric
            value: Metric value
            variant: "control" or "treatment"
            metadata: Additional metadata
        """
        if exp_id not in self._samples:
            raise ValueError(f"Experiment {exp_id} not found")
        
        if metric_name not in self._samples[exp_id]:
            self._samples[exp_id][metric_name] = []
        
        sample = MetricSample(
            value=value,
            variant=variant,
            metadata=metadata or {},
        )
        self._samples[exp_id][metric_name].append(sample)
    
    def get_results(self, exp_id: str) -> dict[str, ABTestResult]:
        """
        Get current results for an experiment.
        
        Args:
            exp_id: Experiment ID
            
        Returns:
            Dict of metric results
        """
        if exp_id not in self._experiments:
            raise ValueError(f"Experiment {exp_id} not found")
        
        config = self._experiments[exp_id]["config"]
        runner = ABTestRunner(
            control_model=config.control_model,
            treatment_model=config.treatment_model,
            metrics=config.metrics,
            statistical_test=config.statistical_test,
            significance_level=config.significance_level,
        )
        
        return runner.analyze(self._samples[exp_id])
    
    def complete_experiment(self, exp_id: str) -> dict[str, ABTestResult]:
        """
        Mark experiment as complete and return final results.
        
        Args:
            exp_id: Experiment ID
            
        Returns:
            Final experiment results
        """
        if exp_id not in self._experiments:
            raise ValueError(f"Experiment {exp_id} not found")
        
        self._experiments[exp_id]["status"] = ExperimentStatus.COMPLETED
        self._experiments[exp_id]["completed_at"] = datetime.now()
        
        return self.get_results(exp_id)