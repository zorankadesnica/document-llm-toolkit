import numpy as np
from src.evaluation.ab_testing import StatisticalAnalyzer, ABTestRunner

# Simulated metric scores from two model variants
control_scores = np.random.normal(0.75, 0.1, 100)    # Model A
treatment_scores = np.random.normal(0.78, 0.1, 100)  # Model B

# Statistical analysis
analyzer = StatisticalAnalyzer()
    
t_result = analyzer.welch_t_test(control_scores, treatment_scores)
print("\nWelch's t-test:")
print(f"   statistic: {t_result.statistic:.4f}")
print(f"   p-value: {t_result.p_value:.4f}")
print(f"   significant: {t_result.significant}")
print(f"\nEffect Size:")
print(f"   Cohen's d: {t_result.effect_size:.3f}")
print(f"   Interpretation: {t_result.effect_size_interpretation}")
print(f"\n95% Confidence Interval for difference:")
print(
    f"   [{t_result.confidence_interval[0]:.4f}, "
    f"{t_result.confidence_interval[1]:.4f}]"
)