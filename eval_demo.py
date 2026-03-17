from src.evaluation.metrics import RougeMetric, BleuMetric, MetricSuite

# Reference and generated text
reference = "The quick brown fox jumps over the lazy dog."
generated = "A fast brown fox leaps over a sleepy dog."

# Individual metrics
rouge = RougeMetric()
bleu = BleuMetric()

print("ROUGE Scores:", rouge.compute(generated, reference))
print("BLEU Score:", bleu.compute(generated, reference))

# Full metric suite
metrics = [RougeMetric(), BleuMetric()]
results = {metric.__class__.__name__: metric.compute(generated, reference) for metric in metrics}
print("\nFull Evaluation:", results)