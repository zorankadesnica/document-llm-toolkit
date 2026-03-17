# DocCopilot AI

**Document-Level LLM Evaluation and Processing Framework**

Framework for document processing, statistical A/B testing, coherence analysis, and LLM-as-Judge evaluation. Built with FastAPI and designed for rigorous evaluation of language model outputs.

---

## Features

| Category | Capabilities |
|----------|--------------|
| **Evaluation Metrics** | ROUGE (1/2/L), BLEU, BERTScore, Semantic Similarity |
| **Statistical Testing** | Welch's t-test, Cohen's d, Confidence Intervals, Power Analysis |
| **Document Coherence** | Lexical Cohesion, Semantic Coherence, Entity Consistency |
| **Complexity Analysis** | Full vs Hierarchical vs Sparse Attention Comparison |
| **LLM Processing** | Summarization, Q&A, Text Enhancement |
| **LLM-as-Judge** | Multi-dimensional Scoring, Pairwise Comparison, Tournament Ranking |

---

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/doc-copilot-ai.git
cd doc-copilot-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your OPENAI_API_KEY
```

---

## Quick Start

```bash
# Run demo suite
python demo.py

# Start API server
uvicorn src.api.main:app --reload
# Open: http://localhost:8000/docs
```

---

## Demo Suite

| Demo | Module | API Required | Description |
|------|--------|--------------|-------------|
| 1 | Evaluation Metrics | ❌ | ROUGE, BLEU scoring |
| 2 | A/B Testing | ❌ | Welch's t-test, Cohen's d |
| 3 | Document Coherence | ❌ | Lexical & semantic coherence |
| 4 | Complexity Analysis | ❌ | Attention complexity comparison |
| 5 | LLM Processing | ✅ | Summarize, Q&A, enhance |
| 6 | LLM-as-Judge | ✅ | Multi-dim scoring, pairwise |
| 7 | API Info | ❌ | Endpoint documentation |

### Additional Standalone Demos

```bash
# Oracle Experiments (WMT methodology, upper bound analysis)
python src/evaluation/oracle_experiments.py

# Contrastive Evaluation (ContraPro-style, adversarial perturbation)
python src/evaluation/contrastive_evaluation.py
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/api/v1/summarize` | Document summarization |
| POST | `/api/v1/enhance` | Text enhancement |
| POST | `/api/v1/ask` | Question answering |
| POST | `/api/v1/evaluate` | Metric evaluation |
| GET | `/api/v1/metrics` | List available metrics |
| POST | `/api/v1/judge` | LLM-as-Judge scoring |
| POST | `/api/v1/judge/compare` | Pairwise comparison |

---

## Mathematical Foundations

See `DocCopilot_Mathematical_Foundations.tex` for complete derivations.

### Key Formulas

**ROUGE-L F₁:**
```
F = (1 + β²)PR / (R + β²P)
where P = |LCS|/|generated|, R = |LCS|/|reference|
```

**BLEU:**
```
BLEU = BP × exp(Σ wₙ log pₙ)
BP = exp(1 - r/c) if c ≤ r else 1
```

**Welch's t-statistic:**
```
t = (Ȳ - X̄) / √(s²_A/n_A + s²_B/n_B)
```

**Cohen's d:**
```
d = (Ȳ - X̄) / √((s²_A + s²_B)/2)
```

**Sample Size (per group):**
```
n = 2 × ((z_{1-α/2} + z_{1-β}) / d)²
```

**Jaccard Similarity:**
```
J(A, B) = |A ∩ B| / |A ∪ B|
```

**Attention Complexity:**
```
Full:         O(N²L²d)
Hierarchical: O(Nk²L²d + N²d)
Speedup:      ~3-5× for typical documents
```

**Gated Context Fusion:**
```
g = σ(W_l·c_local + W_g·c_global + W_m·h + b)
c_final = g ⊙ c_local + (1-g) ⊙ c_global
```

---

## Project Structure

```
doc-copilot-ai/
├── demo.py                                    # Demo script
├── requirements.txt                           # Dependencies
├── DocCopilot_Mathematical_Foundations.pdf    # Technical report
├── src/
│   ├── api/
│   │   └── main.py                           # FastAPI app
│   ├── core/
│   │   ├── processor.py                      # Document processor
│   │   ├── hierarchical_context.py           # Hierarchical attention
│   │   └── domain_adaptation.py              # Zero-resource adaptation
│   ├── evaluation/
│   │   ├── metrics.py                        # ROUGE, BLEU, BERTScore
│   │   ├── ab_testing.py                     # Statistical testing
│   │   ├── document_coherence.py             # Coherence metrics
│   │   ├── complexity_analysis.py            # Attention analysis
│   │   ├── llm_judge.py                      # LLM-as-Judge
│   │   ├── contrastive_evaluation.py         # Challenge sets
│   │   └── oracle_experiments.py             # Oracle methodology
│   └── models/
│       ├── providers.py                      # LLM providers
│       └── prompts.py                        # Prompt templates
└── tests/
    └── unit/
        ├── test_metrics.py
        └── test_ab_testing.py
```

---

## Usage Examples

### Evaluation Metrics

```python
from src.evaluation.metrics import RougeMetric, BleuMetric

rouge = RougeMetric()
result = rouge.compute(
    predictions=["A fast brown fox leaps over a sleepy dog."],
    references=["The quick brown fox jumps over the lazy dog."]
)
print(f"ROUGE-L: {result.score:.4f}")  # 0.4167
```

### A/B Testing

```python
from src.evaluation.ab_testing import StatisticalAnalyzer
import numpy as np

analyzer = StatisticalAnalyzer()
result = analyzer.welch_t_test(
    control=np.random.normal(0.72, 0.08, 50),
    treatment=np.random.normal(0.78, 0.09, 50)
)
print(f"t-statistic: {result.statistic:.4f}")
print(f"p-value: {result.p_value:.4f}")
print(f"Cohen's d: {result.effect_size:.3f}")
print(f"Interpretation: {result.effect_size_interpretation}")
```

### Document Coherence

```python
from src.evaluation.document_coherence import DocumentCoherenceEvaluator

evaluator = DocumentCoherenceEvaluator()
result = evaluator.evaluate([
    "The bank announced record profits today.",
    "The financial institution has been growing.",
    "The bank's CEO addressed concerns.",
])
print(f"Lexical Cohesion: {result.lexical_cohesion:.3f}")
print(f"Semantic Coherence: {result.semantic_coherence:.3f}")
print(f"Overall: {result.overall_score:.3f}")
```

### Complexity Analysis

```python
from src.evaluation.complexity_analysis import (
    AttentionComplexityAnalyzer,
    DocumentConfig
)

config = DocumentConfig(
    num_sentences=50,
    avg_sentence_length=30,
    model_dim=512,
    num_heads=8,
    local_context_size=3,
    sparsity_factor=0.1
)

analyzer = AttentionComplexityAnalyzer(config)
results = analyzer.compare_all()

for method, result in results.items():
    print(f"{method}: {result.theoretical_ops:,} ops")
```

### LLM-as-Judge

```python
from src.evaluation.llm_judge import LLMJudge
from src.models.providers import LLMProvider

judge = LLMJudge(provider=LLMProvider.OPENAI, model="gpt-4o-mini")

# Multi-dimensional evaluation
result = await judge.evaluate_summary(
    source="Original document...",
    summary="Generated summary..."
)
print(f"Overall: {result.overall_score:.2f}")

# Pairwise comparison
comparison = await judge.pairwise_compare(
    source="Original...",
    response_a="Summary A...",
    response_b="Summary B...",
    criteria="accuracy"
)
print(f"Winner: {comparison['winner']}")
```

---

## Configuration

### Environment Variables

```bash
# .env
OPENAI_API_KEY=sk-your-key-here
ANTHROPIC_API_KEY=sk-ant-your-key-here  # Optional
```

## Testing

```bash
pytest                              # Run all tests
pytest --cov=src --cov-report=html  # With coverage
pytest tests/unit/test_metrics.py   # Specific file
```

---

The  report covers:
1. N-gram metrics (ROUGE, BLEU) with derivations
2. BERTScore and contextual embeddings
3. Welch's t-test and Welch-Satterthwaite degrees of freedom
4. Cohen's d effect size interpretation
5. Sample size power analysis
6. Jaccard similarity for coherence
7. Attention complexity analysis (O(N²L²) vs O(NkL²))
8. Autoregressive language modeling
9. Bradley-Terry model and Elo ratings
10. Hierarchical context with gated fusion
11. Contrastive evaluation for pronouns
12. Zero-resource domain adaptation (FiLM)

---

## References

- Lin, C.Y. (2004). ROUGE: A Package for Automatic Evaluation of Summaries
- Papineni et al. (2002). BLEU: A Method for Automatic Evaluation of Machine Translation
- Zhang et al. (2020). BERTScore: Evaluating Text Generation with BERT
- Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences
- Vaswani et al. (2017). Attention Is All You Need
- Müller et al. (2018). A Large-Scale Test Set for Context-Dependent Pronoun Translation

---

## License

MIT License