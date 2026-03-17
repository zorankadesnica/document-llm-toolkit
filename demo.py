"""
DocCopilot AI - Demo Script
Run this to test the project functionality.

Usage:
    python demo.py
"""

import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Note: python-dotenv not installed. Using system environment variables.")


def check_api_keys():
    """Check if API keys are configured."""
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not openai_key and not anthropic_key:
        print("=" * 60)
        print("WARNING: No API keys found!")
        print("=" * 60)
        print("\nTo use LLM features, create a .env file with:")
        print("  OPENAI_API_KEY=sk-your-key-here")
        print("  ANTHROPIC_API_KEY=sk-ant-your-key-here")
        print("\nFor now, running evaluation demos only...")
        print("=" * 60)
        return False
    return True


def demo_evaluation_metrics():
    """Demo: Evaluation Metrics (no API needed)"""
    print("\n" + "=" * 60)
    print("DEMO 1: Evaluation Metrics")
    print("=" * 60)
    
    from src.evaluation.metrics import RougeMetric, BleuMetric, MetricRegistry
    
    reference = "The quick brown fox jumps over the lazy dog near the forest."
    generated = "A fast brown fox leaps over a sleepy dog by the woods."
    
    print(f"\nReference: {reference}")
    print(f"Generated: {generated}")
    
    # ROUGE - expects lists
    rouge = RougeMetric()
    rouge_result = rouge.compute([generated], [reference])
    print(f"\nROUGE Scores:")
    print(f"   Primary Score (ROUGE-L): {rouge_result.score:.4f}")
    if 'rouge_scores' in rouge_result.details:
        for rouge_type, score in rouge_result.details['rouge_scores'].items():
            print(f"   {rouge_type}: {score:.4f}")
    
    # BLEU - expects lists
    bleu = BleuMetric()
    bleu_result = bleu.compute([generated], [reference])
    print(f"\nBLEU Score: {bleu_result.score:.4f}")
    
    # Available metrics
    print(f"\nAll Available Metrics: {MetricRegistry.list_available()}")


def demo_ab_testing():
    """Demo: A/B Testing Framework (no API needed)"""
    print("\n" + "=" * 60)
    print("DEMO 2: A/B Testing Framework")
    print("=" * 60)
    
    import numpy as np
    from src.evaluation.ab_testing import StatisticalAnalyzer
    
    # Simulate scores from two model variants
    np.random.seed(42)
    model_a_scores = np.random.normal(0.72, 0.08, 50)  # Control
    model_b_scores = np.random.normal(0.78, 0.09, 50)  # Treatment
    
    print(f"\nModel A (control): mean={model_a_scores.mean():.3f}, n={len(model_a_scores)}")
    print(f"Model B (treatment): mean={model_b_scores.mean():.3f}, n={len(model_b_scores)}")
    
    analyzer = StatisticalAnalyzer()
    
    t_result = analyzer.welch_t_test(model_a_scores, model_b_scores)

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

    required_n = t_result.effect_size_interpretation
    print(f"\nSample Size Calculation:")
    print(f"   To detect d=0.5 with 80% power: n={required_n} per group")


async def demo_llm_processing():
    """Demo: LLM Document Processing (requires API key)"""
    print("\n" + "=" * 60)
    print("DEMO 5: LLM Document Processing")
    print("=" * 60)
    
    from src.core.processor import DocumentProcessor
    from src.models.providers import LLMProvider
    from src.models.prompts import PromptStyle
    
    # Sample document
    document = """
    Artificial intelligence has transformed numerous industries over the past decade.
    Machine learning models can now process natural language with remarkable accuracy,
    recognize complex patterns in images, and generate creative content that rivals
    human output. Major technology companies are investing billions of dollars in AI
    research and development, racing to build more capable systems.
    
    The healthcare industry has seen particularly significant advances. AI systems
    can now detect certain cancers earlier than human radiologists, predict patient
    outcomes, and accelerate drug discovery. Financial institutions use AI for fraud
    detection, algorithmic trading, and risk assessment.
    
    Experts predict that AI will fundamentally reshape the global economy by 2030,
    creating new job categories while automating others. The key challenge remains
    ensuring these systems are developed responsibly and equitably.
    """
    
    try:
        # Initialize using DocumentProcessor
        processor = DocumentProcessor(
            provider=LLMProvider.OPENAI,
            model="gpt-4o-mini"
        )
        
        # Test 1: Summarization
        print("\nSummarization (Executive Style):")
        print("-" * 40)
        result = await processor.summarize(document, style=PromptStyle.EXECUTIVE)
        print(result.content)
        
        # Test 2: Question Answering
        print("\nQuestion Answering:")
        print("-" * 40)
        question = "What industries have been most impacted by AI?"
        print(f"Q: {question}")
        result = await processor.ask(document, question)
        print(f"A: {result.content}")
        
        # Test 3: Text Enhancement
        print("\nText Enhancement (Clarity):")
        print("-" * 40)
        rough_text = "AI good for business. Many company use it. Results are better than before."
        print(f"Original: {rough_text}")
        result = await processor.enhance(rough_text, aspects=["clarity", "grammar"])
        print(f"Enhanced: {result.content}")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure your API key is valid and you have credits.")


async def demo_llm_judge():
    """Demo: LLM-as-Judge Evaluation (requires API key)"""
    print("\n" + "=" * 60)
    print("DEMO 6: LLM-as-Judge Evaluation")
    print("=" * 60)
    
    from src.evaluation.llm_judge import LLMJudge
    from src.models.providers import LLMProvider
    
    source = """
    The quarterly report shows that revenue increased by 15% year-over-year,
    reaching $2.3 billion. Operating margins improved to 22%, up from 19%
    last year. The company launched three new products and expanded into
    two new markets. Customer satisfaction scores reached an all-time high
    of 92%.
    """
    
    summary_good = """
    Q4 results exceeded expectations with 15% revenue growth to $2.3B.
    Margins improved 3 percentage points to 22%. Growth driven by new
    product launches and market expansion. Customer satisfaction at record 92%.
    """
    
    summary_poor = """
    The company did okay this quarter. Revenue went up and margins got better.
    They made some new stuff and customers seem happy.
    """
    
    try:
        judge = LLMJudge(
            provider=LLMProvider.OPENAI,
            model="gpt-4o-mini",
        )
        
        # Evaluate summary quality
        print("\nEvaluating Good Summary:")
        print("-" * 40)
        result = await judge.evaluate_summary(source, summary_good)
        print(f"Overall Score: {result.overall_score:.2f}")
        for r in result.results:
            print(f"  {r.dimension.value}: {r.score:.2f} - {r.justification[:60]}...")
        
        # Pairwise comparison
        print("\nPairwise Comparison (Good vs Poor):")
        print("-" * 40)
        comparison = await judge.pairwise_compare(
            source=source,
            response_a=summary_good,
            response_b=summary_poor,
            criteria="accuracy and professionalism"
        )
        print(f"Winner: Response {comparison['winner']}")
        print(f"Confidence: {comparison['confidence']}")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure your API key is valid.")


def demo_document_coherence():
    """Demo: Document Coherence Metrics (no API needed)"""
    print("\n" + "=" * 60)
    print("DEMO 3: Document Coherence Evaluation")
    print("=" * 60)
    
    from src.evaluation.document_coherence import DocumentCoherenceEvaluator
    
    # Coherent document
    coherent_doc = [
        "The bank announced record profits today.",
        "The financial institution has been growing steadily.",
        "However, some analysts remain skeptical about the bank.",
        "The bank's CEO addressed these concerns directly.",
    ]
    
    # Incoherent document (inconsistent terminology)
    incoherent_doc = [
        "The bank announced record profits today.",
        "The river flowed steadily past the building.",
        "However, the cat sat on the mat.",
        "The weather was nice outside.",
    ]
    
    evaluator = DocumentCoherenceEvaluator()
    
    print("\nCoherent Document:")
    print("-" * 40)
    result = evaluator.evaluate(coherent_doc)
    print(f"  Lexical Cohesion: {result.lexical_cohesion:.3f}")
    print(f"  Semantic Coherence: {result.semantic_coherence:.3f}")
    print(f"  Overall Score: {result.overall_score:.3f}")
    
    print("\nIncoherent Document:")
    print("-" * 40)
    result = evaluator.evaluate(incoherent_doc)
    print(f"  Lexical Cohesion: {result.lexical_cohesion:.3f}")
    print(f"  Semantic Coherence: {result.semantic_coherence:.3f}")
    print(f"  Overall Score: {result.overall_score:.3f}")


def demo_complexity_analysis():
    """Demo: Computational Complexity Analysis (no API needed)"""
    print("\n" + "=" * 60)
    print("DEMO 4: Attention Complexity Analysis")
    print("=" * 60)
    
    from src.evaluation.complexity_analysis import (
        AttentionComplexityAnalyzer,
        DocumentConfig,
    )
    
    config = DocumentConfig(
        num_sentences=50,
        avg_sentence_length=30,
        model_dim=512,
        num_heads=8,
        local_context_size=3,
        sparsity_factor=0.1,
    )
    
    analyzer = AttentionComplexityAnalyzer(config)
    results = analyzer.compare_all()
    
    print(f"\nDocument: {config.num_sentences} sentences × {config.avg_sentence_length} tokens")
    print("-" * 50)
    
    for method, result in results.items():
        print(f"\n{method}:")
        print(f"  Operations: {result.theoretical_ops:,}")
        print(f"  Memory: {result.memory_bytes / 1024 / 1024:.2f} MB")
        print(f"  Bottleneck: {result.bottleneck}")
    
    # Speedup
    speedup = results["full_concatenation"].theoretical_ops / results["hierarchical"].theoretical_ops
    print(f"\nHierarchical Speedup: {speedup:.1f}x faster than full concatenation!")


def demo_api_info():
    """Show API endpoints info"""
    print("\n" + "=" * 60)
    print("DEMO 7: API Server")
    print("=" * 60)
    print("""
To start the API server, run:

    uvicorn src.api.main:app --reload

Then open: http://localhost:8000/docs

Available endpoints:
    GET  /health              - Health check
    POST /api/v1/summarize    - Summarize a document
    POST /api/v1/enhance      - Enhance text
    POST /api/v1/ask          - Question answering
    POST /api/v1/evaluate     - Evaluate text with metrics
    GET  /api/v1/metrics      - List available metrics
    POST /api/v1/judge        - LLM-as-Judge evaluation
    POST /api/v1/judge/compare - Pairwise comparison
""")


async def main():
    print("=" * 60)
    print("DocCopilot AI - Demo Suite")
    print("=" * 60)
    
    has_api_keys = check_api_keys()
    
    # Always run these (no API needed)
    demo_evaluation_metrics()
    demo_ab_testing()
    demo_document_coherence()
    demo_complexity_analysis()
    
    # Only run if API keys are available
    if has_api_keys:
        await demo_llm_processing()
        await demo_llm_judge()
    
    demo_api_info()
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())