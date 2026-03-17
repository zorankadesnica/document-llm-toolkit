"""
LLM-as-Judge Evaluation Module

Uses large language models to evaluate the quality of generated content
across various dimensions including coherence, relevance, and factuality.
"""

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.models.providers import (
    BaseLLMProvider,
    LLMConfig,
    LLMProvider,
    Message,
    ProviderFactory,
)
from src.models.prompts import PromptLibrary


class EvaluationDimension(Enum):
    """Dimensions for LLM-based evaluation."""
    QUALITY = "quality"
    COHERENCE = "coherence"
    RELEVANCE = "relevance"
    FACTUALITY = "factuality"
    FLUENCY = "fluency"
    COMPLETENESS = "completeness"


@dataclass
class JudgeResult:
    """Result from LLM judge evaluation."""
    dimension: EvaluationDimension
    score: float  # 0-1 normalized score
    raw_score: int  # Original 1-5 scale
    justification: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiDimensionResult:
    """Results from multi-dimension evaluation."""
    results: list[JudgeResult]
    overall_score: float
    raw_response: str | None = None
    
    def get_dimension(self, dimension: EvaluationDimension) -> JudgeResult | None:
        """Get result for a specific dimension."""
        for r in self.results:
            if r.dimension == dimension:
                return r
        return None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_score": self.overall_score,
            "dimensions": {
                r.dimension.value: {
                    "score": r.score,
                    "raw_score": r.raw_score,
                    "justification": r.justification,
                }
                for r in self.results
            },
        }


class LLMJudge:
    """
    LLM-based evaluator for text quality assessment.
    
    Uses a large language model to evaluate generated content
    against various quality dimensions, providing both scores
    and detailed justifications.
    
    Example:
        >>> judge = LLMJudge(provider=LLMProvider.OPENAI, model="gpt-4")
        >>> result = await judge.evaluate_summary(
        ...     source="Original document...",
        ...     summary="Generated summary..."
        ... )
        >>> print(f"Overall: {result.overall_score:.2f}")
    """
    
    def __init__(
        self,
        provider: LLMProvider = LLMProvider.OPENAI,
        model: str = "gpt-4",
        api_key: str | None = None,
        temperature: float = 0.0,  # Low temp for consistent evaluation
        **kwargs: Any,
    ):
        """
        Initialize the LLM judge.
        
        Args:
            provider: LLM provider to use
            model: Model identifier
            api_key: API key for the provider
            temperature: Temperature for generation (low for consistency)
            **kwargs: Additional config options
        """
        config = LLMConfig(
            provider=provider,
            model=model,
            api_key=api_key,
            temperature=temperature,
            **kwargs,
        )
        self._llm: BaseLLMProvider = ProviderFactory.create(config)
    
    def _parse_scores(self, response: str) -> dict[str, tuple[int, str]]:
        """
        Parse scores and justifications from LLM response.
        
        Returns:
            Dict mapping dimension name to (score, justification)
        """
        results = {}
        
        # Pattern to match: "Dimension: score - justification"
        pattern = r"(\w+):\s*(\d)\s*[-–]\s*(.+?)(?=\n\w+:|$)"
        matches = re.findall(pattern, response, re.DOTALL)
        
        for dimension, score, justification in matches:
            dimension_lower = dimension.lower().strip()
            results[dimension_lower] = (int(score), justification.strip())
        
        return results
    
    async def evaluate_summary(
        self,
        source: str,
        summary: str,
    ) -> MultiDimensionResult:
        """
        Evaluate a summary against its source document.
        
        Evaluates on: Accuracy, Coverage, Conciseness, Coherence, Factuality
        
        Args:
            source: The source document
            summary: The generated summary
            
        Returns:
            MultiDimensionResult with scores for each dimension
        """
        system_content = PromptLibrary.EVAL_QUALITY_SYSTEM.format()
        user_content = PromptLibrary.EVAL_SUMMARY_QUALITY.format(
            source=source,
            summary=summary,
        )
        
        messages = [
            Message(role="system", content=system_content),
            Message(role="user", content=user_content),
        ]
        
        response = await self._llm.complete(messages)
        
        # Parse the response
        parsed_scores = self._parse_scores(response.content)
        
        # Map to standard dimensions
        dimension_mapping = {
            "accuracy": EvaluationDimension.QUALITY,
            "coverage": EvaluationDimension.COMPLETENESS,
            "conciseness": EvaluationDimension.QUALITY,
            "coherence": EvaluationDimension.COHERENCE,
            "factuality": EvaluationDimension.FACTUALITY,
        }
        
        results = []
        for dim_name, (raw_score, justification) in parsed_scores.items():
            dim_enum = dimension_mapping.get(dim_name, EvaluationDimension.QUALITY)
            results.append(
                JudgeResult(
                    dimension=dim_enum,
                    score=raw_score / 5.0,  # Normalize to 0-1
                    raw_score=raw_score,
                    justification=justification,
                    metadata={"original_dimension": dim_name},
                )
            )
        
        overall = sum(r.score for r in results) / len(results) if results else 0.0
        
        return MultiDimensionResult(
            results=results,
            overall_score=overall,
            raw_response=response.content,
        )
    
    async def evaluate_factuality(
        self,
        source: str,
        generated: str,
    ) -> JudgeResult:
        """
        Evaluate factual accuracy of generated text against source.
        
        Args:
            source: The source document
            generated: The generated text to evaluate
            
        Returns:
            JudgeResult with factuality assessment
        """
        system_content = PromptLibrary.EVAL_QUALITY_SYSTEM.format()
        user_content = PromptLibrary.EVAL_FACTUALITY.format(
            source=source,
            generated=generated,
        )
        
        messages = [
            Message(role="system", content=system_content),
            Message(role="user", content=user_content),
        ]
        
        response = await self._llm.complete(messages)
        
        # Extract factuality score (percentage)
        score_match = re.search(r"(\d+(?:\.\d+)?)\s*%", response.content)
        if score_match:
            score = float(score_match.group(1)) / 100
        else:
            # Try to extract from structured output
            score = 0.5  # Default if parsing fails
        
        return JudgeResult(
            dimension=EvaluationDimension.FACTUALITY,
            score=score,
            raw_score=int(score * 5),
            justification=response.content,
            metadata={"full_analysis": response.content},
        )
    
    async def evaluate_coherence(
        self,
        text: str,
    ) -> JudgeResult:
        """
        Evaluate the coherence of a text.
        
        Args:
            text: The text to evaluate
            
        Returns:
            JudgeResult with coherence assessment
        """
        system_content = PromptLibrary.EVAL_QUALITY_SYSTEM.format()
        user_content = PromptLibrary.EVAL_COHERENCE.format(text=text)
        
        messages = [
            Message(role="system", content=system_content),
            Message(role="user", content=user_content),
        ]
        
        response = await self._llm.complete(messages)
        
        # Parse scores from response
        parsed = self._parse_scores(response.content)
        
        # Calculate average of coherence-related dimensions
        scores = [s for s, _ in parsed.values()]
        avg_score = sum(scores) / len(scores) if scores else 3
        
        return JudgeResult(
            dimension=EvaluationDimension.COHERENCE,
            score=avg_score / 5.0,
            raw_score=int(avg_score),
            justification=response.content,
            metadata={"detailed_scores": parsed},
        )
    
    async def evaluate_relevance(
        self,
        question: str,
        context: str,
        response_text: str,
    ) -> JudgeResult:
        """
        Evaluate how well a response answers a question.
        
        Args:
            question: The question asked
            context: The context/document
            response_text: The generated response
            
        Returns:
            JudgeResult with relevance assessment
        """
        system_content = PromptLibrary.EVAL_QUALITY_SYSTEM.format()
        user_content = PromptLibrary.EVAL_RELEVANCE.format(
            question=question,
            context=context,
            response=response_text,
        )
        
        messages = [
            Message(role="system", content=system_content),
            Message(role="user", content=user_content),
        ]
        
        response = await self._llm.complete(messages)
        
        # Parse scores
        parsed = self._parse_scores(response.content)
        relevance_score = parsed.get("relevance", (3, ""))[0]
        
        return JudgeResult(
            dimension=EvaluationDimension.RELEVANCE,
            score=relevance_score / 5.0,
            raw_score=relevance_score,
            justification=response.content,
            metadata={"all_scores": parsed},
        )
    
    async def pairwise_compare(
        self,
        source: str,
        response_a: str,
        response_b: str,
        criteria: str = "overall quality",
    ) -> dict[str, Any]:
        """
        Compare two responses and determine which is better.
        
        Args:
            source: The source document/context
            response_a: First response
            response_b: Second response
            criteria: Evaluation criteria
            
        Returns:
            Dict with winner, confidence, and justification
        """
        prompt = f"""Compare these two responses based on {criteria}.

Source/Context:
{source}

Response A:
{response_a}

Response B:
{response_b}

Evaluate both responses and determine which is better.
Provide your analysis in this format:

Analysis of A: [brief analysis]
Analysis of B: [brief analysis]
Winner: [A or B or TIE]
Confidence: [HIGH, MEDIUM, or LOW]
Justification: [why the winner is better]"""

        messages = [
            Message(
                role="system",
                content="You are an expert evaluator. Compare responses objectively."
            ),
            Message(role="user", content=prompt),
        ]
        
        response = await self._llm.complete(messages)
        
        # Parse winner
        winner_match = re.search(r"Winner:\s*([AB]|TIE)", response.content, re.IGNORECASE)
        confidence_match = re.search(
            r"Confidence:\s*(HIGH|MEDIUM|LOW)",
            response.content,
            re.IGNORECASE
        )
        
        return {
            "winner": winner_match.group(1).upper() if winner_match else "TIE",
            "confidence": confidence_match.group(1).upper() if confidence_match else "MEDIUM",
            "full_analysis": response.content,
        }


class BatchJudge:
    """Evaluate multiple samples efficiently."""
    
    def __init__(self, judge: LLMJudge):
        """
        Initialize batch judge.
        
        Args:
            judge: LLMJudge instance to use
        """
        self._judge = judge
    
    async def evaluate_summaries(
        self,
        sources: list[str],
        summaries: list[str],
    ) -> list[MultiDimensionResult]:
        """
        Evaluate multiple summaries.
        
        Args:
            sources: List of source documents
            summaries: List of generated summaries
            
        Returns:
            List of MultiDimensionResult objects
        """
        import asyncio
        
        if len(sources) != len(summaries):
            raise ValueError("Number of sources and summaries must match")
        
        tasks = [
            self._judge.evaluate_summary(source, summary)
            for source, summary in zip(sources, summaries)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Return a default low-score result on error
                processed.append(
                    MultiDimensionResult(
                        results=[],
                        overall_score=0.0,
                        raw_response=f"Error: {str(result)}",
                    )
                )
            else:
                processed.append(result)
        
        return processed
    
    async def pairwise_tournament(
        self,
        source: str,
        responses: list[str],
        criteria: str = "overall quality",
    ) -> dict[str, Any]:
        """
        Run a pairwise tournament to rank responses.
        
        Args:
            source: Source document
            responses: List of responses to compare
            criteria: Evaluation criteria
            
        Returns:
            Dict with rankings and comparison details
        """
        import asyncio
        from itertools import combinations
        
        n = len(responses)
        wins = {i: 0 for i in range(n)}
        comparisons = []
        
        # Generate all pairs
        pairs = list(combinations(range(n), 2))
        
        # Compare all pairs
        tasks = [
            self._judge.pairwise_compare(
                source,
                responses[i],
                responses[j],
                criteria,
            )
            for i, j in pairs
        ]
        
        results = await asyncio.gather(*tasks)
        
        for (i, j), result in zip(pairs, results):
            winner = result["winner"]
            if winner == "A":
                wins[i] += 1
            elif winner == "B":
                wins[j] += 1
            else:  # TIE
                wins[i] += 0.5
                wins[j] += 0.5
            
            comparisons.append({
                "pair": (i, j),
                "winner": i if winner == "A" else (j if winner == "B" else "tie"),
                "confidence": result["confidence"],
            })
        
        # Rank by wins
        ranking = sorted(wins.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "ranking": [{"response_idx": idx, "wins": w} for idx, w in ranking],
            "comparisons": comparisons,
            "total_comparisons": len(comparisons),
        }
