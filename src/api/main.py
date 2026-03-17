"""
FastAPI Application for Document AI Services

RESTful API for document processing, evaluation, and agent capabilities.
"""

import os
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Look for .env in project root (parent of src/)
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv not installed, use system env vars

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.core.processor import DocumentProcessor, ProcessingResult
from src.models.providers import LLMProvider
from src.models.prompts import PromptStyle, ToneStyle


# =============================================================================
# Pydantic Models
# =============================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    version: str = "0.1.0"


class SummarizeRequest(BaseModel):
    """Request model for summarization."""
    document: str = Field(..., min_length=1, description="Document text to summarize")
    style: str = Field(
        default="detailed",
        description="Summarization style: executive, detailed, bullet_points"
    )
    max_length: int = Field(
        default=300,
        ge=50,
        le=2000,
        description="Maximum summary length in words"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "document": "Your long document text here...",
                "style": "executive",
                "max_length": 200
            }
        }


class SummarizeResponse(BaseModel):
    """Response model for summarization."""
    summary: str
    model: str
    tokens_used: int
    style: str


class EnhanceRequest(BaseModel):
    """Request model for content enhancement."""
    text: str = Field(..., min_length=1, description="Text to enhance")
    aspects: list[str] = Field(
        default=["clarity", "grammar", "conciseness"],
        description="Aspects to focus on"
    )
    target_tone: str = Field(
        default="professional",
        description="Target tone: professional, friendly, formal, academic"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "This text needs improvement...",
                "aspects": ["clarity", "grammar"],
                "target_tone": "professional"
            }
        }


class EnhanceResponse(BaseModel):
    """Response model for enhancement."""
    enhanced_text: str
    model: str
    tokens_used: int
    aspects: list[str]


class QARequest(BaseModel):
    """Request model for question answering."""
    document: str = Field(..., min_length=1, description="Document context")
    question: str = Field(..., min_length=1, description="Question to answer")
    use_chain_of_thought: bool = Field(
        default=False,
        description="Use step-by-step reasoning"
    )
    include_evidence: bool = Field(
        default=True,
        description="Include supporting evidence"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "document": "The quarterly report shows...",
                "question": "What were the main findings?",
                "use_chain_of_thought": True
            }
        }


class QAResponse(BaseModel):
    """Response model for question answering."""
    answer: str
    model: str
    tokens_used: int
    question: str


class EvaluateRequest(BaseModel):
    """Request model for evaluation."""
    predictions: list[str] = Field(
        ...,
        min_length=1,
        description="Generated texts to evaluate"
    )
    references: list[str] = Field(
        ...,
        min_length=1,
        description="Reference texts"
    )
    metrics: list[str] = Field(
        default=["rouge", "bertscore"],
        description="Metrics to compute"
    )


class EvaluateResponse(BaseModel):
    """Response model for evaluation."""
    overall_score: float
    metrics: dict[str, float]
    details: dict[str, Any]


class AgentRequest(BaseModel):
    """Request model for agent tasks."""
    task: str = Field(..., min_length=1, description="Task to accomplish")
    context: str | None = Field(
        default=None,
        description="Optional document context"
    )
    max_iterations: int = Field(
        default=10,
        ge=1,
        le=20,
        description="Maximum reasoning iterations"
    )


class AgentResponse(BaseModel):
    """Response model for agent tasks."""
    answer: str
    success: bool
    iterations: int
    trace: str | None = None


# =============================================================================
# Application Setup
# =============================================================================

# Global processor instance
processor: DocumentProcessor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    global processor
    # Startup: Initialize processor
    # In production, you'd load config from environment
    processor = DocumentProcessor(
        provider=LLMProvider.OPENAI,
        model="gpt-4",
    )
    yield
    # Shutdown: Cleanup
    processor = None


app = FastAPI(
    title="DocCopilot AI",
    description="Intelligent Document Processing API powered by LLMs",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Routes
# =============================================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse()


@app.post(
    "/api/v1/summarize",
    response_model=SummarizeResponse,
    tags=["Document Processing"],
)
async def summarize_document(request: SummarizeRequest) -> SummarizeResponse:
    """
    Summarize a document.
    
    Generates a summary of the provided document using the specified style
    and length constraints.
    """
    if processor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not initialized"
        )
    
    try:
        style_map = {
            "executive": PromptStyle.EXECUTIVE,
            "detailed": PromptStyle.DETAILED,
            "bullet_points": PromptStyle.BULLET_POINTS,
            "technical": PromptStyle.TECHNICAL,
        }
        style = style_map.get(request.style.lower(), PromptStyle.DETAILED)
        
        result = await processor.summarize(
            document=request.document,
            style=style,
            max_length=request.max_length,
        )
        
        return SummarizeResponse(
            summary=result.content,
            model=result.model,
            tokens_used=result.tokens_used,
            style=request.style,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post(
    "/api/v1/enhance",
    response_model=EnhanceResponse,
    tags=["Document Processing"],
)
async def enhance_content(request: EnhanceRequest) -> EnhanceResponse:
    """
    Enhance text content.
    
    Improves the text for clarity, grammar, and style based on
    the specified aspects and target tone.
    """
    if processor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not initialized"
        )
    
    try:
        tone_map = {
            "professional": ToneStyle.PROFESSIONAL,
            "friendly": ToneStyle.FRIENDLY,
            "formal": ToneStyle.FORMAL,
            "academic": ToneStyle.ACADEMIC,
        }
        tone = tone_map.get(request.target_tone.lower(), ToneStyle.PROFESSIONAL)
        
        result = await processor.enhance(
            text=request.text,
            aspects=request.aspects,
            target_tone=tone,
        )
        
        return EnhanceResponse(
            enhanced_text=result.content,
            model=result.model,
            tokens_used=result.tokens_used,
            aspects=request.aspects,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post(
    "/api/v1/ask",
    response_model=QAResponse,
    tags=["Document Processing"],
)
async def ask_question(request: QARequest) -> QAResponse:
    """
    Answer a question about a document.
    
    Uses the document as context to answer the provided question,
    optionally using chain-of-thought reasoning.
    """
    if processor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not initialized"
        )
    
    try:
        result = await processor.ask(
            document=request.document,
            question=request.question,
            use_chain_of_thought=request.use_chain_of_thought,
            include_evidence=request.include_evidence,
        )
        
        return QAResponse(
            answer=result.content,
            model=result.model,
            tokens_used=result.tokens_used,
            question=request.question,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post(
    "/api/v1/evaluate",
    response_model=EvaluateResponse,
    tags=["Evaluation"],
)
async def evaluate_outputs(request: EvaluateRequest) -> EvaluateResponse:
    """
    Evaluate generated outputs against references.
    
    Computes specified metrics comparing predictions to references.
    """
    try:
        from src.evaluation.metrics import (
            BertScoreMetric,
            BleuMetric,
            EvaluationResult,
            MetricRegistry,
            RougeMetric,
        )
        
        if len(request.predictions) != len(request.references):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Number of predictions must match references"
            )
        
        results = []
        for metric_name in request.metrics:
            try:
                metric = MetricRegistry.get(metric_name)
                result = metric.compute(request.predictions, request.references)
                results.append(result)
            except ValueError:
                continue  # Skip unknown metrics
        
        if not results:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"No valid metrics specified. Available: {MetricRegistry.list_available()}"
            )
        
        eval_result = EvaluationResult(metrics=results)
        
        return EvaluateResponse(
            overall_score=eval_result.overall_score or 0.0,
            metrics={m.metric_name: m.score for m in results},
            details={m.metric_name: m.details for m in results},
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get(
    "/api/v1/metrics",
    tags=["Evaluation"],
)
async def list_metrics() -> dict[str, list[str]]:
    """List available evaluation metrics."""
    from src.evaluation.metrics import MetricRegistry
    
    return {"available_metrics": MetricRegistry.list_available()}


# =============================================================================
# LLM-as-Judge Endpoints
# =============================================================================

class LLMJudgeRequest(BaseModel):
    """Request model for LLM-based evaluation."""
    source: str = Field(..., min_length=1, description="Source document")
    generated: str = Field(..., min_length=1, description="Generated text to evaluate")
    evaluation_type: str = Field(
        default="summary",
        description="Type: summary, factuality, coherence"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "source": "The quarterly report shows revenue increased 15%...",
                "generated": "Revenue grew by 15% this quarter.",
                "evaluation_type": "summary"
            }
        }


class LLMJudgeResponse(BaseModel):
    """Response model for LLM-based evaluation."""
    overall_score: float
    dimensions: dict[str, dict[str, Any]]
    raw_response: str | None = None


class PairwiseCompareRequest(BaseModel):
    """Request for pairwise comparison."""
    source: str = Field(..., description="Source document")
    response_a: str = Field(..., description="First response")
    response_b: str = Field(..., description="Second response")
    criteria: str = Field(default="overall quality", description="Evaluation criteria")


class PairwiseCompareResponse(BaseModel):
    """Response from pairwise comparison."""
    winner: str  # "A", "B", or "TIE"
    confidence: str  # "HIGH", "MEDIUM", "LOW"
    analysis: str


@app.post(
    "/api/v1/judge",
    response_model=LLMJudgeResponse,
    tags=["LLM Evaluation"],
)
async def llm_judge_evaluate(request: LLMJudgeRequest) -> LLMJudgeResponse:
    """
    Evaluate generated text using LLM-as-Judge.
    
    Uses a large language model to assess quality across multiple
    dimensions including coherence, relevance, and factuality.
    """
    try:
        from src.evaluation.llm_judge import LLMJudge
        
        judge = LLMJudge(
            provider=LLMProvider.OPENAI,
            model="gpt-4",
        )
        
        if request.evaluation_type == "summary":
            result = await judge.evaluate_summary(
                source=request.source,
                summary=request.generated,
            )
            return LLMJudgeResponse(
                overall_score=result.overall_score,
                dimensions=result.to_dict()["dimensions"],
                raw_response=result.raw_response,
            )
        
        elif request.evaluation_type == "factuality":
            result = await judge.evaluate_factuality(
                source=request.source,
                generated=request.generated,
            )
            return LLMJudgeResponse(
                overall_score=result.score,
                dimensions={
                    result.dimension.value: {
                        "score": result.score,
                        "raw_score": result.raw_score,
                        "justification": result.justification,
                    }
                },
            )
        
        elif request.evaluation_type == "coherence":
            result = await judge.evaluate_coherence(text=request.generated)
            return LLMJudgeResponse(
                overall_score=result.score,
                dimensions={
                    result.dimension.value: {
                        "score": result.score,
                        "raw_score": result.raw_score,
                        "justification": result.justification,
                    }
                },
            )
        
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown evaluation type: {request.evaluation_type}"
            )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post(
    "/api/v1/judge/compare",
    response_model=PairwiseCompareResponse,
    tags=["LLM Evaluation"],
)
async def pairwise_compare(request: PairwiseCompareRequest) -> PairwiseCompareResponse:
    """
    Compare two responses using LLM-as-Judge.
    
    Determines which response is better according to specified criteria.
    """
    try:
        from src.evaluation.llm_judge import LLMJudge
        
        judge = LLMJudge(
            provider=LLMProvider.OPENAI,
            model="gpt-4",
        )
        
        result = await judge.pairwise_compare(
            source=request.source,
            response_a=request.response_a,
            response_b=request.response_b,
            criteria=request.criteria,
        )
        
        return PairwiseCompareResponse(
            winner=result["winner"],
            confidence=result["confidence"],
            analysis=result["full_analysis"],
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)