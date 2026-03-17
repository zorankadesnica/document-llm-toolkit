"""
Document Processor

Main interface for document processing capabilities including
summarization, enhancement, Q&A, and content generation.
"""

from dataclasses import dataclass
from typing import Any

from src.models.providers import (
    BaseLLMProvider,
    LLMConfig,
    LLMProvider,
    LLMResponse,
    Message,
    ProviderFactory,
)
from src.models.prompts import PromptLibrary, PromptStyle, ToneStyle


@dataclass
class ProcessingResult:
    """Result of a document processing operation."""
    content: str
    operation: str
    model: str
    tokens_used: int
    metadata: dict[str, Any] | None = None


class DocumentProcessor:
    """
    Main document processor that provides high-level document AI capabilities.
    
    This class serves as the primary interface for document processing tasks
    including summarization, content enhancement, question answering, and
    content generation.
    
    Example:
        >>> processor = DocumentProcessor(
        ...     provider=LLMProvider.OPENAI,
        ...     model="gpt-4",
        ...     api_key="your-api-key"
        ... )
        >>> result = await processor.summarize(
        ...     document="Long document text...",
        ...     style=PromptStyle.EXECUTIVE,
        ...     max_length=200
        ... )
        >>> print(result.content)
    """
    
    def __init__(
        self,
        provider: LLMProvider = LLMProvider.OPENAI,
        model: str = "gpt-4",
        api_key: str | None = None,
        **kwargs: Any,
    ):
        """
        Initialize the document processor.
        
        Args:
            provider: The LLM provider to use
            model: The model name/identifier
            api_key: API key for the provider
            **kwargs: Additional configuration options
        """
        config = LLMConfig(
            provider=provider,
            model=model,
            api_key=api_key,
            **kwargs,
        )
        self._llm: BaseLLMProvider = ProviderFactory.create(config)
        self._model = model
    
    async def summarize(
        self,
        document: str,
        style: PromptStyle = PromptStyle.DETAILED,
        max_length: int = 300,
        **kwargs: Any,
    ) -> ProcessingResult:
        """
        Generate a summary of the document.
        
        Args:
            document: The document text to summarize
            style: The summarization style (executive, detailed, etc.)
            max_length: Maximum length of summary in words
            **kwargs: Additional LLM parameters
            
        Returns:
            ProcessingResult containing the summary
        """
        # Build system message
        system_content = PromptLibrary.SUMMARIZE_SYSTEM.format(
            style=style.value,
            max_length=max_length,
        )
        
        # Get appropriate prompt template
        prompt_template = PromptLibrary.get_summarize_prompt(style)
        user_content = prompt_template.format(document=document)
        
        messages = [
            Message(role="system", content=system_content),
            Message(role="user", content=user_content),
        ]
        
        response = await self._llm.complete(messages, **kwargs)
        
        return ProcessingResult(
            content=response.content,
            operation="summarize",
            model=response.model,
            tokens_used=response.total_tokens,
            metadata={
                "style": style.value,
                "max_length": max_length,
                "finish_reason": response.finish_reason,
            },
        )
    
    async def enhance(
        self,
        text: str,
        aspects: list[str] | None = None,
        target_tone: ToneStyle = ToneStyle.PROFESSIONAL,
        **kwargs: Any,
    ) -> ProcessingResult:
        """
        Enhance the text for clarity, grammar, and style.
        
        Args:
            text: The text to enhance
            aspects: Aspects to focus on (clarity, grammar, conciseness, tone)
            target_tone: The desired tone for the enhanced text
            **kwargs: Additional LLM parameters
            
        Returns:
            ProcessingResult containing the enhanced text
        """
        aspects = aspects or ["clarity", "grammar", "conciseness"]
        
        # Build system message
        system_content = PromptLibrary.ENHANCE_SYSTEM.format(
            tone=target_tone.value,
            focus_areas=", ".join(aspects),
        )
        
        # Use complete enhancement prompt
        user_content = PromptLibrary.ENHANCE_COMPLETE.format(
            text=text,
            target_tone=target_tone.value,
        )
        
        messages = [
            Message(role="system", content=system_content),
            Message(role="user", content=user_content),
        ]
        
        response = await self._llm.complete(messages, **kwargs)
        
        return ProcessingResult(
            content=response.content,
            operation="enhance",
            model=response.model,
            tokens_used=response.total_tokens,
            metadata={
                "aspects": aspects,
                "target_tone": target_tone.value,
            },
        )
    
    async def ask(
        self,
        document: str,
        question: str,
        use_chain_of_thought: bool = False,
        include_evidence: bool = True,
        **kwargs: Any,
    ) -> ProcessingResult:
        """
        Answer a question based on the document.
        
        Args:
            document: The document to use as context
            question: The question to answer
            use_chain_of_thought: Whether to use step-by-step reasoning
            include_evidence: Whether to include supporting evidence
            **kwargs: Additional LLM parameters
            
        Returns:
            ProcessingResult containing the answer
        """
        system_content = PromptLibrary.QA_SYSTEM.format()
        
        # Select appropriate prompt based on options
        if use_chain_of_thought:
            user_content = PromptLibrary.QA_CHAIN_OF_THOUGHT.format(
                document=document,
                question=question,
            )
        elif include_evidence:
            user_content = PromptLibrary.QA_WITH_EVIDENCE.format(
                document=document,
                question=question,
            )
        else:
            user_content = PromptLibrary.QA_BASIC.format(
                document=document,
                question=question,
            )
        
        messages = [
            Message(role="system", content=system_content),
            Message(role="user", content=user_content),
        ]
        
        response = await self._llm.complete(messages, **kwargs)
        
        return ProcessingResult(
            content=response.content,
            operation="qa",
            model=response.model,
            tokens_used=response.total_tokens,
            metadata={
                "question": question,
                "use_chain_of_thought": use_chain_of_thought,
                "include_evidence": include_evidence,
            },
        )
    
    async def generate_outline(
        self,
        topic: str,
        audience: str,
        doc_type: str = "article",
        length: str = "medium",
        key_points: list[str] | None = None,
        **kwargs: Any,
    ) -> ProcessingResult:
        """
        Generate an outline for a document.
        
        Args:
            topic: The main topic of the document
            audience: Target audience description
            doc_type: Type of document (article, report, etc.)
            length: Approximate length (short, medium, long)
            key_points: Key points to cover
            **kwargs: Additional LLM parameters
            
        Returns:
            ProcessingResult containing the outline
        """
        key_points_str = ", ".join(key_points) if key_points else "Not specified"
        
        user_content = PromptLibrary.GENERATE_OUTLINE.format(
            topic=topic,
            audience=audience,
            doc_type=doc_type,
            length=length,
            key_points=key_points_str,
        )
        
        messages = [
            Message(role="user", content=user_content),
        ]
        
        response = await self._llm.complete(messages, **kwargs)
        
        return ProcessingResult(
            content=response.content,
            operation="generate_outline",
            model=response.model,
            tokens_used=response.total_tokens,
            metadata={
                "topic": topic,
                "doc_type": doc_type,
            },
        )
    
    async def extract_key_sentences(
        self,
        document: str,
        num_sentences: int = 5,
        **kwargs: Any,
    ) -> ProcessingResult:
        """
        Extract the most important sentences from a document.
        
        Args:
            document: The document to extract from
            num_sentences: Number of sentences to extract
            **kwargs: Additional LLM parameters
            
        Returns:
            ProcessingResult containing extracted sentences
        """
        user_content = PromptLibrary.SUMMARIZE_EXTRACTIVE.format(
            document=document,
            num_sentences=num_sentences,
        )
        
        messages = [
            Message(role="user", content=user_content),
        ]
        
        response = await self._llm.complete(messages, **kwargs)
        
        return ProcessingResult(
            content=response.content,
            operation="extract_key_sentences",
            model=response.model,
            tokens_used=response.total_tokens,
            metadata={
                "num_sentences": num_sentences,
            },
        )


class BatchProcessor:
    """Process multiple documents in batch."""
    
    def __init__(self, processor: DocumentProcessor):
        """
        Initialize the batch processor.
        
        Args:
            processor: The document processor to use
        """
        self._processor = processor
    
    async def summarize_batch(
        self,
        documents: list[str],
        style: PromptStyle = PromptStyle.DETAILED,
        max_length: int = 300,
    ) -> list[ProcessingResult]:
        """
        Summarize multiple documents.
        
        Args:
            documents: List of documents to summarize
            style: Summarization style
            max_length: Maximum length per summary
            
        Returns:
            List of ProcessingResult objects
        """
        import asyncio
        
        tasks = [
            self._processor.summarize(doc, style=style, max_length=max_length)
            for doc in documents
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(
                    ProcessingResult(
                        content=f"Error processing document {i}: {str(result)}",
                        operation="summarize",
                        model="unknown",
                        tokens_used=0,
                        metadata={"error": True},
                    )
                )
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def ask_batch(
        self,
        document: str,
        questions: list[str],
    ) -> list[ProcessingResult]:
        """
        Answer multiple questions about a document.
        
        Args:
            document: The document to query
            questions: List of questions to answer
            
        Returns:
            List of ProcessingResult objects
        """
        import asyncio
        
        tasks = [
            self._processor.ask(document, question)
            for question in questions
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(
                    ProcessingResult(
                        content=f"Error processing question {i}: {str(result)}",
                        operation="qa",
                        model="unknown",
                        tokens_used=0,
                        metadata={"error": True, "question": questions[i]},
                    )
                )
            else:
                processed_results.append(result)
        
        return processed_results
