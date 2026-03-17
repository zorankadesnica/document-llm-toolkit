"""
Utility Functions

Common utilities for text processing, logging, and other helpers.
"""

import re
import unicodedata
from typing import Any

from loguru import logger


def normalize_text(text: str) -> str:
    """
    Normalize text for consistent processing.
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)
    
    # Replace multiple whitespaces with single space
    text = re.sub(r"\s+", " ", text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def truncate_text(
    text: str,
    max_length: int,
    suffix: str = "...",
) -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Input text
        max_length: Maximum character length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[: max_length - len(suffix)] + suffix


def count_tokens_approx(text: str) -> int:
    """
    Approximate token count (rough estimate).
    
    Uses simple word/character heuristics.
    For accurate counts, use tiktoken.
    
    Args:
        text: Input text
        
    Returns:
        Approximate token count
    """
    # Rough estimate: ~4 characters per token for English
    return len(text) // 4


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 100,
) -> list[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Input text
        chunk_size: Size of each chunk in characters
        overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence end within last 100 chars of chunk
            search_start = max(end - 100, start)
            for char in [". ", "! ", "? ", "\n"]:
                pos = text.rfind(char, search_start, end)
                if pos != -1:
                    end = pos + 1
                    break
        
        chunks.append(text[start:end].strip())
        start = end - overlap
    
    return chunks


def extract_sentences(text: str) -> list[str]:
    """
    Extract sentences from text.
    
    Args:
        text: Input text
        
    Returns:
        List of sentences
    """
    # Simple sentence tokenization
    # For better results, use nltk.sent_tokenize
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def setup_logging(
    level: str = "INFO",
    log_file: str | None = None,
) -> None:
    """
    Configure logging with loguru.
    
    Args:
        level: Log level
        log_file: Optional log file path
    """
    # Remove default handler
    logger.remove()
    
    # Add console handler
    logger.add(
        sink=lambda msg: print(msg, end=""),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        level=level,
        colorize=True,
    )
    
    # Add file handler if specified
    if log_file:
        logger.add(
            log_file,
            rotation="10 MB",
            retention="7 days",
            level=level,
        )


def safe_dict_get(
    d: dict[str, Any],
    *keys: str,
    default: Any = None,
) -> Any:
    """
    Safely get nested dictionary value.
    
    Args:
        d: Dictionary to search
        *keys: Keys to traverse
        default: Default value if not found
        
    Returns:
        Value or default
    """
    result = d
    for key in keys:
        if isinstance(result, dict):
            result = result.get(key, default)
        else:
            return default
    return result
