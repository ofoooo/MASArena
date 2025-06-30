"""
Utility functions for evaluators.
"""

from .sanitize import sanitize
from .answer_extraction import (
    extract_answer_generic,
    extract_answer_numeric, 
    extract_answer_simple_tags,
    extract_answer  # backward compatibility
)

__all__ = [
    "sanitize",
    "extract_answer_generic",
    "extract_answer_numeric", 
    "extract_answer_simple_tags",
    "extract_answer"
]
