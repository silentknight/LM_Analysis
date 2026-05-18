from .base import BaseTokenizer
from .word import WordTokenizer
from .char import CharTokenizer

__all__ = ["BaseTokenizer", "WordTokenizer", "CharTokenizer", "from_ifwords"]


def from_ifwords(ifwords: int) -> BaseTokenizer:
    """Backward-compatible factory: 1 → WordTokenizer, 0 → CharTokenizer."""
    if ifwords == 1:
        return WordTokenizer()
    return CharTokenizer()
