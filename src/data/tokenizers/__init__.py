from .base import BaseTokenizer
from .char import CharTokenizer
from .word import WordTokenizer

# Maps tokenizer name → class.  Add new tokenizers here.
_REGISTRY: dict = {
    "word": WordTokenizer,
    "char": CharTokenizer,
}

__all__ = [
    "BaseTokenizer",
    "WordTokenizer",
    "CharTokenizer",
    "_REGISTRY",
    "get_tokenizer",
    "from_ifwords",
]


def get_tokenizer(name: str) -> BaseTokenizer:
    """Return a tokenizer instance by name."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown tokenizer {name!r}. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()


def from_ifwords(ifwords: int) -> BaseTokenizer:
    """Backward-compatible shim: 1 → word, 0 → char."""
    return get_tokenizer("word" if ifwords == 1 else "char")
