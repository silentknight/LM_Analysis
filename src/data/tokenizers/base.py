from abc import ABC, abstractmethod


class BaseTokenizer(ABC):
    name: str

    @abstractmethod
    def tokenize(self, line: str) -> list[str]:
        """Tokenize a single line into a list of string tokens."""
