from .base import BaseTokenizer


class WordTokenizer(BaseTokenizer):
    name = "word"

    def tokenize(self, line: str) -> list[str]:
        return line.split() + ["<eos>"]
