from .base import BaseTokenizer


class CharTokenizer(BaseTokenizer):
    name = "char"

    def tokenize(self, line: str) -> list[str]:
        # Replace <unk> with a sentinel that cannot appear as a real character
        return list(line.strip().replace("<unk>", "\x00")) + [" "]
