from .base import BaseTokenizer


class MidiTokenizer(BaseTokenizer):
    """Tokenizer for MIDI token sequences produced by dataset/download_music.py.

    Input lines are already space-separated tokens of the form
    p<pitch>, d<duration>, r<rest>  (e.g. "p60 d8 r4 p62 d16").
    This tokenizer just splits on whitespace and appends <eos>, matching
    the granularity used by the word tokenizer for text.
    """

    name = "midi"

    def tokenize(self, line: str) -> list[str]:
        return line.split() + ["<eos>"]
