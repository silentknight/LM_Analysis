"""BPE tokenizer backed by HuggingFace tokenizers library.

Workflow:
  1. Train a BPE model on your corpus:
       python dataset/train_bpe.py --data dataset/text/english/ptb \\
           --vocab_size 8000 --output bpe_models/ptb_8k.json
  2. Run analysis with:
       python run_all.py --data dataset/text/english/ptb/.full \\
           --tokenizer bpe --bpe_model bpe_models/ptb_8k.json

The model path is read from the BPE_MODEL_PATH environment variable or the
--bpe_model flag (run_all.py sets this on the class before instantiating).
"""

import os

from .base import BaseTokenizer

# Set by run_all.py (or directly by tests) before get_tokenizer("bpe") is called.
_model_path: str | None = None


class BpeTokenizer(BaseTokenizer):
    name = "bpe"

    def __init__(self):
        path = _model_path or os.environ.get("BPE_MODEL_PATH")
        if not path:
            raise RuntimeError(
                "BPE model path not set. Either:\n"
                "  • pass --bpe_model <path> to run_all.py, or\n"
                "  • set the BPE_MODEL_PATH environment variable.\n"
                "Train a model first with: python dataset/train_bpe.py"
            )
        try:
            from tokenizers import Tokenizer as HFTokenizer
        except ImportError:
            raise ImportError("pip install tokenizers")
        self._tok = HFTokenizer.from_file(path)

    def tokenize(self, line: str) -> list[str]:
        enc = self._tok.encode(line.strip())
        return enc.tokens + ["<eos>"]
