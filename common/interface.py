from dataclasses import dataclass
from typing import Optional


@dataclass
class GenResult:
    text: str
    latency_s: float
    tokens: int
    memory_mb: float
    confidence: Optional[float]  # mean token probability, 0-1 (None if unavailable)


class Backend:
    name = "base"
     

    def load(self, precisions=("int4", "int8")):
        """Load model weights for the specified precisions into memory."""
        raise NotImplementedError("Each backend must implement its own load method.")

    def generate(self, prompt: str, precision: str, max_new_tokens: int = 128) -> GenResult:
        """Generate response given a prompt and precision target."""
        raise NotImplementedError("Each backend must implement its own generate method.")
