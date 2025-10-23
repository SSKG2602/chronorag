"""Signal container passed between retrieval and the DHQC controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class RetrievalSignals:
    """Lightweight metrics describing retrieval health."""

    coverage: float = 0.0
    contradiction: float = 0.0
    low_confidence: float = 0.0
    authority: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Return a dict representation to simplify logging/serialization."""
        return {
            "coverage": self.coverage,
            "contradiction": self.contradiction,
            "low_confidence": self.low_confidence,
            "authority": self.authority,
        }
