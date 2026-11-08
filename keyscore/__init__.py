"""KeyScore: training-free, query-conditioned frame selection."""

from keyscore.scoring import FrameScores, KeyScore, SelectionResult

__version__ = "0.1.0"

__all__ = [
    "KeyScore", "FrameScores", "SelectionResult", "encoders", "proposals", "scoring",
]
