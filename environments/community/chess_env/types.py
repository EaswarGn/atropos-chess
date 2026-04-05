from typing import Any, List, Tuple

from pydantic import BaseModel, Field


class ChessPuzzleItem(BaseModel):
    prompt: Tuple[Any, ...] = Field(
        description="The conversation history as a tuple of frozensets."
    )
    """The conversation history as a tuple of frozensets."""

    best_move: str = Field(description="The UCI string of the winning move.")
    """The UCI string of the winning move."""

    rating: int = Field(description="The Elo rating of the puzzle.")
    """The Elo rating of the puzzle."""

    fen: str = Field(description="The FEN position before the best move.")
    """The FEN position before the best move."""

    tags: List[str] = Field(
        default_factory=list, description="List of tactical themes (e.g., 'fork')."
    )
    """List of tactical themes (e.g., 'fork')."""
