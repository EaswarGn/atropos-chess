import os

from pydantic import BaseModel, Field

from atroposlib.envs.base import BaseEnvConfig


class DatasetConfig(BaseModel):
    """Configuration for managing curriculum learning sdataset."""

    dataset_name: str = Field(
        default="codingmonster1234/chess-puzzles-rlvr",
        description="The Hugging Face hub path or local path to the dataset (e.g., 'username/repo').",
    )
    """The Hugging Face hub path or local path to the dataset (e.g., 'username/repo')."""

    split: str = Field(
        default="train",
        description="The specific dataset split to load (e.g., 'train', 'test', or 'validation').",
    )
    """The specific dataset split to load (e.g., 'train', 'test', or 'validation')."""

    use_curriculum: bool = Field(
        default=True,
        description="Whether to enable progressive difficulty scaling during training.",
    )
    """Whether to enable progressive difficulty scaling during training."""

    bucket_size: int = Field(
        default=100,
        description="The rating range span for each bucket (e.g., 100 for bins like 400-500, 500-600).",
    )
    """The rating range span for each bucket (e.g., 100 for bins like 400-500, 500-600)."""

    min_rating: int = Field(
        default=400,
        description="The floor rating to begin bucketing (smallest is 400).",
    )
    """The floor rating to begin bucketing (smallest is 400)."""

    max_rating: int = Field(
        default=3300, description="The ceiling rating to end bucketing (max 3300)."
    )
    """The ceiling rating to end bucketing (max 3300)."""

    start_bucket: str = Field(
        default="bucket:400-500",
        description="The initial bucket key to start training from (e.g., 'bucket:400-500').",
    )
    """The initial bucket key to start training from (e.g., 'bucket:400-500')."""

    curr_bucket_lookup_prob: float = Field(
        default=0.8,
        description="Probability of returning a puzzle from current bucket for the current item vs a previous bucket.",
    )
    """The probability of returning a puzzle from current bucket for the current item vs a previous bucket."""

    dataset_percent_to_use: float = Field(
        default=1.0,
        description="The percentage of the dataset to utilize (e.g., 0.5 for 50%).",
    )
    """The percentage of the dataset to utilize (e.g., 0.5 for 50%)."""

    use_infinite_looping: bool = Field(
        default=False,
        description="Whether to enable infinite looping over the dataset for continuous training.",
    )
    """Whether to enable infinite looping over the dataset for continuous training."""


class ChessEnvConfig(BaseEnvConfig):
    """Configuration for the Chess puzzle solving environment."""

    # --- Engine Settings ---
    stockfish_depth: int = Field(
        15, description="Search depth for Stockfish evaluation"
    )
    engine_time_limit: float = Field(
        0.2, description="Seconds allowed per move evaluation"
    )
    reward_scaling_factor: float = Field(
        200.0, description="Centipawn divisor for sigmoid reward"
    )
    stockfish_path: str = Field(
        os.getenv("STOCKFISH_PATH", "none"),
        description="Path to executable stockfish engine. Set as an environment variable as: STOCKFISH_PATH",
    )
    max_concurrent_evals: int = Field(
        8, description="Total number of engine instances initialized"
    )

    # --- Penalties ---
    invalid_move_notation_reward: float = Field(
        0.01, description="Reward assigned to moves with invalid notation"
    )
    illegal_move_reward: float = Field(
        0.1, description="Reward assigned to moves that are illegal"
    )
    format_fail_reward: float = Field(
        0.0, description="Reward assigned to invalid response formats"
    )
    length_penalty_coefficient: float = Field(
        0.5, description="Strength of the penalty for excessive response length"
    )

    # --- Rollouts ---
    training_rollout_temperature: float = Field(
        1.0,
        description="High temperature for training rollouts to encourage exploration",
    )
    eval_rollout_temperature: float = Field(
        0.0, description="Low temperature for eval rollouts for deterministic answers"
    )

    # --- Datasets ---
    train_dataset_config: DatasetConfig = Field(
        default_factory=lambda: DatasetConfig(
            dataset_name="codingmonster1234/chess-puzzles-rlvr",
            split="train",
            use_curriculum=True,
            bucket_size=100,
            min_rating=400,
            max_rating=3300,
            start_bucket="bucket:400-500",
            curr_bucket_lookup_prob=0.8,
            dataset_percent_to_use=1.0,
            use_infinite_looping=False,
        ),
        description="Configuration for the training dataset",
    )

    validation_dataset_config: DatasetConfig = Field(
        default_factory=lambda: DatasetConfig(
            dataset_name="codingmonster1234/chess-puzzles-rlvr",
            split="validation",
            use_curriculum=False,
            dataset_percent_to_use=1.0,
            use_infinite_looping=False,
        ),
        description="Configuration for the validation dataset",
    )
