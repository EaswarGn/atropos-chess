import os
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

from atroposlib.envs.base import BaseEnvConfig


class ValidationDatasetConfig(BaseModel):
    """
    Configuration settings for the validation/test dataset.

    This class defines the parameters for loading and filtering the subset of
    data used to evaluate model performance during or after training.

    Attributes:
        dataset_name (str): The Hugging Face hub path or local path to the dataset.
        split (str): The specific dataset split to load (e.g., 'train', 'test').
        min_rating (int): The lower bound for puzzle difficulty ratings (no lower than 400).
        max_rating (int): The upper bound for puzzle difficulty ratings (no higher than 3300).
        max_items (int): The total number of items to sample for the validation set.
    """

    dataset_name: str = Field(
        default="codingmonster1234/chess-puzzles-rlvr",
        description="The Hugging Face hub path or local path to the dataset (e.g., 'username/repo').",
    )

    split: str = Field(
        default="train",
        description="The specific dataset split to load (e.g., 'train', 'test', or 'validation').",
    )

    min_rating: int = Field(
        default=400,
        description="The minimum rating for the dataset (minimum 400).",
    )

    max_rating: int = Field(
        default=3300, description="The maximum rating for the dataset (max 3300)."
    )

    max_items: int = Field(
        default=1000,
        description="Max number of items to sample for validation.",
    )


class CurriculumStrategy(str, Enum):
    """
    Enumeration of strategies for sampling items based on performance.

    Attributes:
        UNIFORM: Samples all available items with equal probability.
        EASY_FIRST: Prioritizes items with higher success rates.
        COMPETENCE_BASED: Adjusts sampling weight based on the model's current mastery.
    """

    UNIFORM = "uniform"
    EASY_FIRST = "easy_first"
    COMPETENCE_BASED = "competence_based"


class ScheduleConfig(BaseModel):
    """
    Configuration for the macro-curriculum difficulty scheduler.

    Determines how the difficulty 'window' of the training data expands over
    time based on training steps.

    Attributes:
        min_rating (int): The absolute starting difficulty floor.
        max_rating (int): The absolute maximum difficulty ceiling.
        start_rating (int): The initial difficulty threshold used at step 0.
        schedule_type (str): Shape of growth (e.g., 'fixed_linear', 'root', 'discrete').
        total_curriculum_step (int): Total steps until the schedule reaches max_rating.
        difficulty_step (int): The increment size for difficulty increases (e.g., Elo steps).
        root_degree (int): The degree of the root function for non-linear scaling.
        difficulty_levels (Optional[List[int]]): Specific tiers for discrete scheduling.
        max_steps (Optional[List[int]]): Step thresholds for discrete scheduling.
        enable_infinite_loop (bool): Whether to loop the curriculum after completion.
    """

    min_rating: int = Field(default=400, description="The starting difficulty floor.")

    max_rating: int = Field(default=3300, description="The maximum difficulty ceiling.")

    start_rating: int = Field(
        default=500, description="The initial difficulty threshold used at step 0."
    )

    schedule_type: str = Field(
        default="fixed_linear", description="The shape of the macro-curriculum growth."
    )

    total_curriculum_step: int = Field(
        default=300, description="Total number of steps for the curriculum schedule."
    )

    difficulty_step: int = Field(
        default=50, description="The increment size for difficulty increases."
    )

    root_degree: int = Field(
        default=2, description="The degree of the root function for non-linear scaling."
    )

    difficulty_levels: Optional[List[int]] = Field(
        default=None, description="Specific difficulty tiers for discrete scheduling."
    )

    max_steps: Optional[List[int]] = Field(
        default=None, description="Step thresholds corresponding to difficulty_levels."
    )

    enable_infinite_loop: bool = Field(
        default=False,
        description="Whether to loop the curriculum after reaching max difficulty.",
    )


class CurriculumConfig(BaseModel):
    """
    Comprehensive configuration for the training curriculum.

    Combines dataset source information with both macro-scheduling (rating progression)
    and micro-scheduling (performance-based binning).

    Attributes:
        dataset_name (str): Path to the training dataset.
        split (str): Dataset split for training.
        max_items (int): Maximum items to sample for training.
        curriculum_type (str): Attribute used for difficulty (e.g., 'rating').
        schedule_config (ScheduleConfig): Detailed parameters for the macro-scheduler.
        n_bins (int): Number of performance bins for micro-curriculum.
        temperature (float): Softmax temperature for bin sampling.
        ema_alpha (float): Smoothing factor for item performance EMA updates.
        performance_strategy (CurriculumStrategy): Strategy for sampling from bins.
        rebin_interval (int): Frequency of recalculating bin boundaries.
    """

    dataset_name: str = Field(
        default="codingmonster1234/chess-puzzles-rlvr",
        description="The Hugging Face hub path or local path to the dataset.",
    )

    split: str = Field(
        default="train",
        description="The specific dataset split to load.",
    )

    max_items: int = Field(
        default=5000,
        description="Max number of items to sample for training.",
    )

    curriculum_type: str = Field(
        default="rating", description="The attribute used for difficulty."
    )

    schedule_config: ScheduleConfig = Field(
        default_factory=ScheduleConfig,
        description="Detailed parameters for the macro-scheduler.",
    )

    n_bins: int = Field(
        default=5, description="Number of performance bins for micro-curriculum."
    )

    temperature: float = Field(
        default=1.0, description="Softmax temperature for bin sampling."
    )

    ema_alpha: float = Field(
        default=0.3, description="Smoothing factor for item performance EMA updates."
    )

    performance_strategy: CurriculumStrategy = Field(
        default=CurriculumStrategy.COMPETENCE_BASED,
        description="Strategy for sampling from performance bins.",
    )

    rebin_interval: int = Field(
        default=100,
        description="How many updates to wait before recalculating bin boundaries.",
    )


class ChessEnvConfig(BaseEnvConfig):
    """
    High-level configuration for the Chess Reinforcement Learning environment.

    This class manages engine parameters, reward shaping, penalty coefficients,
    and the overall training/validation data setup.

    Attributes:
        stockfish_depth (int): Search depth for engine evaluations.
        engine_time_limit (float): Seconds allocated per engine move.
        reward_scaling_factor (float): Divisor used to scale centipawn losses into rewards.
        stockfish_path (str): Filepath to the Stockfish executable.
        max_concurrent_evals (int): Size of the parallel engine instance pool.
        invalid_move_notation_reward (float): Penalty/reward for syntactic errors.
        illegal_move_reward (float): Penalty/reward for illegal chess moves.
        format_fail_reward (float): Penalty/reward for incorrect response formatting.
        length_penalty_coefficient (float): Scaling factor for the verbosity penalty.
        training_rollout_temperature (float): Exploration temperature for training.
        eval_rollout_temperature (float): Determinism temperature for evaluation.
        train_dataset_config (CurriculumConfig): Config for training data and curriculum.
        validation_dataset_config (ValidationDatasetConfig): Config for validation data.
        train_dataset_checkpoint_path (Optional[str]): Path to resume curriculum state.
    """

    stockfish_depth: int = Field(
        default=15, description="Search depth for Stockfish evaluation"
    )

    engine_time_limit: float = Field(
        default=0.2, description="Seconds allowed per move evaluation"
    )

    reward_scaling_factor: float = Field(
        default=200.0, description="Centipawn divisor for sigmoid reward"
    )

    stockfish_path: str = Field(
        default=os.getenv("STOCKFISH_PATH", "none"),
        description="Path to executable stockfish engine.",
    )

    max_concurrent_evals: int = Field(
        default=8, description="Total number of engine instances initialized"
    )

    invalid_move_notation_reward: float = Field(
        default=0.01, description="Reward assigned to moves with invalid notation"
    )

    illegal_move_reward: float = Field(
        default=0.1, description="Reward assigned to moves that are illegal"
    )

    format_fail_reward: float = Field(
        default=0.0, description="Reward assigned to invalid response formats"
    )

    length_penalty_coefficient: float = Field(
        default=0.5, description="Strength of the penalty for excessive response length"
    )

    training_rollout_temperature: float = Field(
        default=1.0,
        description="High temperature for training rollouts to encourage exploration",
    )

    eval_rollout_temperature: float = Field(
        default=0.0,
        description="Low temperature for eval rollouts for deterministic answers",
    )

    train_dataset_config: CurriculumConfig = Field(
        default_factory=lambda: CurriculumConfig(
            dataset_name="codingmonster1234/chess-puzzles-rlvr",
            split="train",
            max_items=5000,
            curriculum_type="rating",
            schedule_config=ScheduleConfig(
                min_rating=400,
                max_rating=3300,
                start_rating=500,
                schedule_type="fixed_linear",
                total_curriculum_step=5000,
                difficulty_step=50,
                root_degree=2,
            ),
            n_bins=5,
            temperature=1,
            ema_alpha=0.3,
            performance_strategy=CurriculumStrategy.COMPETENCE_BASED,
            rebin_interval=100,
        ),
        description="Configuration for the training dataset",
    )

    validation_dataset_config: ValidationDatasetConfig = Field(
        default_factory=lambda: ValidationDatasetConfig(
            dataset_name="codingmonster1234/chess-puzzles-rlvr",
            split="validation",
            min_rating=400,
            max_rating=3300,
            max_items=1000,
        ),
        description="Configuration for the validation dataset",
    )

    train_dataset_checkpoint_path: Optional[str] = Field(
        default=None, description="Path to training dataset state dict"
    )
