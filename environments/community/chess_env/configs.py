import os
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

from atroposlib.envs.base import BaseEnvConfig


class ValidationDatasetConfig(BaseModel):
    """Configuration settings for the validation/test dataset.

    This class defines the parameters for loading and filtering the subset of
    data used to evaluate model performance during or after training.

    Attributes:
        dataset_name (str): The Hugging Face hub path or local path to the dataset.
            Defaults to "codingmonster1234/chess-puzzles-rlvr".
        split (str): The specific dataset split to load (e.g., 'train', 'test').
            Defaults to "train".
        min_rating (int): The lower bound for puzzle difficulty ratings.
            Defaults to 400.
        max_rating (int): The upper bound for puzzle difficulty ratings.
            Defaults to 3300.
        max_items (int): The total number of items to sample for the validation set.
            Defaults to 1000.
    """

    dataset_name: str = Field(
        default="codingmonster1234/chess-puzzles-rlvr",
        description="The Hugging Face hub path or local path to the dataset.",
    )

    split: str = Field(
        default="train",
        description="The specific dataset split to load.",
    )

    min_rating: int = Field(
        default=400,
        description="The minimum rating for the dataset (minimum 400).",
    )

    max_rating: int = Field(
        default=3300,
        description="The maximum rating for the dataset (max 3300).",
    )

    max_items: int = Field(
        default=1000,
        description="Max number of items to sample for validation.",
    )


class CurriculumStrategy(str, Enum):
    """Enumeration of strategies for sampling items based on performance.

    Attributes:
        UNIFORM (str): Samples all available items with equal probability.
            Value is "uniform".
        EASY_FIRST (str): Prioritizes items with higher success rates.
            Value is "easy_first".
        COMPETENCE_BASED (str): Adjusts sampling weight based on the model's current mastery.
            Value is "competence_based".
    """

    UNIFORM = "uniform"
    EASY_FIRST = "easy_first"
    COMPETENCE_BASED = "competence_based"


class ScheduleConfig(BaseModel):
    """Configuration for the macro-curriculum difficulty scheduler.

    Determines how the difficulty 'window' of the training data expands over
    time based on training steps.

    Attributes:
        min_rating (int): The absolute starting difficulty floor.
            Defaults to 400.
        max_rating (int): The absolute maximum difficulty ceiling.
            Defaults to 3300.
        start_rating (int): The initial difficulty threshold used at step 0.
            Defaults to 500.
        schedule_type (str): Shape of growth (e.g., 'fixed_linear', 'root', 'discrete').
            Defaults to "fixed_linear".
        total_curriculum_step (int): Total steps until the schedule reaches max_rating.
            Defaults to 300.
        difficulty_step (int): The increment size for difficulty increases.
            Defaults to 50.
        root_degree (int): The degree of the root function for non-linear scaling.
            Defaults to 2.
        difficulty_levels (List[int], optional): Specific tiers for discrete scheduling.
            Defaults to None.
        max_steps (List[int], optional): Step thresholds for discrete scheduling.
            Defaults to None.
        enable_infinite_loop (bool): Whether to loop the curriculum after completion.
            Defaults to False.
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
    """Comprehensive configuration for the training curriculum.

    Combines dataset source information with both macro-scheduling (rating progression)
    and micro-scheduling (performance-based binning).

    Attributes:
        dataset_name (str): Path to the training dataset.
            Defaults to "codingmonster1234/chess-puzzles-rlvr".
        split (str): Dataset split for training.
            Defaults to "train".
        max_items (int): Maximum items to sample for training.
            Defaults to 5000.
        curriculum_type (str): Attribute used for difficulty (e.g., 'rating').
            Defaults to "rating".
        schedule_config (ScheduleConfig): Detailed parameters for the macro-scheduler.
            Defaults to a default-initialized ScheduleConfig instance.
        n_bins (int): Number of performance bins for micro-curriculum.
            Defaults to 5.
        temperature (float): Softmax temperature for bin sampling.
            Defaults to 1.0.
        ema_alpha (float): Smoothing factor for item performance EMA updates.
            Defaults to 0.3.
        performance_strategy (CurriculumStrategy): Strategy for sampling from bins.
            Defaults to CurriculumStrategy.COMPETENCE_BASED.
        rebin_interval (int): Frequency of recalculating bin boundaries.
            Defaults to 100.
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
    """High-level configuration for the Chess Reinforcement Learning environment.

    Attributes:
        stockfish_depth (int): Search depth for engine evaluations.
            Defaults to 15.
        engine_time_limit (float): Seconds allocated per engine move.
            Defaults to 0.2.
        reward_scaling_factor (float): Divisor used to scale centipawn losses into rewards.
            Defaults to 200.0.
        stockfish_path (str): Filepath to the Stockfish executable.
            Defaults to the STOCKFISH_PATH environment variable or "none".
        max_concurrent_evals (int): Size of the parallel engine instance pool.
            Defaults to 8.
        invalid_move_notation_reward (float): Penalty/reward for syntactic errors.
            Defaults to 0.01.
        illegal_move_reward (float): Penalty/reward for illegal chess moves.
            Defaults to 0.1.
        format_fail_reward (float): Penalty/reward for incorrect response formatting.
            Defaults to 0.0.
        length_penalty_coefficient (float): Scaling factor for the verbosity penalty.
            Defaults to 0.5.
        training_rollout_temperature (float): Exploration temperature for training.
            Defaults to 1.0.
        eval_rollout_temperature (float): Determinism temperature for evaluation.
            Defaults to 0.0.
        train_dataset_config (CurriculumConfig): Config for training data and curriculum.
            Defaults to a CurriculumConfig with chess-puzzles-rlvr presets.
        validation_dataset_config (ValidationDatasetConfig): Config for validation data.
            Defaults to a ValidationDatasetConfig with validation split presets.
        train_dataset_checkpoint_path (str, optional): Path to resume curriculum state.
            Defaults to None.
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
