import bisect
import logging
import math
import random
from typing import Any, Dict, List, Optional

import chess
from datasets import load_dataset

from .chess_env_types import ChessPuzzleItem
from .configs import CurriculumConfig, CurriculumStrategy, ScheduleConfig
from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


def format_item(row: Dict[str, Any]) -> ChessPuzzleItem:
    """Helper to convert raw dict to ```ChessPuzzleItem```"""

    board = chess.Board(row["fen"])
    user_prompt = USER_PROMPT_TEMPLATE.format(
        fen_string=row["fen"],
        ascii_board=str(board),
        legal_moves_list=", ".join(board.san(move) for move in board.legal_moves),
        turn=row["turn"],
    )
    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    return ChessPuzzleItem(
        uuid=row["uuid"],
        prompt=prompt,
        best_move=row["uci_moves"][0],
        rating=row["rating"],
        fen=row["fen"],
        tags=row["tags"],
    )


class CurriculumScheduler:
    def __init__(self, schedule_config: ScheduleConfig):
        self.sched_cfg = schedule_config

        # Validate shared rating bounds for all schedule types
        assert (
            self.sched_cfg.min_rating <= self.sched_cfg.max_rating
        ), f"min_rating ({self.sched_cfg.min_rating}) must be <= max_rating ({self.sched_cfg.max_rating})"
        assert (
            self.sched_cfg.start_rating >= self.sched_cfg.min_rating
        ), f"start_rating ({self.sched_cfg.start_rating}) must be >= min_rating ({self.sched_cfg.min_rating})"
        assert (
            self.sched_cfg.start_rating <= self.sched_cfg.max_rating
        ), f"start_rating ({self.sched_cfg.start_rating}) must be <= max_rating ({self.sched_cfg.max_rating})"

        if self.sched_cfg.schedule_type in ("fixed_linear", "fixed_root"):
            schedule_type = self.sched_cfg.schedule_type
            assert (
                self.sched_cfg.total_curriculum_step is not None
            ), f"{schedule_type} requires schedule_config.total_curriculum_step"
            assert (
                self.sched_cfg.total_curriculum_step > 0
            ), f"total_curriculum_step must be > 0, got {self.sched_cfg.total_curriculum_step}"
            if schedule_type == "fixed_root":
                assert (
                    self.sched_cfg.root_degree is not None
                ), "fixed_root requires schedule_config.root_degree"
                assert (
                    self.sched_cfg.root_degree > 0
                ), f"root_degree must be > 0, got {self.sched_cfg.root_degree}"

        elif self.sched_cfg.schedule_type == "fixed_discrete":
            assert (
                self.sched_cfg.difficulty_levels is not None
            ), "fixed_discrete requires schedule_config.difficulty_levels"
            assert (
                self.sched_cfg.max_steps is not None
            ), "fixed_discrete requires schedule_config.max_steps"
            assert len(self.sched_cfg.difficulty_levels) == len(
                self.sched_cfg.max_steps
            ), (
                f"difficulty_levels (len={len(self.sched_cfg.difficulty_levels)}) and "
                f"max_steps (len={len(self.sched_cfg.max_steps)}) must have the same length"
            )
            assert all(
                self.sched_cfg.min_rating <= d <= self.sched_cfg.max_rating
                for d in self.sched_cfg.difficulty_levels
            ), (
                f"All difficulty_levels must be within [{self.sched_cfg.min_rating}, {self.sched_cfg.max_rating}]. "
                f"Got: {self.sched_cfg.difficulty_levels}"
            )
            assert self.sched_cfg.max_steps == sorted(
                self.sched_cfg.max_steps
            ), f"max_steps must be in ascending order. Got: {self.sched_cfg.max_steps}"
            assert self.sched_cfg.difficulty_levels == sorted(
                self.sched_cfg.difficulty_levels
            ), "difficulty_levels must be in ascending order for fixed_discrete"
        else:
            raise ValueError(
                f"Unknown schedule_type: '{self.sched_cfg.schedule_type}'. "
                "Must be one of: 'fixed_linear', 'fixed_root', 'fixed_discrete'."
            )

        logger.info(
            f"Initialized CurriculumScheduler with type: {self.sched_cfg.schedule_type}"
        )

    def get_difficulty(self, current_step: int) -> int:
        if self.sched_cfg.schedule_type == "fixed_linear":
            return self._fixed_linear(current_step)
        elif self.sched_cfg.schedule_type == "fixed_root":
            return self._fixed_root(current_step)
        else:  # fixed_discrete — validated exhaustively in __init__
            return self._fixed_discrete(current_step)

    def _apply_difficulty_step(self, raw_difficulty: float) -> int:
        step = self.sched_cfg.difficulty_step
        if step is None or step <= 1:
            return int(raw_difficulty)
        return int(math.floor(raw_difficulty / step) * step)

    def _fixed_linear(self, current_step: int) -> int:
        start = self.sched_cfg.start_rating
        end = self.sched_cfg.max_rating
        progress = min(1.0, current_step / max(1, self.sched_cfg.total_curriculum_step))
        raw_diff = start + progress * (end - start)
        return min(end, self._apply_difficulty_step(raw_diff))

    def _fixed_root(self, current_step: int) -> int:
        start = self.sched_cfg.start_rating
        end = self.sched_cfg.max_rating
        progress = min(1.0, current_step / max(1, self.sched_cfg.total_curriculum_step))
        root_progress = progress ** (1.0 / self.sched_cfg.root_degree)
        raw_diff = start + root_progress * (end - start)
        return min(self.sched_cfg.max_rating, self._apply_difficulty_step(raw_diff))

    def _fixed_discrete(self, current_step: int) -> int:
        for diff, max_step in zip(
            self.sched_cfg.difficulty_levels, self.sched_cfg.max_steps
        ):
            if current_step <= max_step:
                return diff
        return self.sched_cfg.difficulty_levels[-1]


class StatefulCurriculumManager:
    def __init__(
        self,
        curriculum_config: CurriculumConfig,
    ):
        self.config = curriculum_config

        logger.info(
            f"Loading dataset {self.config.dataset_name} (split: {self.config.split})..."
        )
        hf_dataset = load_dataset(self.config.dataset_name, split=self.config.split)

        # filter by rating and take specified number of samples

        hf_dataset = hf_dataset.filter(
            lambda x: self.config.schedule_config.min_rating
            <= x["rating"]
            <= self.config.schedule_config.max_rating
        )
        if self.config.max_items is not None:
            if self.config.max_items < 1:
                raise ValueError(
                    f"max_items must be at least 1, got {self.config.max_items}"
                )
            if self.config.max_items > len(hf_dataset):
                raise ValueError(
                    f"max_items ({self.config.max_items}) exceeds available ({len(hf_dataset)})"
                )
            all_indices = list(range(len(hf_dataset)))
            selected_indices = random.sample(all_indices, self.config.max_items)
            hf_dataset = hf_dataset.select(selected_indices)

        logger.info("Sorting dataset by rating...")
        self.dataset = hf_dataset.sort("rating")
        self._ratings_list = list(self.dataset["rating"])
        self._uuids = list(self.dataset["uuid"])

        self.scheduler = CurriculumScheduler(self.config.schedule_config)

        self._current_step = 0
        self._max_rating = self.scheduler.get_difficulty(self._current_step)

        self._item_scores = {}  # unique uuid -> (ema_score, update_count)
        self._bin_boundaries = []

        self._total_updates = 0
        self._last_rebin_count = 0
        self.n = 0

        logger.info(
            f"Dataset initialized: {len(self.dataset)} items. Starting Max Rating: {self._max_rating}"
        )

    # ==========================================
    # STATE MANAGEMENT & LOGGING
    # ==========================================

    def state_dict(self) -> Dict[str, Any]:
        logger.info("Capturing dataset state_dict...")
        return {
            "current_step": self._current_step,
            "max_rating": self._max_rating,
            "item_scores": dict(self._item_scores),
            "bin_boundaries": list(self._bin_boundaries),
            "total_updates": self._total_updates,
            "last_rebin_count": self._last_rebin_count,
            "n": self.n,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        logger.info("Loading dataset state_dict...")
        self._current_step = state.get("current_step", 0)
        self._max_rating = state.get(
            "max_rating", self.scheduler.get_difficulty(self._current_step)
        )

        self._item_scores.clear()
        self._item_scores.update(state.get("item_scores", {}))

        self._bin_boundaries = list(state.get("bin_boundaries", []))

        self._total_updates = state.get("total_updates", 0)
        self._last_rebin_count = state.get("last_rebin_count", 0)
        self.n = state.get("n", 0)
        logger.info(
            f"Resumed at step {self._current_step} with {len(self._item_scores)} scored items."
        )

    def get_metrics(self) -> Dict[str, Any]:
        current_step = self._current_step
        max_rating = self._max_rating
        total_updates = self._total_updates
        boundaries = list(self._bin_boundaries)
        scored_items = len(self._item_scores)
        unlocked_items = bisect.bisect_right(self._ratings_list, self._max_rating)

        total_items = len(self.dataset)
        unlocked_ratio = unlocked_items / total_items if total_items > 0 else 0.0

        metrics = {
            "curriculum/current_step": current_step,
            "curriculum/max_rating": max_rating,
            "curriculum/unlocked_items": unlocked_items,
            "curriculum/unlocked_ratio": unlocked_ratio,
            "curriculum/scored_items": scored_items,
            "curriculum/total_updates": total_updates,
        }

        for i, bound in enumerate(boundaries):
            metrics[f"curriculum/bin_boundary_{i}"] = bound

        return metrics

    # ==========================================
    # MACRO-CURRICULUM (RATING)
    # ==========================================

    def increment_step(self, steps: int = 1) -> None:
        old_rating = self._max_rating
        self._current_step += steps
        self._current_step = min(self._current_step, self._resolve_total_steps())
        self._max_rating = self.scheduler.get_difficulty(self._current_step)

        if self._max_rating > old_rating:
            logger.info(
                f"Curriculum Level Up! Rating: {old_rating} -> {self._max_rating} (Step: {self._current_step})"
            )

    @property
    def valid_indices_count(self) -> int:
        """
        Returns how many puzzles are currently 'unlocked'.
        Uses the pre-calculated ratings list for O(log N) performance.
        """
        rating_cap = self._max_rating
        return bisect.bisect_right(self._ratings_list, rating_cap)

    @property
    def step(self) -> int:
        """
        Returns current step.
        """
        return self._current_step

    def get_next_item(self, tournament_size: int = 50) -> ChessPuzzleItem:
        """
        Fused Sampling via O(1) Tournament Selection:
        1. Macro-filter: Get the valid count of unlocked items.
        2. Micro-strategy: Pick a target performance bin.
        3. Selection: Sample a fixed K batch of indices, evaluate their bins,
           and return the item that closest matches the target bin.
        """
        count = self.valid_indices_count

        # Decide which performance bin to target (Micro)
        target_bin = self.sample_bin()

        # Tournament Selection: Sample K random indices from the unlocked pool
        actual_tournament_size = min(tournament_size, count)
        candidate_indices = random.sample(range(count), actual_tournament_size)

        best_idx = None
        best_distance = float("inf")

        for idx in candidate_indices:
            uuid = self._uuids[idx]
            item_bin = self.get_item_bin(uuid)

            distance = abs(item_bin - target_bin)

            if distance == 0:
                best_idx = idx
                break  # Exact match found, early exit to save compute
            elif distance < best_distance:
                best_distance = distance
                best_idx = idx

        if best_idx is None:
            logger.debug("Tournament selection failed, picking random unlocked index.")
            best_idx = random.choice(range(count))

        self.n += 1

        chosen_item = self.dataset[best_idx]
        return format_item(chosen_item)

    # ==========================================
    # MICRO-CURRICULUM (PERFORMANCE BINNING)
    # ==========================================

    def update(self, item_key: str, score: float) -> None:
        if item_key in self._item_scores:
            old_ema, count = self._item_scores[item_key]
            new_ema = (
                self.config.ema_alpha * score + (1 - self.config.ema_alpha) * old_ema
            )
            self._item_scores[item_key] = (new_ema, count + 1)
        else:
            self._item_scores[item_key] = (float(score), 1)

        self._total_updates += 1
        current_updates = self._total_updates

        if current_updates - self._last_rebin_count >= self.config.rebin_interval:
            logger.info(f"Re-binning micro-curriculum at {current_updates} updates...")
            self._compute_and_set_bins()
            self._last_rebin_count = current_updates

    def update_group(self, item_key: str, scores: List[float]) -> None:
        if not scores:
            return
        avg_score = sum(scores) / len(scores)
        self.update(item_key, avg_score)

    def get_item_difficulty(self, item_key: str) -> Optional[float]:
        if item_key not in self._item_scores:
            return None
        return self._item_scores[item_key][0]

    def get_item_bin(self, item_key: str) -> int:
        difficulty = self.get_item_difficulty(item_key)
        if difficulty is None:
            return self.config.n_bins // 2

        boundaries = list(self._bin_boundaries)
        if not boundaries:
            self._compute_and_set_bins()
            boundaries = list(self._bin_boundaries)

        for i, boundary in enumerate(boundaries):
            if difficulty >= boundary:
                return i
        return self.config.n_bins // 2

    def _resolve_total_steps(self) -> int:
        """
        Return the effective total curriculum steps for bin probability computation.

        Each schedule type has a different canonical source for this value:
        - fixed_linear / fixed_root: total_curriculum_step is the direct, validated answer.
        - fixed_discrete: total_curriculum_step is not required by the macro schedule, so
          we derive it from max_steps[-1] (the last stage boundary) instead.
        - Curriculum disabled: fall back to total_curriculum_step if set, otherwise
          treat the current position as fully progressed (progress clamps to 1.0).
        """
        sched_cfg = self.config.schedule_config

        if sched_cfg.schedule_type in ("fixed_linear", "fixed_root"):
            # Guaranteed non-None by __init__ validation.
            return sched_cfg.total_curriculum_step

        if sched_cfg.schedule_type == "fixed_discrete":
            # max_steps[-1] is the last stage boundary — the natural total span.
            # Guaranteed to exist by __init__ validation.
            return sched_cfg.max_steps[-1]

    def sample_bin(self) -> int:
        if self.config.performance_strategy == CurriculumStrategy.UNIFORM:
            return random.randint(0, self.config.n_bins - 1)

        total_steps = self._resolve_total_steps()
        probs = self._compute_bin_probabilities(self._current_step, total_steps)

        if self.config.temperature != 1.0:
            log_probs = [
                math.log(max(p, 1e-10)) / self.config.temperature for p in probs
            ]
            max_lp = max(log_probs)
            exp_probs = [math.exp(lp - max_lp) for lp in log_probs]
            total = sum(exp_probs)
            probs = [p / total for p in exp_probs]

        return random.choices(range(self.config.n_bins), weights=probs, k=1)[0]

    def _compute_bin_probabilities(
        self, current_step: int, total_steps: int
    ) -> List[float]:
        progress = min(1.0, max(0.0, current_step / max(1, total_steps)))

        if self.config.performance_strategy == CurriculumStrategy.EASY_FIRST:
            return self._easy_first_probs(progress)
        elif self.config.performance_strategy == CurriculumStrategy.COMPETENCE_BASED:
            return self._competence_based_probs(progress)
        else:
            return [1.0 / self.config.n_bins] * self.config.n_bins

    def _easy_first_probs(self, progress: float) -> List[float]:
        probs = []
        for i in range(self.config.n_bins):
            uniform_prob = 1.0 / self.config.n_bins
            easy_bias = math.exp(-2.0 * i / max(1, self.config.n_bins - 1))
            prob = (1.0 - progress) * easy_bias + progress * uniform_prob
            probs.append(prob)
        total = sum(probs)
        return [p / total for p in probs]

    def _competence_based_probs(self, progress: float) -> List[float]:
        """
        Samples from the current learning frontier using a Gaussian centered on
        the frontier bin, which shifts from mid-difficulty toward harder bins as
        training progresses.

        Bin ordering (set by _compute_and_set_bins):
            bin 0   = highest EMA score = model already solves these (easy)
            bin n-1 = lowest EMA score  = model consistently fails these (hard)

        Frontier logic:
            - At progress=0.0: frontier is at the middle bin (neither mastered nor
            impossible — the true learning edge at the start of training).
            - At progress=1.0: frontier has shifted to bin n-1 (pushing into the
            hardest unsolved material at the end of training).

        The Gaussian sigma widens slightly as progress increases, preventing the
        sampler from over-focusing on a single bin when pushing into hard territory.
        """
        n = self.config.n_bins

        # Start at the middle bin (true frontier), shift toward hard bins (n-1) over time.
        # At progress=0.0: frontier = (n-1)/2  (middle)
        # At progress=1.0: frontier = (n-1)    (hardest bin)
        mid = (n - 1) / 2.0
        frontier_bin = mid + progress * mid  # shifts from mid → n-1

        # Sigma widens as we push into harder, less-explored territory.
        # Keeps sampling from a broader range when the model is uncertain.
        base_sigma = max(1.0, n / 4.0)
        sigma = base_sigma * (1.0 + 0.5 * progress)

        probs = []
        for i in range(n):
            distance = i - frontier_bin  # signed: positive = harder than frontier
            prob = math.exp(-0.5 * (distance / sigma) ** 2)
            probs.append(prob)

        total = sum(probs)
        return [p / total for p in probs]

    def _compute_and_set_bins(self) -> None:
        if not self._item_scores:
            self._bin_boundaries.clear()
            return

        scores = sorted([ema for ema, _ in self._item_scores.values()], reverse=True)

        if len(scores) < self.config.n_bins:
            logger.debug(
                "Too few scores to compute proper quantiles; using linear range."
            )
            min_s, max_s = min(scores), max(scores)
            if max_s == min_s:
                self._bin_boundaries = []  # No information — signal unscorable
            else:
                step = (max_s - min_s) / self.config.n_bins
                self._bin_boundaries = [
                    max_s - i * step for i in range(self.config.n_bins)
                ]
            return

        boundaries = []
        for i in range(self.config.n_bins):
            idx = int((i + 1) * len(scores) / self.config.n_bins) - 1
            idx = min(max(idx, 0), len(scores) - 1)
            boundaries.append(scores[idx])

        self._bin_boundaries = boundaries
        logger.info(
            f"New Bin Boundaries: {[round(b, 3) for b in self._bin_boundaries]}"
        )

    def reset(self) -> None:
        """
        Resets the macro-curriculum schedule back to step 0 for a new epoch,
        while preserving all accumulated learning state.

        Resets:
            _current_step   — schedule position, back to 0
            _max_rating     — recomputed from scheduler at step 0
            n               — per-epoch draw counter, back to 0

        Preserved (carry over into the next epoch):
            _item_scores        — all EMA scores; binning quality improves with epochs
            _bin_boundaries     — derived from scores; stays valid across resets
            _total_updates      — running total for metrics and rebin interval tracking
            _last_rebin_count   — prevents a spurious immediate rebin on resume
        """
        prev_step = self._current_step
        prev_rating = self._max_rating

        self._current_step = 0
        self._max_rating = self.scheduler.get_difficulty(0)
        self.n = 0

        logger.info(
            f"Curriculum reset: step {prev_step} -> 0, "
            f"max_rating {prev_rating} -> {self._max_rating} "
            f"({len(self._item_scores)} scores retained)"
        )

    # ==========================================
    # DUNDER METHODS
    # ==========================================

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> ChessPuzzleItem:
        return format_item(self.dataset[idx])

    def __iter__(self):
        return self

    def __next__(self) -> ChessPuzzleItem:
        return self.get_next_item()

    def __str__(self) -> str:
        total = len(self.dataset)
        unlocked = self.valid_indices_count
        percent = (unlocked / total) * 100 if total > 0 else 0

        current_step = self._current_step
        max_rating = self._max_rating
        bounds = list(self._bin_boundaries)

        bounds_str = f"[{', '.join(f'{b:.2f}' for b in bounds)}]" if bounds else "[]"

        return (
            f"StatefulCurriculumDataset | "
            f"Step: {current_step} | "
            f"Unlocked: {unlocked}/{total} ({percent:.1f}%) | "
            f"Max Elo: {max_rating} | "
            f"Bins: {bounds_str}"
        )

    def __repr__(self) -> str:
        current_step = self._current_step
        max_rating = self._max_rating
        evaluated_items = len(self._item_scores)

        return (
            f"<{self.__class__.__name__}(total_items={len(self.dataset)}, "
            f"current_step={current_step}, "
            f"max_rating={max_rating}, "
            f"valid_items={self.valid_indices_count}, "
            f"evaluated_items={evaluated_items})>"
        )
