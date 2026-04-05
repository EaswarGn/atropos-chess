import bisect
import json
import random
from collections.abc import Iterator
from typing import Any, Dict, List

import chess
from datasets import load_dataset
from rich.console import Console
from rich.syntax import Syntax

from .chess_env_types import ChessPuzzleItem
from .configs import DatasetConfig
from .prompts import (
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
)


class CurriculumManager:
    def __init__(self, cfg: DatasetConfig):
        self.cfg = cfg

        self.dataset = load_dataset(self.cfg.dataset_name, split=self.cfg.split)

        self.len = 0  # length of the dataset (after applying percentage filter)

        if self.cfg.split == "validation" or self.cfg.split == "test":
            full_range = list(range(len(self.dataset)))
            num_samples = int(len(full_range) * self.cfg.dataset_percent_to_use)
            self.indices = random.sample(full_range, num_samples)
            self.len = len(self.indices)

        if self.cfg.split == "train":
            if self.cfg.use_curriculum is True:
                self.buckets: Dict[str, List[int]] = {}
                self.create_buckets()

                self.curr_bucket_key = self.cfg.start_bucket
            else:
                full_range = list(range(len(self.dataset)))
                num_samples = int(len(full_range) * self.cfg.dataset_percent_to_use)
                self.indices = random.sample(full_range, num_samples)
                self.len = len(self.indices)
        self.n = 0  # counter for how many puzzles have been served so far
        self.curr_idx = 0  # pointer to the current position in the dataset

    "______Dunder Methods______"

    def __len__(self) -> int:
        return self.len

    def __repr__(self) -> str:
        data = {
            "type": self.__class__.__name__,
            "config": {
                "dataset": self.cfg.dataset_name,
                "split": self.cfg.split,
                "use_curriculum": self.cfg.use_curriculum,
                "bucket_size": self.cfg.bucket_size,
                "curr_bucket_prob": self.cfg.curr_bucket_lookup_prob,
            },
            "status": {
                "current_bucket": (
                    self.curr_bucket_key if self.cfg.use_curriculum else "N/A"
                ),
                "progress": (
                    f"{(self.n / self.__len__() * 100):.2f}%"
                    if self.__len__() > 0
                    else "0%"
                ),
                "puzzles_served": self.n,
                "remaining": self.__len__() - self.n,
                "current_index": self.curr_idx,
            },
            "bucket_metrics": {
                "curr_bucket_remaining": (
                    len(self.buckets.get(self.curr_bucket_key, []))
                    if self.cfg.use_curriculum
                    else "N/A"
                ),
                "history_buckets_available": (
                    len(self.get_all_previous_buckets(self.curr_bucket_key))
                    if self.cfg.use_curriculum
                    else 0
                ),
            },
        }

        json_str = json.dumps(data, indent=4)
        console = Console(width=80, force_terminal=True)

        with console.capture() as capture:
            console.print(
                Syntax(json_str, "json", theme="monokai", background_color="default")
            )

        return capture.get()

    def __str__(self):
        return self.__repr__()

    def __iter__(self) -> Iterator:
        """Required by the iterator protocol. Returns the iterator object itself."""
        return self

    def __next__(self) -> Dict[str, Any]:
        """
        Required by the iterator protocol.
        Returns the next puzzle or raises StopIteration.
        """

        item = self.get_next_item()

        return item

    def __getitem__(self, idx: int):
        return self.dataset[idx]

    "______End of Dunder Methods______"

    def state_dict(self) -> Dict[str, Any]:
        """Returns the current state of the manager for checkpointing."""
        return {
            "config": self.cfg.model_dump(),
            "n": self.n,
            "curr_bucket_key": getattr(self, "curr_bucket_key", None),
            "buckets": self.buckets,  # The current lists of remaining indices in each bucket
            "indices": getattr(self, "indices", None),  # For non-curriculum mode
            "random_state": random.getstate(),
            "curr_idx": self.curr_idx,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Restores the state from a saved dictionary."""

        self.cfg = DatasetConfig(**state_dict["config"])

        self.n = state_dict["n"]
        self.buckets = state_dict["buckets"]
        self.indices = state_dict["indices"]
        self.curr_idx = state_dict.get("curr_idx", 0)
        if state_dict["curr_bucket_key"]:
            self.curr_bucket_key = state_dict["curr_bucket_key"]

        if "random_state" in state_dict:
            state_list = state_dict["random_state"]
            state_tuple = (state_list[0], tuple(state_list[1]), state_list[2])
            random.setstate(state_tuple)

    def create_buckets(self) -> None:
        """Sorts the dataset and subsets each bucket based on dataset_percent_to_use."""
        self.dataset = self.dataset.sort("rating")
        ratings: List[int] = self.dataset["rating"]

        step: int = self.cfg.bucket_size
        min_r: int = self.cfg.min_rating
        max_r: int = self.cfg.max_rating
        percent: float = self.cfg.dataset_percent_to_use  # e.g., 0.1 for 10%

        for start in range(min_r, max_r, step):
            end: int = start + step
            bucket_name: str = f"bucket:{start}-{end}"

            # 1. Get the full range of indices for this rating bracket
            start_idx: int = bisect.bisect_left(ratings, start)
            end_idx: int = bisect.bisect_left(ratings, end)
            full_range = list(range(start_idx, end_idx))

            # 2. Calculate subset size
            subset_size = int(len(full_range) * percent)

            # 3. Sample the subset
            # Using random.sample ensures you don't just get the 'easiest'
            # puzzles within that specific bucket range.
            if subset_size > 0:
                self.buckets[bucket_name] = random.sample(full_range, subset_size)
            else:
                # Handle cases where a bucket might be very small
                self.buckets[bucket_name] = []

        total_indices = sum(len(indices) for indices in self.buckets.values())
        self.len = total_indices

    def get_all_previous_buckets(self, current_bucket_key: str) -> List[str]:
        valid_prev_buckets = []
        range_part = current_bucket_key.split(":")[1]
        low, _ = map(int, range_part.split("-"))
        step_size = self.cfg.bucket_size
        min_r = self.cfg.min_rating

        # Iterate backwards until we hit the minimum rating limit
        while low > min_r:
            low -= step_size
            high = low + step_size
            prev_key = f"bucket:{low}-{high}"

            # Append only if it exists AND still has unused puzzles
            if prev_key in self.buckets and len(self.buckets[prev_key]) > 0:
                valid_prev_buckets.append(prev_key)

        return valid_prev_buckets

    def reset_buckets(self):
        self.buckets: Dict[str, List[int]] = {}
        self.create_buckets()

    def get_next_bucket(self, current_bucket_key: str) -> str:
        self.reset_buckets()

        # 1. Parse current values
        range_part = current_bucket_key.split(":")[1]
        low, high = map(int, range_part.split("-"))
        step_size = self.cfg.bucket_size

        # 2. Calculate next bucket
        next_low = low + step_size
        next_high = high + step_size
        next_key = f"bucket:{next_low}-{next_high}"

        if next_key in self.buckets:
            self.curr_bucket_key = next_key
        else:
            # If the next bucket is invalid/out of bounds,
            # loop back to the first valid bucket (min_rating)
            first_low = self.cfg.min_rating
            first_high = first_low + self.cfg.bucket_size
            self.curr_bucket_key = f"bucket:{first_low}-{first_high}"
        return self.curr_bucket_key

    def get_next_idx(self):

        if self.cfg.use_infinite_looping is False:
            if self.n >= self.__len__():
                raise StopIteration(
                    "All puzzles have been served. No more puzzles available."
                )

        if self.cfg.use_curriculum is False:
            if not self.indices:
                indices = list(range(len(self.dataset)))
                random.shuffle(indices)
            random_pos = random.randrange(len(self.indices))
            idx = self.indices.pop(random_pos)
            return idx

        indices = self.buckets[self.curr_bucket_key]
        if indices is None or len(indices) == 0:
            indices = self.buckets.get(self.get_next_bucket(self.curr_bucket_key), [])

        sample_current_bucket = random.random() < self.cfg.curr_bucket_lookup_prob

        if sample_current_bucket:
            # 1. Select and remove from current bucket
            random_pos = random.randrange(len(indices))
            idx = indices.pop(random_pos)
            return idx
        else:
            # Look up valid previous buckets
            prev_buckets = self.get_all_previous_buckets(self.curr_bucket_key)

            if not prev_buckets:
                # Fallback to current bucket if no history exists
                if not indices:
                    raise ValueError(
                        f"Bucket {self.curr_bucket_key} is empty and no history exists."
                    )

                random_pos = random.randrange(len(indices))
                idx = indices.pop(random_pos)
                return idx

            # 2. Select a random previous bucket
            selected_prev_key = random.choice(prev_buckets)
            prev_indices = self.buckets[selected_prev_key]

            # 3. Select and remove from that previous bucket
            random_pos = random.randrange(len(prev_indices))
            idx = prev_indices.pop(random_pos)

            return idx

    def get_next_item(self) -> ChessPuzzleItem:
        """Gets the next puzzle and formats it into a ChessPuzzleItem with the appropriate prompt structure."""

        idx = self.get_next_idx()
        self.curr_idx = idx

        row = self.dataset[idx]
        board = chess.Board(row["fen"])
        best_move = row["uci_moves"][0]

        user_prompt = USER_PROMPT_TEMPLATE.format(
            fen_string=row["fen"],
            ascii_board=str(board),
            legal_moves_list=", ".join(board.san(move) for move in board.legal_moves),
            turn=row["turn"],
        )

        prompt = []
        prompt.append(frozenset({"role": "system", "content": SYSTEM_PROMPT}.items()))
        prompt.append(frozenset({"role": "user", "content": user_prompt}.items()))

        self.n += 1

        return ChessPuzzleItem(
            prompt=tuple(prompt),
            best_move=best_move,
            rating=row["rating"],
            fen=row["fen"],
            tags=row["tags"],
        )
