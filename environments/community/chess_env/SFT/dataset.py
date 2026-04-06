import logging

import chz
import tinker
from datasets import load_dataset
from tinker_cookbook.supervised.data import (
    SupervisedDatasetFromHFDataset,
    conversation_to_datum,
)
from tinker_cookbook.supervised.types import (
    ChatDatasetBuilder,
    SupervisedDataset,
)

logger = logging.getLogger(__name__)


@chz.chz
class HFChessDatasetBuilder(ChatDatasetBuilder):
    """Builder that fetches the HF dataset and injects it into our ChessDataset."""

    dataset_name: str = "codingmonster1234/chess-reasoning-sft"
    train_split: str = "train"
    eval_split: str = "test"  # Added eval split configuration

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        # Load and build training dataset
        logger.info(f"Loading train dataset: {self.dataset_name} ({self.train_split})")
        train_hf_ds = load_dataset(self.dataset_name, split=self.train_split)

        train_dataset = SupervisedDatasetFromHFDataset(
            hf_dataset=train_hf_ds,
            batch_size=self.common_config.batch_size,
            map_fn=self.map_row_to_datum,
        )

        # Load and build evaluation dataset
        logger.info(f"Loading eval dataset: {self.dataset_name} ({self.eval_split})")
        eval_hf_ds = load_dataset(self.dataset_name, split=self.eval_split)

        eval_dataset = SupervisedDatasetFromHFDataset(
            hf_dataset=eval_hf_ds,
            batch_size=self.common_config.batch_size,
            map_fn=self.map_row_to_datum,
        )

        # Return both datasets so the training loop can compute validation metrics
        return train_dataset, eval_dataset

    def map_row_to_datum(self, row: dict) -> tinker.Datum:
        # Combine the prompt and completion lists into one chat sequence
        conversation = row["prompt"] + row["completion"]

        # Use the provided utility function
        return conversation_to_datum(
            conversation=conversation,
            renderer=self.renderer,
            max_length=self.common_config.max_length,
            train_on_what=self.common_config.train_on_what,
        )
