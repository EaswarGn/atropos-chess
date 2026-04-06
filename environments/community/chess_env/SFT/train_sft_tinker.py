import asyncio
import logging

from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.train import Config
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

from .dataset import HFChessDatasetBuilder

logger = logging.getLogger(__name__)


async def main():

    MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
    DATASET = "codingmonster1234/chess-reasoning-sft"
    STEPS = 250
    BATCH_SIZE = 4

    dataset_builder = HFChessDatasetBuilder(
        common_config=ChatDatasetBuilderCommonConfig(
            model_name_for_tokenizer=MODEL_NAME,
            renderer_name="qwen3_instruct",
            max_length=2048,  # Plenty of room for deep <think> reasoning
            batch_size=BATCH_SIZE,  # Adjust based on your VRAM
            train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,  # Defaults to ALL_ASSISTANT_MESSAGES
        )
    )

    config = Config(
        # Required parameters
        log_path="./logs/qwen3-chess-sft",
        model_name=MODEL_NAME,
        load_checkpoint_path=None,
        renderer_name="qwen3_instruct",
        dataset_builder=dataset_builder,
        # Training parameters
        learning_rate=2e-5,
        lr_schedule="linear",
        num_epochs=1,
        max_steps=STEPS,  # Hard cap on training steps as requested
        # Model parameters
        lora_rank=64,
        # Infrastructure parameters
        base_url=None,
        # Checkpointing and evaluation
        evaluator_builders=[],
        infrequent_evaluator_builders=[],
        save_every=50,  # Save checkpoint every 50 steps
        eval_every=50,  # Set to compute eval loss on the test set every 50 steps
        infrequent_eval_every=0,  # Disabled
        ttl_seconds=604800,  # 7 days
        rolling_save_every=10,  # Rolling state backup every 10 steps
        rolling_ttl_seconds=7200,  # 2 hours
        # Adam optimizer parameters
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
        # Logging parameters
        wandb_project="chess-reasoning-sft-tinker",
        wandb_name=f"{MODEL_NAME}_{STEPS}_steps",
        enable_trace=True,
        span_chart_every=50,
    )

    logger.info(f"Starting {MODEL_NAME} SFT on {DATASET}...")
    await train.main(config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
