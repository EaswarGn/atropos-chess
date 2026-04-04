import asyncio
import json
import os
import random
import re
from typing import Dict, List, Optional, Tuple, Union

import chess
import chess.engine
from dataset import (
    ChessPuzzleItem,
    CurriculumManager,
    DatasetConfig,
)
from rich import print as rprint
from tqdm.asyncio import tqdm_asyncio

# from transformers import AutoTokenizer
from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    EvalHandlingEnum,
    ScoredDataGroup,
    logger,
)


class ChessEnv(BaseEnv):
    def __init__(
        self,
        config: BaseEnvConfig,
        server_configs: List[APIServerConfig],
        testing=False,
        slurm=False,
    ):
        """
        Initialize the Chess reasoning and playing environment.
        """
        super().__init__(config, server_configs, testing)
        self.eval_metrics = list()
        self.engine = None
        self.stockfish_path = os.getenv("STOCKFISH_PATH")

    @classmethod
    def config_init(self) -> Tuple[BaseEnvConfig, List[APIServerConfig]]:
        env_config = BaseEnvConfig(
            tokenizer_name="./base_model/checkpoint-168",
            group_size=8,
            use_wandb=False,
            max_num_workers=128,
            rollout_server_url="http://localhost:8000",
            total_steps=100,
            batch_size=2,
            steps_per_eval=20,
            max_token_length=1024 * 4,
            inference_weight=1.0,
            wandb_name="chess_reasoning_rl",
            data_path_to_save_groups=None,
            eval_handling=EvalHandlingEnum.LIMIT_TRAIN,
            eval_limit_ratio=0.1,
        )
        server_configs = [
            APIServerConfig(
                tokenizer_name="./base_model/checkpoint-168",
                base_url="http://localhost:9001/v1",
                api_key="x",
                num_requests_for_eval=256,
                server_type="vllm",
            )
        ]

        return env_config, server_configs

    async def setup(self):
        """
        Set up the environment by loading the dataset and starting Stockfish.
        """
        train_config = DatasetConfig(
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
        )
        test_config = DatasetConfig(
            dataset_name="codingmonster1234/chess-puzzles-rlvr",
            split="validation",
            use_curriculum=False,
            dataset_percent_to_use=1.0,
            use_infinite_looping=False,
        )

        self.train = CurriculumManager(cfg=train_config)
        self.test = CurriculumManager(cfg=test_config)

        rprint(
            f"Loaded dataset: [bold magenta]{len(self.train)}[/bold magenta] train | "
            f"[bold magenta]{len(self.test)}[/bold magenta] test examples"
        )

        # Initialize Async Stockfish Engine
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
            rprint(
                "[bold cyan]Stockfish[/bold cyan] engine [green]initialized successfully[/green]."
            )
        except Exception as e:
            rprint(
                f"[bold red]Error:[/bold red] Could not start [bold cyan]Stockfish[/bold cyan]. "
                f"Check path: [yellow]{self.stockfish_path}[/yellow]\n[red]{e}[/red]"
            )

        self.iter = 0

    def save_checkpoint(self, step, data=None):
        logger.info("Saving checkpoint at step %s with data %s", step, data)
        ckpt_dir = os.path.join(
            self.checkpoint_dir, "env_checkpoints", self.wandb_prepend
        )
        # create directory if necessary
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(
            self.checkpoint_dir,
            "env_checkpoints",
            self.wandb_prepend,
            f"step-{step}.json",
        )
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

        # save only the dataset to checkpoint, model saving is handled by trainer
        with open(ckpt_path, "w") as f:
            json.dump([self.train.state_dict(), self.test.state_dict()], f)

    async def get_next_item(self) -> ChessPuzzleItem:
        """
        Get the next training item from the dataset.
        """
        item = self.train.get_next_item()
        return item

    async def collect_trajectories(
        self, item: ChessPuzzleItem
    ) -> Tuple[ScoredDataGroup, List]:
        """
        Generate and collect model responses for scoring.
        """

        messages = []
        for role_dict in item.prompt:
            messages.append(dict(role_dict))

        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        async with self.server.managed_server(tokenizer=self.tokenizer) as managed:
            completions = await managed.completion(
                prompt=prompt,
                n=self.config.group_size,
                max_tokens=1024 * 4,  # Chess reasoning can be long, but keep bounded
                temperature=1.0,
            )

            state = managed.get_state()
            nodes = state["nodes"]

        to_score = list()

        for i, completion_choice in enumerate(completions.choices):
            trajectory_messages = []
            for role_dict in item.prompt:
                trajectory_messages.append(dict(role_dict))

            trajectory_messages.append(
                {"role": "assistant", "content": completion_choice.text}
            )

            to_score.append(
                {
                    "messages": tuple(trajectory_messages),
                    "best_move": item.best_move,
                    "rating": item.rating,
                    "fen": item.fen,
                    "tags": item.tags,
                    "tokens": nodes[i].tokens,
                    "masks": nodes[i].masked_tokens,
                    "logprobs": nodes[i].logprobs,
                }
            )

        scored_data = await self.score(to_score)
        to_backlog = []

        return scored_data, to_backlog

    def _extract_prediction(self, text):
        """
        Extract the chess move from the <answer> tags, ensuring <think> is present.
        """
        # Require <think> block
        think_match = re.search(
            r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE
        )
        # Require <answer> block
        answer_match = re.search(
            r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE
        )

        if not think_match or not answer_match:
            return None

        # Return just the cleaned inner text from the answer block
        return answer_match.group(1).strip()

    async def _get_move_eval(self, fen: str, move_str: str) -> float:
        """
        Evaluate a move in centipawns using Stockfish.
        Returns -10000 for invalid/illegal moves.
        """
        board = chess.Board(fen)
        try:
            # Try to parse SAN (e.g. Nf3) or UCI (e.g. g1f3)
            try:
                move = board.parse_san(move_str)
            except ValueError:
                move = board.parse_uci(move_str)

            if move not in board.legal_moves:
                return -10000.0

            board.push(move)
            # Eval time strictly limited to prevent hanging
            info = self.engine.analyse(board, chess.engine.Limit(depth=15))

            # Get score from the perspective of the player who just moved
            score = info["score"].pov(not board.turn).score(mate_score=10000)
            return float(score)
        except Exception:
            return -10000.0

    async def score(
        self, rollout_group_data
    ) -> Union[Optional[ScoredDataGroup], List[Optional[ScoredDataGroup]]]:
        """
        Score the generated model responses. 0 for format failure.
        Calculates proportional reward based on Stockfish eval difference.
        """
        scores = ScoredDataGroup()
        scores["tokens"] = list()
        scores["masks"] = list()
        scores["scores"] = list()
        scores["inference_logprobs"] = list()

        fen = rollout_group_data[0]["fen"]
        best_move = rollout_group_data[0]["best_move"]

        # Calculate the ground truth best move evaluation once per group
        eval_best = await self._get_move_eval(fen, best_move)

        random.shuffle(rollout_group_data)

        # Batch evaluate model moves to save time
        eval_tasks = []
        extracted_moves = []

        for item in rollout_group_data:
            model_response = item["messages"][-1]["content"]
            prediction = self._extract_prediction(model_response)
            extracted_moves.append(prediction)

            if prediction is None:
                # Mock task for failed formats
                async def mock_eval():
                    return None

                eval_tasks.append(mock_eval())
            else:
                eval_tasks.append(self._get_move_eval(fen, prediction))

        model_evals = await asyncio.gather(*eval_tasks)

        # print(f"\n\nscoring model responses for fen: {fen}, best move: {best_move}, eval of best move: {eval_best}")
        for i, item in enumerate(rollout_group_data):
            # print("\n\n")
            # print("_" * 50)
            # print("scoring prediction:", extracted_moves[i], "with eval:", model_evals[i])
            prediction = extracted_moves[i]
            eval_chosen = model_evals[i]

            if prediction is None:
                # Hard punishment/0 reward if the model fails the `<think>` and `<answer>` formatting
                final_score = 0.0

                # print("model failed to produce valid format, assigning score of 0.0")
            else:
                # Calculate the loss in centipawns.
                # We use max(0, ...) because the model shouldn't be penalized
                # if it finds a move even better than the 'best_move' reference.
                eval_diff = max(0.0, eval_best - eval_chosen)

                # print(f"eval of best move: {eval_best}, eval of chosen move: {eval_chosen}, diff: {eval_diff}")

                # Using a Sigmoid-like scaling function:
                # A 'scaling_factor' of 200.0 means a 200cp (2 pawn) blunder
                # results in a score of ~0.5.
                # A 0cp diff always results in 1.0.
                scaling_factor = 200.0
                final_score = 1.0 / (1.0 + (eval_diff / scaling_factor))

                # print(f"Final normalized score: {final_score}")

            # Apply length penalty
            response_tokens = len(
                self.tokenizer.encode(item["messages"][-1]["content"])
            )
            """print(f"Response tokens: {response_tokens}")
            print("Finished scoring prediction: final score (before length penalty): ", final_score)
            """

            if response_tokens > self.config.max_token_length * 0.95:
                final_score -= 0.5 * (response_tokens / self.config.max_token_length)
            """print("Final score (after length penalty): ", final_score)
            print("_" * 50)
            print("\n\n")"""

            tokens = item["tokens"]
            masks = item["masks"]
            logprobs = item["logprobs"]

            if len([1 for mask in masks if mask != -100]) < 10:
                continue

            scores["tokens"].append(tokens)
            scores["masks"].append(masks)
            scores["inference_logprobs"].append(logprobs)
            scores["scores"].append(final_score)

            if len(scores["tokens"]) >= self.config.group_size:
                break

        if not scores["scores"] or all(
            scores["scores"][0] == score for score in scores["scores"]
        ):
            return None

        return scores

    async def rollout_and_score_eval(
        self, test_item: ChessPuzzleItem
    ) -> Dict[str, Union[int, float, None]]:
        fen = test_item["fen"]
        best_move = test_item["best_move"]

        messages = [
            {"role": "system", "content": test_item.prompt[0]["content"]},
            {"role": "user", "content": test_item.prompt[1]["content"]},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        async with self.server.managed_server(tokenizer=self.tokenizer) as managed:
            completion = await managed.completion(
                prompt=prompt,
                n=1,
                max_tokens=1024 * 4,
                temperature=0.2,
                split="eval",
            )

        model_response = completion.choices[0].text
        prediction = self._extract_prediction(model_response)

        if prediction is None:
            return {"format_correct": 0, "eval_diff": None, "perfect_move": 0}

        eval_best = await self._get_move_eval(fen, best_move)
        eval_chosen = await self._get_move_eval(fen, prediction)

        eval_diff = max(0, eval_best - eval_chosen)
        perfect_move = 1 if eval_diff == 0 else 0

        return {
            "format_correct": 1,
            "eval_diff": eval_diff / 100.0,  # Diff in pawns
            "perfect_move": perfect_move,
        }

    async def evaluate(self, *args, **kwargs):
        eval_tasks = [self.rollout_and_score_eval(test_item) for test_item in self.test]
        all_scores = await tqdm_asyncio.gather(*eval_tasks)

        format_correct = [score["format_correct"] for score in all_scores]
        perfect_moves = [
            score["perfect_move"] for score in all_scores if score["format_correct"]
        ]
        eval_diffs = [
            score["eval_diff"]
            for score in all_scores
            if score["format_correct"] and score["eval_diff"] is not None
        ]

        format_accuracy = (
            sum(format_correct) / len(format_correct) if format_correct else 0
        )
        perfect_move_accuracy = (
            sum(perfect_moves) / len(perfect_moves) if perfect_moves else 0
        )
        avg_eval_diff = sum(eval_diffs) / len(eval_diffs) if eval_diffs else 0

        self.eval_metrics.append(("eval/format_accuracy", format_accuracy))
        self.eval_metrics.append(("eval/perfect_move_accuracy", perfect_move_accuracy))
        self.eval_metrics.append(("eval/avg_pawn_loss", avg_eval_diff))

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        if wandb_metrics is None:
            wandb_metrics = {}

        for item in self.eval_metrics:
            wandb_metrics[item[0]] = item[1]
        self.eval_metrics = list()

        await super().wandb_log(wandb_metrics)


if __name__ == "__main__":
    ChessEnv.cli()
