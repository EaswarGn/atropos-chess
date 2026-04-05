import asyncio
import json
import os
import random
import re
from typing import Dict, List, Optional, Tuple, Union

import chess
import chess.engine
from rich import box
from rich import print as rprint
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from tqdm.asyncio import tqdm_asyncio
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

# from transformers import AutoTokenizer
from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    EvalHandlingEnum,
    ScoredDataGroup,
    logger,
)

from .chess_env_types import ChessPuzzleItem
from .configs import ChessEnvConfig
from .dataset import CurriculumManager


class ChessEnv(BaseEnv):

    config: ChessEnvConfig
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
    _instance = None  # Track the active instance for cleanup

    def __init__(
        self,
        config: ChessEnvConfig,
        server_configs: List[APIServerConfig],
        testing=False,
        slurm=False,
    ):
        """
        Initialize the Chess reasoning and playing environment.
        """
        super().__init__(config, server_configs, slurm, testing)
        ChessEnv._instance = self  # Store reference to this instance
        self.eval_metrics = list()
        self.engine = None
        self.eval_semaphore = asyncio.Semaphore(self.config.max_concurrent_evals)
        self.engine_pool: asyncio.Queue[chess.engine.UciProtocol] = asyncio.Queue()
        self.dataset_lock = asyncio.Lock()

    @classmethod
    def config_init(self) -> Tuple[ChessEnvConfig, List[APIServerConfig]]:
        env_config = ChessEnvConfig(
            tokenizer_name="codingmonster1234/chess-sft-modelv2",
            group_size=8,
            use_wandb=False,
            max_num_workers=128,
            rollout_server_url="http://localhost:8000",
            total_steps=100,
            batch_size=8,
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
                tokenizer_name="codingmonster1234/chess-sft-modelv2",
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

        rprint("[yellow]Loading Dataset...[/yellow]")

        self.train = CurriculumManager(cfg=self.config.train_dataset_config)
        self.test = CurriculumManager(cfg=self.config.validation_dataset_config)

        if self.config.train_dataset_checkpoint_path is not None:
            with open(self.config.train_dataset_checkpoint_path, "r") as f:
                train_state_dict = json.load(f)
            self.train.load_state_dict(train_state_dict)
        if self.config.validation_dataset_checkpoint_path is not None:
            with open(self.config.validation_dataset_checkpoint_path, "r") as f:
                validation_state_dict = json.load(f)
            self.test.load_state_dict(validation_state_dict)

        train_info = (
            f"[bold cyan]Total:[/bold cyan] {len(self.train):,}\n"
            f"[bold cyan]Avg Elo:[/bold cyan] {self.train.avg_dataset_elo:.2f}"
        )
        test_info = (
            f"[bold magenta]Total:[/bold magenta] {len(self.test):,}\n"
            f"[bold magenta]Avg Elo:[/bold magenta] {self.test.avg_dataset_elo:.2f}"
        )

        # Wrap them in nested panels for a "Dashboard" look
        summary_panel = Panel(
            Columns(
                [
                    Panel(
                        train_info,
                        title="[bold white]Train Set[/bold white]",
                        border_style="cyan",
                    ),
                    Panel(
                        test_info,
                        title="[bold white]Test Set[/bold white]",
                        border_style="magenta",
                    ),
                ]
            ),
            title="[bold green]Dataset Loaded Successfully[/bold green]",
            expand=False,
        )

        rprint(summary_panel)

        # Initialize Async Stockfish Engine
        try:
            # self.engine = chess.engine.SimpleEngine.popen_uci(self.config.stockfish_path)

            max_engines = self.config.max_concurrent_evals
            # Initialize N independent engines
            for _ in tqdm_asyncio(range(max_engines), desc="Initializing Engine Pool"):
                transport, engine = await chess.engine.popen_uci(
                    self.config.stockfish_path
                )
                # await engine.configure({"Threads": self.config.engine_threads})
                self.engine_pool.put_nowait(engine)

            rprint(
                f"[bold cyan]{max_engines} Stockfish[/bold cyan] engine(s) [green]initialized successfully\n[/green]."
            )
        except Exception as e:
            rprint(
                f"[bold red]Error:[/bold red] Could not start [bold cyan]Stockfish[/bold cyan]. "
                f"Check path: [yellow]{self.config.stockfish_path}[/yellow]\n[red]{e}[/red]"
            )

        self.iter = 0

    def save_checkpoint(self, step, data=None):
        super().save_checkpoint(step, data=data)

        logger.info(f"Train data state: {str(self.train)}")
        logger.info(f"Validation data state: {str(self.test)}")

        ckpt_dir = os.path.join(
            self.checkpoint_dir, "dataset_checkpoints", self.wandb_prepend
        )
        # create directory if necessary
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(
            self.checkpoint_dir,
            "dataset_checkpoints",
            self.wandb_prepend,
            f"step-{step}.json",
        )
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

        # save only the dataset to checkpoint, model saving is handled by trainer
        with open(ckpt_path, "w") as f:
            json.dump([self.train.state_dict(), self.test.state_dict()], f)
        rprint("[green]Training and validation dataset dicts saved.[/green]")

    def load_checkpoint(self):
        super().load_checkpoint(self)

        if self.config.train_dataset_checkpoint_path:
            path = self.config.train_dataset_checkpoint_path
            if os.path.exists(path):
                rprint(
                    f"[bold cyan]Loading Train Dataset checkpoint from:[/bold cyan] {path}"
                )
                with open(path, "r") as f:
                    # Your save_checkpoint saves a LIST: [train_dict, test_dict]
                    full_state = json.load(f)

                    self.train.load_state_dict(full_state[0])
            else:
                rprint(
                    f"[bold yellow]Warning:[/bold yellow] Train checkpoint path {path} not found."
                )

        if self.config.validation_dataset_checkpoint_path:
            path = self.config.validation_dataset_checkpoint_path
            if os.path.exists(path):
                rprint(
                    f"[bold magenta]Loading Validation Dataset checkpoint from:[/bold magenta] {path}"
                )
                with open(path, "r") as f:
                    full_state = json.load(f)
                    # Use the second element for validation
                    self.test.load_state_dict(full_state[1])
            else:
                rprint(
                    f"[bold yellow]Warning:[/bold yellow] Validation checkpoint path {path} not found."
                )

    async def get_next_item(self) -> ChessPuzzleItem:
        """
        Get the next training item from the dataset.
        """
        # Use a lock to prevent concurrent access to the CurriculumManager
        async with self.dataset_lock:
            # Run the synchronous manager in a thread to avoid blocking the loop
            try:
                return await asyncio.to_thread(self.train.get_next_item)
            except StopIteration:
                rprint(
                    "[yellow]Dataset exhausted. Wrapping up or restarting...[/yellow]"
                )
                return None

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
                max_tokens=self.config.max_token_length,
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

    def _extract_prediction(self, rollout_item: ChessPuzzleItem, model_response: str):
        """
        Extract the chess move from the <answer> tags, ensuring <think> is present.
        """
        # Require <think> block
        think_match = re.search(
            r"<think>(.*?)</think>", model_response, re.DOTALL | re.IGNORECASE
        )
        # Require <answer> block
        answer_match = re.search(
            r"<answer>(.*?)</answer>", model_response, re.DOTALL | re.IGNORECASE
        )

        if not think_match or not answer_match:
            return "INVALID_ANSWER_FORMAT"

        # reasoning_str = think_match.group(1).strip()
        model_move_str = answer_match.group(1).strip()

        # check if model move is legal
        board = chess.Board(rollout_item.fen)
        try:
            # Try SAN first (e.g., "Nf3")
            move = board.parse_san(model_move_str)
        except ValueError:
            try:
                # If SAN fails, try UCI (e.g., "g1f3")
                move = board.parse_uci(model_move_str)
            except ValueError:
                # If both fail, the format is invalid
                return "INVALID_MOVE_FORMAT"

        if move not in board.legal_moves:
            return "ILLEGAL_MOVE"

        # Return just the cleaned inner text from the answer block
        return model_move_str

    async def _get_move_eval(
        self, engine: chess.engine.UciProtocol, fen: str, move_str: str
    ) -> float:
        """
        Evaluate a move in centipawns using Stockfish.
        Returns positive score if the move is good for the player to move, negative if bad.
        1000 centipawns is equivalent to 1 pawn.
        A score of +100 means the move is evaluated as giving a 1 pawn advantage to the player to move.
        Returns 10000 for a checkmate in favor of the player to move, -10000 for a checkmate against.
        """
        board = chess.Board(fen)
        # Try to parse SAN (e.g. Nf3) or UCI (e.g. g1f3)
        try:
            move = board.parse_san(move_str)
        except ValueError:
            move = board.parse_uci(move_str)

        board.push(move)
        # Eval time strictly limited to prevent hanging
        info = await engine.analyse(
            board,
            chess.engine.Limit(
                depth=self.config.stockfish_depth, time=self.config.engine_time_limit
            ),
        )

        # Get score from the perspective of the player who just moved
        score = info["score"].pov(not board.turn).score(mate_score=10000)
        return float(score)

    @classmethod
    async def shutdown(cls):
        """Class method to shut down engines of the active instance."""
        if cls._instance is None:
            rprint("[yellow]No active ChessEnv instance found to shut down.[/yellow]")
            return

        rprint("[bold red]Shutting down Stockfish engine pool...[/bold red]")

        pool = cls._instance.engine_pool
        while not pool.empty():
            try:
                engine = await pool.get()
                await engine.quit()
            except Exception as e:
                rprint(f"[dim red]Error closing an engine: {e}[/dim red]")

        rprint("[green]All engines shut down successfully.[/green]")
        cls._instance = None  # Reset tracker

    async def throttled_move_eval(self, fen: str, move_str: str) -> float:
        async with (
            self.eval_semaphore
        ):  # This line waits if self.config.max_concurrent_evals tasks are already running
            engine = await self.engine_pool.get()
            try:
                return await self._get_move_eval(engine, fen, move_str)
            finally:
                self.engine_pool.put_nowait(engine)

    def get_eval_reward(self, eval_best, eval_chosen):
        """Calculate a reward based on the difference in evaluation between the best move and the chosen move."""
        # Calculate the loss in centipawns.
        # We use max(0, ...) because the model shouldn't be penalized
        # if it finds a move even better than the 'best_move' reference.
        eval_diff = max(0.0, eval_best - eval_chosen)

        # Using a Sigmoid-like scaling function:
        # A 'scaling_factor' of 200.0 means a 200cp (2 pawn) blunder
        # results in a score of ~0.5.
        # A 0cp diff always results in 1.0.
        scaling_factor = self.config.reward_scaling_factor
        final_score = 1.0 / (1.0 + (eval_diff / scaling_factor))

        return final_score

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
        eval_best = await self.throttled_move_eval(fen, best_move)

        random.shuffle(rollout_group_data)

        # Batch evaluate model moves to save time
        eval_tasks = []
        extracted_moves = []

        for item in rollout_group_data:
            model_response = item["messages"][-1]["content"]
            prediction = self._extract_prediction(item, model_response)
            extracted_moves.append(prediction)

            if prediction in [
                "INVALID_ANSWER_FORMAT",
                "INVALID_MOVE_FORMAT",
                "ILLEGAL_MOVE",
            ]:
                # Mock task for failed formats
                async def mock_eval():
                    return None

                eval_tasks.append(mock_eval())
            else:
                eval_tasks.append(self.throttled_move_eval(fen, prediction))

        model_evals = await asyncio.gather(*eval_tasks)

        # print(f"\n\nscoring model responses for fen: {fen}, best move: {best_move}, eval of best move: {eval_best}")
        for i, item in enumerate(rollout_group_data):
            # print("\n\n")
            # print("_" * 50)
            # print("scoring prediction:", extracted_moves[i], "with eval:", model_evals[i])
            prediction = extracted_moves[i]
            eval_chosen = model_evals[i]

            if prediction == "INVALID_ANSWER_FORMAT":
                # Hard punishment/0 reward if the model fails the `<think>` and `<answer>` formatting
                final_score = self.config.format_fail_reward

                # print("model failed to produce valid format, assigning score of 0.0")
            elif prediction == "INVALID_MOVE_FORMAT":
                # 0.01 reward for producing a move not in UCI/SAN format but following <think> and <answer> structure
                # to encourage format correction over complete failure
                final_score = self.config.invalid_move_notation_reward

                # print("model produced invalid move format, assigning score of 0.01")
            elif prediction == "ILLEGAL_MOVE":
                # 0.1 reward for producing a move in SAN/UCI format but that is illegal
                # to encourage improvement over time rather than complete discouragement
                final_score = self.config.illegal_move_reward

                # print("model produced illegal move, assigning score of 0.1")
            else:

                final_score = self.get_eval_reward(eval_best, eval_chosen)
                # print(f"Final normalized score: {final_score}")

            # Apply length penalty
            response_tokens = len(
                self.tokenizer.encode(item["messages"][-1]["content"])
            )
            """print(f"Response tokens: {response_tokens}")
            print("Finished scoring prediction: final score (before length penalty): ", final_score)
            """

            if response_tokens > self.config.max_token_length * 0.95:
                final_score -= self.config.length_penalty_coefficient * (
                    response_tokens / self.config.max_token_length
                )
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
        """Rollout a single test item and score it for evaluation metrics."""

        fen = test_item.fen
        best_move = test_item.best_move

        messages = []
        for role_dict in test_item.prompt:
            messages.append(dict(role_dict))

        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        async with self.server.managed_server(tokenizer=self.tokenizer) as managed:
            completion = await managed.completion(
                prompt=prompt,
                n=1,
                max_tokens=self.config.max_token_length,
                temperature=0.0,
            )

        model_response = completion.choices[0].text
        prediction = self._extract_prediction(test_item, model_response)

        length_penalty = 0
        response_tokens = len(self.tokenizer.encode(model_response))
        if response_tokens > self.config.max_token_length * 0.95:
            length_penalty = self.config.length_penalty_coefficient * (
                response_tokens / self.config.max_token_length
            )

        if prediction == "INVALID_ANSWER_FORMAT":
            return {
                "format_correct": 0,
                "eval_reward": 0,
                "eval_diff": 0,
                "perfect_move": 0,
                "is_legal_move": 0,
                "is_valid_move_notation": 0,
                "length_penalty": length_penalty,
                "final_reward": 0.0 - length_penalty,
                "elo_solved": None,
            }
        if prediction == "INVALID_MOVE_FORMAT":
            return {
                "format_correct": 1,
                "eval_reward": 0,
                "eval_diff": 0,
                "perfect_move": 0,
                "is_legal_move": 0,
                "is_valid_move_notation": 0,
                "length_penalty": length_penalty,
                "final_reward": 0.01 - length_penalty,
                "elo_solved": None,
            }
        if prediction == "ILLEGAL_MOVE":
            return {
                "format_correct": 1,
                "eval_reward": 0,
                "eval_diff": 0,
                "perfect_move": 0,
                "is_legal_move": 0,
                "is_valid_move_notation": 1,
                "length_penalty": length_penalty,
                "final_reward": 0.1 - length_penalty,
                "elo_solved": None,
            }

        eval_best = await self.throttled_move_eval(fen, best_move)
        eval_chosen = await self.throttled_move_eval(fen, prediction)

        eval_reward = self.get_eval_reward(eval_best, eval_chosen)
        eval_diff = max(0.0, eval_best - eval_chosen)
        perfect_move = 1 if eval_chosen >= eval_best else 0

        return {
            "format_correct": 1,
            "eval_reward": eval_reward,
            "eval_diff": eval_diff,
            "perfect_move": perfect_move,
            "is_legal_move": 1,
            "is_valid_move_notation": 1,
            "length_penalty": length_penalty,
            "final_reward": eval_reward - length_penalty,
            "elo_solved": test_item.rating if perfect_move == 1 else None,
        }

    async def evaluate(self, *args, **kwargs):
        eval_tasks = [self.rollout_and_score_eval(test_item) for test_item in self.test]
        all_scores = await tqdm_asyncio.gather(
            *eval_tasks, desc="Scoring model responses"
        )

        format_corrects = [score["format_correct"] for score in all_scores]
        is_legal_moves = [score["is_legal_move"] for score in all_scores]
        is_valid_move_notations = [
            score["is_valid_move_notation"] for score in all_scores
        ]
        eval_rewards = [score["eval_reward"] for score in all_scores]
        eval_diffs = [score["eval_diff"] for score in all_scores]
        perfect_moves = [score["perfect_move"] for score in all_scores]
        length_penalties = [score["length_penalty"] for score in all_scores]
        final_rewards = [score["final_reward"] for score in all_scores]
        elo_solveds = [
            score["elo_solved"]
            for score in all_scores
            if score["elo_solved"] is not None
        ]

        format_accuracy = (
            sum(format_corrects) / len(format_corrects) if format_corrects else 0
        )
        legal_move_accuracy = (
            sum(is_legal_moves) / len(is_legal_moves) if is_legal_moves else 0
        )
        valid_move_notation_accuracy = (
            sum(is_valid_move_notations) / len(is_valid_move_notations)
            if is_valid_move_notations
            else 0
        )
        avg_eval_diff = sum(eval_diffs) / len(eval_diffs) if eval_diffs else 0
        perfect_move_accuracy = (
            sum(perfect_moves) / len(perfect_moves) if perfect_moves else 0
        )
        avg_length_penalty = (
            sum(length_penalties) / len(length_penalties) if length_penalties else 0
        )
        avg_final_reward = (
            sum(final_rewards) / len(final_rewards) if final_rewards else 0
        )
        avg_eval_reward = sum(eval_rewards) / len(eval_rewards) if eval_rewards else 0
        avg_elo_solved = sum(elo_solveds) / len(elo_solveds) if elo_solveds else 0

        metrics_to_add = [
            ("eval/format_accuracy", format_accuracy),
            ("eval/legal_move_accuracy", legal_move_accuracy),
            ("eval/valid_move_notation_accuracy", valid_move_notation_accuracy),
            ("eval/perfect_move_accuracy", perfect_move_accuracy),
            ("eval/avg_eval_diff", avg_eval_diff),
            ("eval/avg_length_penalty", avg_length_penalty),
            ("eval/avg_final_reward", avg_final_reward),
            ("eval/avg_eval_reward", avg_eval_reward),
            ("eval/avg_elo_solved", avg_elo_solved),
        ]

        for name, val in metrics_to_add:
            self.eval_metrics.append((name, val))

        self.print_metrics_summary()

    def print_metrics_summary(self):
        """Prints the eval_metrics list in a structured Rich table."""

        console = Console()

        table = Table(
            title="[bold blue]Chess Environment Evaluation Summary[/bold blue]",
            show_header=True,
            header_style="bold magenta",
            box=box.DOUBLE_EDGE,
            show_footer=True,
        )

        table.add_column("Category", style="cyan", no_wrap=True)
        table.add_column("Metric", style="white")
        table.add_column("Value", justify="right")

        for name, value in self.eval_metrics:
            # Split "eval/accuracy" into "eval" and "accuracy"
            category, metric_name = name.split("/") if "/" in name else ("misc", name)

            # Color coding logic
            if isinstance(value, float):
                val_str = f"{value:.4f}"
                # Green for high accuracy, Red for high pawn loss
                color = "green" if (value > 0.8 and "accuracy" in name) else "white"
                if "eval_diff" in name and value > 1.5:
                    color = "red"
            else:
                val_str = str(value)
                color = "white"

            table.add_row(category, metric_name, f"[{color}]{val_str}[/{color}]")

        console.print("\n")
        console.print(table)
        console.print("\n")

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        if wandb_metrics is None:
            wandb_metrics = {}

        for item in self.eval_metrics:
            wandb_metrics[item[0]] = item[1]
        self.eval_metrics = list()

        await super().wandb_log(wandb_metrics)


def main():
    try:
        ChessEnv.cli()
    except Exception as e:
        rprint(f"[bold red]\n[!] Critical error: {e}[/bold red]")
        raise e
    finally:
        rprint(
            "[bold yellow]\n[System] Entry point finished. Initiating final cleanup...[/bold yellow]"
        )
        try:
            # This now works because shutdown is a @classmethod
            asyncio.run(ChessEnv.shutdown())
            rprint("[bold yellow]Cleanup finished.[/bold yellow]")
        except Exception as cleanup_err:
            rprint(f"[red]Cleanup failed: {cleanup_err}[/red]")


if __name__ == "__main__":
    main()
