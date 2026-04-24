import asyncio
import time
import uuid

import pytest
from chess_env.configs import CurriculumConfig, ScheduleConfig
from chess_env.curriculum_manager import CurriculumScheduler, StatefulCurriculumManager

# ==========================================
# GROUP 1: CURRICULUM SCHEDULER TESTS
# ==========================================
# (These tests test pure math/logic and do not require the dataset)


class TestCurriculumScheduler:

    def test_validation_assertions(self):
        # 1. min > max
        with pytest.raises(AssertionError):
            CurriculumScheduler(ScheduleConfig(min_rating=1000, max_rating=800))

        # 2. start < min
        with pytest.raises(AssertionError):
            CurriculumScheduler(
                ScheduleConfig(min_rating=400, max_rating=3300, start_rating=300)
            )

        # 3. Invalid discrete config
        with pytest.raises(AssertionError):
            CurriculumScheduler(
                ScheduleConfig(
                    schedule_type="fixed_discrete",
                    difficulty_levels=[1000, 2000],
                    max_steps=[100],  # Mismatched length
                )
            )

    def test_fixed_linear_progression(self):
        config = ScheduleConfig(
            schedule_type="fixed_linear",
            min_rating=400,
            max_rating=2000,
            start_rating=1000,
            total_curriculum_step=100,
            difficulty_step=1,
        )
        scheduler = CurriculumScheduler(config)

        assert scheduler.get_difficulty(0) == 1000
        assert scheduler.get_difficulty(50) == 1500
        assert scheduler.get_difficulty(100) == 2000
        assert scheduler.get_difficulty(200) == 2000

    def test_difficulty_step_bucketing(self):
        config = ScheduleConfig(
            schedule_type="fixed_linear",
            min_rating=400,
            max_rating=2000,
            start_rating=1000,
            total_curriculum_step=100,
            difficulty_step=50,
        )
        scheduler = CurriculumScheduler(config)

        # At step 1, raw diff = 1010. Should floor to 1000 because of difficulty_step=50.
        assert scheduler.get_difficulty(1) == 1000
        # At step 5, raw diff = 1050. Should be exactly 1050.
        assert scheduler.get_difficulty(5) == 1050


# ==========================================
# GROUP 2: REAL DATASET CURRICULUM TESTS
# ==========================================


class TestStatefulCurriculumManager:

    def setup_method(self):
        # Using the ACTUAL dataset but limiting to 1000 items for fast testing
        self.config = CurriculumConfig(
            dataset_name="codingmonster1234/chess-puzzles-rlvr",
            split="train",
            max_items=1000,  # CRITICAL: limits download size for the test
            schedule_config=ScheduleConfig(
                start_rating=500,
                total_curriculum_step=100,
                min_rating=400,
                max_rating=3300,
            ),
        )

    def test_initialization_and_filtering(self):
        manager = StatefulCurriculumManager(self.config)

        # We requested 1000 items from HF
        assert len(manager.dataset) == 1000
        assert manager._max_rating == 500

        # Check sorting: The real manager should have sorted the HF dataset by rating
        ratings = manager._ratings_list
        assert ratings == sorted(ratings)
        assert min(ratings) >= 400

    def test_valid_indices_and_step_progression(self):
        manager = StatefulCurriculumManager(self.config)

        initial_valid = manager.valid_indices_count

        # Fast forward the curriculum 50 steps
        manager.increment_step(50)
        assert manager.step == 50
        assert manager._max_rating > 500  # Level up occurred
        assert manager.valid_indices_count >= initial_valid  # Unlocked more puzzles

    def test_state_dict_save_load(self):
        manager = StatefulCurriculumManager(self.config)

        manager.increment_step(20)

        # Fetch a real item to get a valid UUID from the HF dataset
        real_item = manager.get_next_item()
        test_uuid = real_item.uuid
        manager._item_scores[test_uuid] = (0.9, 1)  # Mock a score update

        state = manager.state_dict()
        assert state["current_step"] == 20
        assert test_uuid in state["item_scores"]

        # Spin up new manager and load the state
        manager_2 = StatefulCurriculumManager(self.config)
        manager_2.load_state_dict(state)

        assert manager_2.step == 20
        assert manager_2._item_scores[test_uuid] == (0.9, 1)
        assert manager_2._max_rating == manager._max_rating


# ==========================================
# GROUP 3: ASYNC WORKER INTERACTION (INTEGRATION SIMULATION)
# ==========================================


@pytest.mark.asyncio
async def test_env_worker_integration():
    """
    Simulates the actual RL environment using concurrent workers trying to fetch
    items from the REAL Hugging Face dataset, processing them via mock latency,
    and dynamically scaling the curriculum.
    """

    # We use a very small subset for the integration test so it finishes quickly
    total_to_process = 200

    config = CurriculumConfig(
        dataset_name="codingmonster1234/chess-puzzles-rlvr",
        split="train",
        max_items=500,  # Pull 500 real items
        min_rating=400,
        max_rating=3300,
        schedule_config=ScheduleConfig(
            start_rating=500, total_curriculum_step=100, difficulty_step=10
        ),
    )

    manager = StatefulCurriculumManager(config)
    completed_items = 0

    # 1. Simulate the handler for an individual worker
    async def mock_handle_env(item_uuid):
        nonlocal completed_items

        # Securely pull item using the actual manager's get_next_item
        item = manager.get_next_item()

        if item is None:
            return False

        # Simulate LLM rollout processing time in the environment
        await asyncio.sleep(0.01)

        # Securely write back score and trigger curriculum steps
        completed_items += 1
        # Simulate real environment stepping logic: every 10 completions = 1 batch step
        if completed_items % 10 == 0:
            manager.increment_step(1)

        return True

    # 2. Simulate the `add_train_workers` and `env_manager` loops
    async def simulate_env_manager():
        workers = set()
        max_num_workers = 50  # Simulate 50 concurrent active rollout jobs

        while completed_items < total_to_process:
            # Clean up done workers
            done_workers = [w for w in workers if w.done()]
            for w in done_workers:
                workers.discard(w)

            # Scale up to max workers (simulate add_train_workers loop)
            while (
                len(workers) < max_num_workers
                and (completed_items + len(workers)) < total_to_process
            ):
                item_uuid = str(uuid.uuid4())
                worker_task = asyncio.create_task(mock_handle_env(item_uuid))
                workers.add(worker_task)

            # Allow event loop to tick
            await asyncio.sleep(0.005)

        # Wait for any remaining straggler tasks
        if workers:
            await asyncio.gather(*workers)

    # RUN SIMULATION
    start_time = time.time()
    await simulate_env_manager()

    # VALIDATIONS
    assert (
        completed_items == total_to_process
    ), "Workers dropped datasets or double-processed."
    assert (
        manager.step == total_to_process // 10
    ), "Curriculum did not step correctly across concurrency boundaries."

    # Ensure rating effectively scaled based on linear scheduling logic across the batch iterations
    assert manager._max_rating > manager.config.schedule_config.start_rating

    print(f"Integration test completed in {time.time() - start_time:.2f} seconds.")
