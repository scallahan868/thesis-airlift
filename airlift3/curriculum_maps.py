"""
Curriculum system for the Airlift RL environment.
Similar to the tank curriculum system, this allows for progressive training
across different scenario difficulties.
"""

import random
from curriculum import CurriculumGenerator
import json
import os
import random
import numpy as np

class AirliftCurriculumMap:
    """Base class for Airlift curriculum implementations."""

    def __init__(self, seed=None):
        self.curriculum_generator = CurriculumGenerator()
        self.seed = seed
        self.name = "AirliftCurriculumMap"
        self.reset_counter = 0

    def get_world_generator(self):
        """Returns a world generator for the current curriculum state."""
        raise NotImplementedError("Subclasses must implement get_world_generator")

    def refresh(self):
        """Called at the start of each episode to update curriculum state."""
        self.reset_counter += 1

    def __str__(self):
        return self.name


class RandomCurriculumMap(AirliftCurriculumMap):
    """Randomly selects from all 15 test scenarios each episode."""

    def __init__(self, seed=None):
        super().__init__(seed=seed)
        self.name = "RandomCurriculumMap"
        self.current_testid = 0
        self.current_seed = seed or 12345

    def refresh(self):
        """Randomly select a new testid and seed for each episode."""
        super().refresh()
        self.current_testid = random.randint(0, 14)
        # Generate a new seed based on reset counter to ensure variety
        self.current_seed = (self.seed or 12345) + self.reset_counter
        print(
            f"Episode {self.reset_counter}: testid={self.current_testid}, seed={self.current_seed}"
        )

    def get_world_generator(self):
        """Returns world generator for current random selection."""
        return self.curriculum_generator.create_world_generator(
            self.current_testid, self.current_seed
        )


class ProgressiveCurriculumMap(AirliftCurriculumMap):
    """Progressively moves through scenarios in order."""

    def __init__(self, episodes_per_scenario=100, seed=None):
        super().__init__(seed=seed)
        self.name = "ProgressiveCurriculumMap"
        self.episodes_per_scenario = episodes_per_scenario
        self.current_testid = 0
        self.current_seed = seed or 12345

    def refresh(self):
        """Move to next scenario after specified episodes."""
        super().refresh()

        with open('curriculum.json', 'r') as f:
            read_data = json.load(f)
        self.current_testid = read_data["test_id"]

        self.current_seed = (self.seed or 12345) + self.reset_counter

    def get_world_generator(self):
        """Returns world generator for current progressive scenario."""
        return self.curriculum_generator.create_world_generator(
            self.current_testid, self.current_seed
        )


class CurriculumCycleMap(AirliftCurriculumMap):
    """
    Cycles through specified test scenarios with configurable episode counts.
    Similar to the tank CurriculumCycleMap but for Airlift scenarios.
    """

    def __init__(self, testids=None, episodes_per_testid=None, seed=None):
        super().__init__(seed=seed)
        if testids is None:
            testids = [0, 1, 2, 3, 4]  # Default to first 5 scenarios
        if episodes_per_testid is None:
            episodes_per_testid = [1000] * len(testids)  # Default 1000 episodes each

        assert len(testids) == len(episodes_per_testid), "Lists must be same length"

        self.testids = testids
        self.episodes_per_testid = episodes_per_testid
        self.name = "CurriculumCycleMap"
        self.episode_counter = 0
        self.current_testid_idx = 0
        self.current_testid = self.testids[0]
        self.current_seed = seed or 12345

    def refresh(self):
        """Cycle to next scenario after specified episodes."""
        super().refresh()
        self.episode_counter += 1

        with open('curriculum.json', 'r') as f:
            read_data = json.load(f)
        self.current_testid = read_data["test_id"]

        self.current_seed = (self.seed or 12345) + self.reset_counter

    def get_world_generator(self):
        """Returns world generator for current cycle scenario."""
        return self.curriculum_generator.create_world_generator(
            self.current_testid, self.current_seed
        )

    def __str__(self):
        return f"{self.name} (testid={self.current_testid})"


class DifficultyProgressionMap(AirliftCurriculumMap):
    def __init__(self, seed=None):
        super().__init__(seed=seed)
        self.name = "DifficultyProgressionMap"
        self.current_difficulty_level = 0
        self.current_testid = 0

        self.base_seed = 12345 if seed is None else int(seed)
        self.reset_counter = 0

        # Local RNG so you don't depend on global random state
        self.rng = np.random.default_rng(self.base_seed)

        self.current_seed = self.base_seed

    def set_progression(self, difficulty_level, episode=None):
        """
        Set the current difficulty level (and optionally episode) externally.
        This should be called by the trainer or RLlib callback to control progression globally.
        """
        with open('curriculum.json', 'r') as f:
            read_data = json.load(f)
        self.current_testid = read_data["test_id"]

    def refresh(self):
        # advance counter once per refresh
        self.reset_counter += 1

        # Deterministic per-reset seed derived from base_seed
        self.current_seed = self.base_seed + self.reset_counter
        
        CURRICULUM_JSON_PATH = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../curriculum3.json")
        )
        try:
            with open(CURRICULUM_JSON_PATH, "r") as f:
                data = json.load(f)
            self.current_testid = int(data.get("test_id", self.current_testid))
        except Exception:
            # keep previous testid if file missing/invalid
            pass
        print("[Curriculum] current test id:", self.current_testid)
        print("[Curriculum] current seed:", self.current_seed)

    def get_world_generator(self):
        """Returns world generator for current difficulty level."""
        return self.curriculum_generator.create_world_generator(
            self.current_testid, self.current_seed
        )


# Registry for easy selection
CURRICULUM_REGISTRY = {
    "random": RandomCurriculumMap,
    "progressive": ProgressiveCurriculumMap,
    "cycle": CurriculumCycleMap,
    "difficulty": DifficultyProgressionMap,
}
