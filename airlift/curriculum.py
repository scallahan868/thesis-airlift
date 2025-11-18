"""
Currfrom .generators.world_generators import AirliftWorldGenerator
from .generators.airplane_generators import AirplaneGenerator
from .generators.airport_generators import RandomAirportGenerator
from .generators.route_generators import RouteByDistanceGenerator
from .generators.cargo_generators import DynamicCargoGenerator
from .generators.map_generators import PerlinMapGenerator, PlainMapGenerator
from .plane_types import PlaneType
from .events.event_interval_generator import EventIntervalGeneratoreneration with 15 predefined test cases.
Creates 15 test cases with randomly generated maps that vary in difficulty and complexity.
"""

import random
from typing import Dict, Any, List
from airlift.envs.generators.world_generators import AirliftWorldGenerator
from airlift.envs.plane_types import PlaneType
from airlift.envs.events.event_interval_generator import EventIntervalGenerator
from airlift.envs.generators.airport_generators import RandomAirportGenerator
from airlift.envs.generators.route_generators import RouteByDistanceGenerator
from airlift.envs.generators.cargo_generators import DynamicCargoGenerator
from airlift.envs.generators.airplane_generators import AirplaneGenerator
from airlift.envs.generators.map_generators import PerlinMapGenerator, PlainMapGenerator

# Curriculum system imports were already handled above


class CurriculumGenerator:
    """
    Generates 15 predefined curriculum scenarios with varying difficulty levels.
    Each test case represents a different challenge with specific parameters.
    """

    def __init__(self):
        """
        Initialize the curriculum generator with 15 predefined test cases.
        """
        self.test_cases = self._create_predefined_test_cases()

    def _create_predefined_test_cases(self) -> Dict[int, Dict[str, Any]]:
        """
        Create 15 predefined test cases with varying difficulty levels.
        Each test case has different parameters to create diverse scenarios.
        """
        test_cases = {}

        test_cases[0] = {
            "number_of_agents": 1,
            "number_of_airports": 3,
            "number_of_initial_cargo": 2,
            "max_cargo_per_episode": 2,
            "avg_working_capacity": 2,
            "avg_processing_time": 2,
            "soft_deadline_multiplier": 10,
            "hard_deadline_multiplier": 20,
            "min_degree": 2,
            "max_degree": 3,
            "route_ratio": 3.0,
            "malfunction_rate": 0.0,
            "malfunction_min_steps": 0,
            "malfunction_max_steps": 0,
            "max_soft_deadline": 5000,
            "max_hard_deadline": 5000,
            "map_type": "plain",
        }

        test_cases[1] = {
            "number_of_agents": 2,
            "number_of_airports": 3,
            "number_of_initial_cargo": 4,
            "max_cargo_per_episode": 4,
            "avg_working_capacity": 3,
            "avg_processing_time": 1,
            "soft_deadline_multiplier": 9.5,
            "hard_deadline_multiplier": 19,
            "min_degree": 2,
            "max_degree": 3,
            "route_ratio": 3.0,
            "malfunction_rate": 0.0,
            "malfunction_min_steps": 0,
            "malfunction_max_steps": 0,
            "max_soft_deadline": 4900,
            "max_hard_deadline": 4800,
            "map_type": "plain",
        }

        test_cases[2] = {
            "number_of_agents": 3,
            "number_of_airports": 3,
            "number_of_initial_cargo": 8,
            "max_cargo_per_episode": 8,
            "avg_working_capacity": 2,
            "avg_processing_time": 2,
            "soft_deadline_multiplier": 9,
            "hard_deadline_multiplier": 18,
            "min_degree": 2,
            "max_degree": 3,
            "route_ratio": 2.75,
            "malfunction_rate": 0.0,
            "malfunction_min_steps": 0,
            "malfunction_max_steps": 0,
            "max_soft_deadline": 4650,
            "max_hard_deadline": 4700,
            "map_type": "plain",
        }

        test_cases[3] = {
            "number_of_agents": 5,
            "number_of_airports": 4,
            "number_of_initial_cargo": 12,
            "max_cargo_per_episode": 12,
            "avg_working_capacity": 3,
            "avg_processing_time": 3,
            "soft_deadline_multiplier": 8.5,
            "hard_deadline_multiplier": 15,
            "min_degree": 3,
            "max_degree": 4,
            "route_ratio": 2.5,
            "malfunction_rate": 0.0,
            "malfunction_min_steps": 0,
            "malfunction_max_steps": 0,
            "max_soft_deadline": 4500,
            "max_hard_deadline": 4350,
            "map_type": "plain",
        }

        test_cases[4] = {
            "number_of_agents": 7,
            "number_of_airports": 6,
            "number_of_initial_cargo": 14,
            "max_cargo_per_episode": 14,
            "avg_working_capacity": 4,
            "avg_processing_time": 4,
            "soft_deadline_multiplier": 8,
            "hard_deadline_multiplier": 12,
            "min_degree": 3,
            "max_degree": 6,
            "route_ratio": 2.25,
            "malfunction_rate": 0.0,
            "malfunction_min_steps": 0,
            "malfunction_max_steps": 0,
            "max_soft_deadline": 4400,
            "max_hard_deadline": 4300,
            "map_type": "plain",
        }

        test_cases[5] = {
            "number_of_agents": 9,
            "number_of_airports": 8,
            "number_of_initial_cargo": 20,
            "max_cargo_per_episode": 20,
            "avg_working_capacity": 5,
            "avg_processing_time": 4,
            "soft_deadline_multiplier": 7.5,
            "hard_deadline_multiplier": 10,
            "min_degree": 3,
            "max_degree": 8,
            "route_ratio": 2.0,
            "malfunction_rate": 0.0,
            "malfunction_min_steps": 0,
            "malfunction_max_steps": 0,
            "max_soft_deadline": 4100,
            "max_hard_deadline": 4200,
            "map_type": "plain",
        }

        test_cases[6] = {
            "number_of_agents": 12,
            "number_of_airports": 9,
            "number_of_initial_cargo": 24,
            "max_cargo_per_episode": 24,
            "avg_working_capacity": 5,
            "avg_processing_time": 4,
            "soft_deadline_multiplier": 7,
            "hard_deadline_multiplier": 8,
            "min_degree": 3,
            "max_degree": 11,
            "route_ratio": 2.0,
            "malfunction_rate": 0.0,
            "malfunction_min_steps": 0,
            "malfunction_max_steps": 0,
            "max_soft_deadline": 4000,
            "max_hard_deadline": 4100,
            "map_type": "plain",
        }

        test_cases[7] = {
            "number_of_agents": 14,
            "number_of_airports": 9,
            "number_of_initial_cargo": 30,
            "max_cargo_per_episode": 30,
            "avg_working_capacity": 6,
            "avg_processing_time": 4,
            "soft_deadline_multiplier": 5,
            "hard_deadline_multiplier": 6,
            "min_degree": 3,
            "max_degree": 12,
            "route_ratio": 2.0,
            "malfunction_rate": 0.0,
            "malfunction_min_steps": 0,
            "malfunction_max_steps": 0,
            "max_soft_deadline": 3900,
            "max_hard_deadline": 4000,
            "map_type": "plain",
        }

        test_cases[8] = {
            "number_of_agents": 17,
            "number_of_airports": 10,
            "number_of_initial_cargo": 35,
            "max_cargo_per_episode": 35,
            "avg_working_capacity": 7,
            "avg_processing_time": 6,
            "soft_deadline_multiplier": 5,
            "hard_deadline_multiplier": 6,
            "min_degree": 2,
            "max_degree": 13,
            "route_ratio": 2.0,
            "malfunction_rate": 0.0,
            "malfunction_min_steps": 0,
            "malfunction_max_steps": 0,
            "max_soft_deadline": 3900,
            "max_hard_deadline": 4000,
            "map_type": "plain",
        }

        test_cases[9] = {
            "number_of_agents": 20,
            "number_of_airports": 11,
            "number_of_initial_cargo": 40,
            "max_cargo_per_episode": 40,
            "avg_working_capacity": 8,
            "avg_processing_time": 7,
            "soft_deadline_multiplier": 5,
            "hard_deadline_multiplier": 6,
            "min_degree": 2,
            "max_degree": 14,
            "route_ratio": 2.0,
            "malfunction_rate": 0.0,
            "malfunction_min_steps": 0,
            "malfunction_max_steps": 0,
            "max_soft_deadline": 3800,
            "max_hard_deadline": 3900,
            "map_type": "plain",
        }

        test_cases[10] = {
            "number_of_agents": 22,
            "number_of_airports": 12,
            "number_of_initial_cargo": 45,
            "max_cargo_per_episode": 45,
            "avg_working_capacity": 9,
            "avg_processing_time": 7,
            "soft_deadline_multiplier": 5,
            "hard_deadline_multiplier": 6,
            "min_degree": 2,
            "max_degree": 15,
            "route_ratio": 2.0,
            "malfunction_rate": 0.0,
            "malfunction_min_steps": 0,
            "malfunction_max_steps": 0,
            "max_soft_deadline": 3700,
            "max_hard_deadline": 3800,
            "map_type": "plain",
        }

        test_cases[11] = {
            "number_of_agents": 24,
            "number_of_airports": 12,
            "number_of_initial_cargo": 50,
            "max_cargo_per_episode": 50,
            "avg_working_capacity": 11.25,
            "avg_processing_time": 10,
            "soft_deadline_multiplier": 5,
            "hard_deadline_multiplier": 6,
            "min_degree": 2,
            "max_degree": 18,
            "route_ratio": 2.0,
            "malfunction_rate": 0.0,
            "malfunction_min_steps": 0,
            "malfunction_max_steps": 0,
            "max_soft_deadline": 3006,
            "max_hard_deadline": 3122,
            "map_type": "plain",
        }

        test_cases[12] = {
            "number_of_agents": 24,
            "number_of_airports": 12,
            "number_of_initial_cargo": 60,
            "max_cargo_per_episode": 60,
            "avg_working_capacity": 13,
            "avg_processing_time": 10,
            "soft_deadline_multiplier": 5,
            "hard_deadline_multiplier": 6,
            "min_degree": 0,
            "max_degree": 16,
            "route_ratio": 2.0,
            "malfunction_rate": 0.0,
            "malfunction_min_steps": 0,
            "malfunction_max_steps": 0,
            "max_soft_deadline": 2800,
            "max_hard_deadline": 2869,
            "map_type": "plain",
        }

        test_cases[13] = {
            "number_of_agents": 24,
            "number_of_airports": 12,
            "number_of_initial_cargo": 67,
            "max_cargo_per_episode": 67,
            "avg_working_capacity": 11.6666666666666,
            "avg_processing_time": 10,
            "soft_deadline_multiplier": 5,
            "hard_deadline_multiplier": 6,
            "min_degree": 2,
            "max_degree": 16,
            "route_ratio": 2.0,
            "malfunction_rate": 0.0,
            "malfunction_min_steps": 0,
            "malfunction_max_steps": 0,
            "max_soft_deadline": 2818,
            "max_hard_deadline": 2912,
            "map_type": "plain",
        }

        test_cases[14] = {
            "number_of_agents": 24,
            "number_of_airports": 12,
            "number_of_initial_cargo": 72,
            "max_cargo_per_episode": 72,
            "avg_working_capacity": 12.0833333333333,
            "avg_processing_time": 10.0,
            "soft_deadline_multiplier": 5,
            "hard_deadline_multiplier": 6,
            "min_degree": 2,
            "max_degree": 14,
            "route_ratio": 2.0,
            "malfunction_rate": 0.0,
            "malfunction_min_steps": 0,
            "malfunction_max_steps": 0,
            "max_soft_deadline": 2767,
            "max_hard_deadline": 2843,
            "map_type": "plain",
        }
        # test_cases[0] = {
        #     "number_of_agents": 4,
        #     "number_of_airports": 10,
        #     "number_of_initial_cargo": 8,
        #     "max_cargo_per_episode": 20,
        #     "avg_working_capacity": 2.5,
        #     "avg_processing_time": 10.0,
        #     "soft_deadline_multiplier": 30,
        #     "hard_deadline_multiplier": 60,
        #     "route_ratio": 2.2,
        #     "malfunction_rate": 0.02,
        #     "malfunction_min_steps": 5,
        #     "malfunction_max_steps": 15,
        #     "map_type": "perlin",
        # }

        return test_cases

    def get_test_case(self, test_id: int) -> Dict[str, Any]:
        """
        Get parameters for a specific test case.

        Args:
            test_id: Test ID (0-14)

        Returns:
            Dictionary of parameters for the test case
        """
        if test_id not in self.test_cases:
            raise ValueError(
                f"Test ID {test_id} not found. Available test IDs: {list(self.test_cases.keys())}"
            )

        return self.test_cases[test_id].copy()

    def create_world_generator(
        self, test_id: int, seed: int = None
    ) -> AirliftWorldGenerator:
        """
        Create a world generator for a specific test case.

        Args:
            test_id: Test ID (0-14)
            seed: Random seed for reproducibility

        Returns:
            Configured AirliftWorldGenerator
        """
        params = self.get_test_case(test_id)

        if seed is not None:
            random.seed(seed)

        # Create plane types using standard airlift configurations
        # Use only two established patterns from airlift examples

        plane_types = [
            PlaneType(id=0, model="A0", max_range=1.42, speed=0.21, max_weight=23),
            PlaneType(id=1, model="B1", max_range=0.9,  speed=0.90, max_weight=11),
        ]

        # Calculate derived parameters
        num_drop_off_airports = max(1, params["number_of_airports"] // 6)
        num_pick_up_airports = max(1, params["number_of_airports"] // 6)

        # Calculate cargo creation rate based on episode parameters
        cargo_creation_rate = (
            params["max_cargo_per_episode"] / 5000.0
        )  # Assuming 5000 max cycles

        # Create malfunction generator if malfunction rate > 0
        if params["malfunction_rate"] > 0:
            malfunction_generator = EventIntervalGenerator(
                min_duration=params["malfunction_min_steps"],
                max_duration=params["malfunction_max_steps"],
            )
            poisson_lambda = params["malfunction_rate"]
        else:
            malfunction_generator = EventIntervalGenerator(
                min_duration=0, max_duration=0
            )
            poisson_lambda = 0.0

        # Choose map generator based on map type
        if params["map_type"] == "plain":
            map_generator = PlainMapGenerator()
        else:
            map_generator = PerlinMapGenerator()

        # Create world generator
        world_generator = AirliftWorldGenerator(
            plane_types=plane_types,
            airport_generator=RandomAirportGenerator(
                mapgen=map_generator,
                max_airports=params["number_of_airports"],
                num_drop_off_airports=num_drop_off_airports,
                num_pick_up_airports=num_pick_up_airports,
                processing_time=int(params["avg_processing_time"]),
                working_capacity=int(params["avg_working_capacity"]),
                airports_per_unit_area=2,
            ),
            route_generator=RouteByDistanceGenerator(
                malfunction_generator=malfunction_generator,
                route_ratio=params["route_ratio"],
                poisson_lambda=poisson_lambda,
            ),
            cargo_generator=DynamicCargoGenerator(
                cargo_creation_rate=cargo_creation_rate,
                soft_deadline_multiplier=params["soft_deadline_multiplier"],
                hard_deadline_multiplier=params["hard_deadline_multiplier"],
                num_initial_tasks=params["number_of_initial_cargo"],
                max_cargo_to_create=params["max_cargo_per_episode"]
                - params["number_of_initial_cargo"],
            ),
            airplane_generator=AirplaneGenerator(params["number_of_agents"]),
            max_cycles=5000,
        )

        print("[Curriculum] Number of agents:", params["number_of_agents"])

        return world_generator

    def get_all_test_cases(self) -> List[int]:
        """
        Get list of all available test case IDs.

        Returns:
            List of test IDs (0-14)
        """
        return list(self.test_cases.keys())

    def print_test_case_summary(self, test_id: int = None):
        """
        Print a summary of test case parameters.

        Args:
            test_id: Specific test ID to print. If None, prints all test cases.
        """
        if test_id is not None:
            test_ids = [test_id]
        else:
            test_ids = self.get_all_test_cases()

        for tid in test_ids:
            params = self.get_test_case(tid)
            print(f"\n=== Test Case {tid} ===")
            print(
                f"Difficulty Level: {'Easy' if tid < 3 else 'Medium' if tid < 6 else 'Hard' if tid < 9 else 'Very Hard' if tid < 12 else 'Extreme'}"
            )
            print(f"Map Type: {params['map_type'].title()}")
            print(f"Agents: {params['number_of_agents']}")
            print(f"Airports: {params['number_of_airports']}")
            print(f"Initial Cargo: {params['number_of_initial_cargo']}")
            print(f"Max Cargo/Episode: {params['max_cargo_per_episode']}")
            print(f"Working Capacity: {params['avg_working_capacity']:.1f}")
            print(f"Processing Time: {params['avg_processing_time']:.1f}")
            print(f"Route Ratio: {params['route_ratio']:.1f}")
            print(f"Malfunction Rate: {params['malfunction_rate']:.3f}")
            if params["malfunction_rate"] > 0:
                print(
                    f"Malfunction Duration: {params['malfunction_min_steps']}-{params['malfunction_max_steps']} steps"
                )
            print(
                f"Deadlines: Soft={params['soft_deadline_multiplier']}, Hard={params['hard_deadline_multiplier']}"
            )

            # Determine plane types for this test (using only established airlift patterns)
            if tid < 5:
                print(f"Plane Types: 1 (Starter kit default: max_weight=5)")
            else:
                print(f"Plane Types: 2 (CLI example: Heavy=20, Light=10)")


# Convenience function for easy access
def create_curriculum_world_generator(
    test_id: int, seed: int = None
) -> AirliftWorldGenerator:
    """
    Convenience function to create a world generator for a specific curriculum test case.

    Args:
        test_id: Test ID (0-14)
        seed: Random seed for reproducibility

    Returns:
        Configured AirliftWorldGenerator
    """
    curriculum = CurriculumGenerator()
    return curriculum.create_world_generator(test_id, seed)


# Example usage and testing
if __name__ == "__main__":
    # Create curriculum generator
    curriculum = CurriculumGenerator()

    # Print summary of all test cases
    curriculum.print_test_case_summary()

    # Example: Create world generator for test case 5
    print(f"\n=== Creating World Generator for Test Case 5 ===")
    world_gen = curriculum.create_world_generator(test_id=5, seed=42)
    print(f"World generator created successfully!")
    print(f"Number of plane types: {len(world_gen.plane_types)}")
    print(f"Max airports: {world_gen.airport_generator.max_airports}")
    print(f"Number of agents: {world_gen.airplane_generator.num_agents}")
