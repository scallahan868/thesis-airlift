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

        # test_cases[0] = {
        #     "number_of_agents": 1,
        #     "number_of_airports": 3,
        #     "number_of_initial_cargo": 2,
        #     "max_cargo_per_episode": 2,
        #     "avg_working_capacity": 2,
        #     "avg_processing_time": 2,
        #     "soft_deadline_multiplier": 10,
        #     "hard_deadline_multiplier": 20,
        #     "min_degree": 2,
        #     "max_degree": 3,
        #     "route_ratio": 3.0,
        #     "malfunction_rate": 0.0,
        #     "malfunction_min_steps": 0,
        #     "malfunction_max_steps": 0,
        #     "max_soft_deadline": 5000,
        #     "max_hard_deadline": 5000,
        #     "map_type": "plain",
        # }

        # test_cases[1] = {
        #     "number_of_agents": 1,
        #     "number_of_airports": 3,
        #     "number_of_initial_cargo": 2,
        #     "max_cargo_per_episode": 2,
        #     "avg_working_capacity": 2,
        #     "avg_processing_time": 2,
        #     "soft_deadline_multiplier": 9.9017,
        #     "hard_deadline_multiplier": 19.7542,
        #     "min_degree": 2,
        #     "max_degree": 3,
        #     "route_ratio": 3.0,
        #     "malfunction_rate": 0.0,
        #     "malfunction_min_steps": 0,
        #     "malfunction_max_steps": 0,
        #     "max_soft_deadline": 4975,
        #     "max_hard_deadline": 4951,
        #     "map_type": "plain",
        # }

        # test_cases[2] = {
        #     "number_of_agents": 2,
        #     "number_of_airports": 3,
        #     "number_of_initial_cargo": 3,
        #     "max_cargo_per_episode": 3,
        #     "avg_working_capacity": 2,
        #     "avg_processing_time": 2,
        #     "soft_deadline_multiplier": 9.8034,
        #     "hard_deadline_multiplier": 19.5085,
        #     "min_degree": 2,
        #     "max_degree": 3,
        #     "route_ratio": 3.0,
        #     "malfunction_rate": 0.0,
        #     "malfunction_min_steps": 0,
        #     "malfunction_max_steps": 0,
        #     "max_soft_deadline": 4951,
        #     "max_hard_deadline": 4902,
        #     "map_type": "plain",
        # }

        # test_cases[3] = {
        #     "number_of_agents": 2,
        #     "number_of_airports": 3,
        #     "number_of_initial_cargo": 4,
        #     "max_cargo_per_episode": 4,
        #     "avg_working_capacity": 2.2373,
        #     "avg_processing_time": 2,
        #     "soft_deadline_multiplier": 9.6576,
        #     "hard_deadline_multiplier": 19.2627,
        #     "min_degree": 2,
        #     "max_degree": 3,
        #     "route_ratio": 3.0,
        #     "malfunction_rate": 0.0,
        #     "malfunction_min_steps": 0,
        #     "malfunction_max_steps": 0,
        #     "max_soft_deadline": 4926,
        #     "max_hard_deadline": 4876,
        #     "map_type": "plain",
        # }

        # test_cases[4] = {
        #     "number_of_agents": 2,
        #     "number_of_airports": 3,
        #     "number_of_initial_cargo": 5,
        #     "max_cargo_per_episode": 5,
        #     "avg_working_capacity": 2.4831,
        #     "avg_processing_time": 2,
        #     "soft_deadline_multiplier": 9.5102,
        #     "hard_deadline_multiplier": 19.0169,
        #     "min_degree": 2,
        #     "max_degree": 4,
        #     "route_ratio": 3.0,
        #     "malfunction_rate": 0.0,
        #     "malfunction_min_steps": 0,
        #     "malfunction_max_steps": 0,
        #     "max_soft_deadline": 4902,
        #     "max_hard_deadline": 4852,
        #     "map_type": "plain",
        # }

        # test_cases[5] = {
        #     "number_of_agents": 2,
        #     "number_of_airports": 3,
        #     "number_of_initial_cargo": 6,
        #     "max_cargo_per_episode": 6,
        #     "avg_working_capacity": 2.5458,
        #     "avg_processing_time": 2,
        #     "soft_deadline_multiplier": 9.4085,
        #     "hard_deadline_multiplier": 18.8169,
        #     "min_degree": 2,
        #     "max_degree": 4,
        #     "route_ratio": 2.9542,
        #     "malfunction_rate": 0.0,
        #     "malfunction_min_steps": 0,
        #     "malfunction_max_steps": 0,
        #     "max_soft_deadline": 4854,
        #     "max_hard_deadline": 4804,
        #     "map_type": "plain",
        # }

        # test_cases[6] = {
        #     "number_of_agents": 3,
        #     "number_of_airports": 3,
        #     "number_of_initial_cargo": 7,
        #     "max_cargo_per_episode": 7,
        #     "avg_working_capacity": 2.5949,
        #     "avg_processing_time": 2,
        #     "soft_deadline_multiplier": 9.3102,
        #     "hard_deadline_multiplier": 18.6203,
        #     "min_degree": 2,
        #     "max_degree": 4,
        #     "route_ratio": 2.9051,
        #     "malfunction_rate": 0.0,
        #     "malfunction_min_steps": 0,
        #     "malfunction_max_steps": 0,
        #     "max_soft_deadline": 4805,
        #     "max_hard_deadline": 4755,
        #     "map_type": "plain",
        # }

        # test_cases[7] = {
        #     "number_of_agents": 3,
        #     "number_of_airports": 3,
        #     "number_of_initial_cargo": 8,
        #     "max_cargo_per_episode": 8,
        #     "avg_working_capacity": 2.7763,
        #     "avg_processing_time": 2,
        #     "soft_deadline_multiplier": 9.1678,
        #     "hard_deadline_multiplier": 18.3356,
        #     "min_degree": 2,
        #     "max_degree": 4,
        #     "route_ratio": 2.8559,
        #     "malfunction_rate": 0.0,
        #     "malfunction_min_steps": 0,
        #     "malfunction_max_steps": 0,
        #     "max_soft_deadline": 4756,
        #     "max_hard_deadline": 4706,
        #     "map_type": "plain",
        # }

        # test_cases[8] = {
        #     "number_of_agents": 4,
        #     "number_of_airports": 3,
        #     "number_of_initial_cargo": 10,
        #     "max_cargo_per_episode": 10,
        #     "avg_working_capacity": 2.9729,
        #     "avg_processing_time": 2,
        #     "soft_deadline_multiplier": 9.0203,
        #     "hard_deadline_multiplier": 18.0407,
        #     "min_degree": 2,
        #     "max_degree": 5,
        #     "route_ratio": 2.8068,
        #     "malfunction_rate": 0.0,
        #     "malfunction_min_steps": 0,
        #     "malfunction_max_steps": 0,
        #     "max_soft_deadline": 4707,
        #     "max_hard_deadline": 4657,
        #     "map_type": "plain",
        # }

        # test_cases[9] = {
        #     "number_of_agents": 4,
        #     "number_of_airports": 3,
        #     "number_of_initial_cargo": 11,
        #     "max_cargo_per_episode": 11,
        #     "avg_working_capacity": 3,
        #     "avg_processing_time": 2,
        #     "soft_deadline_multiplier": 8.9153,
        #     "hard_deadline_multiplier": 17.7881,
        #     "min_degree": 2,
        #     "max_degree": 5,
        #     "route_ratio": 2.7576,
        #     "malfunction_rate": 0.0,
        #     "malfunction_min_steps": 0,
        #     "malfunction_max_steps": 0,
        #     "max_soft_deadline": 4679,
        #     "max_hard_deadline": 4629,
        #     "map_type": "plain",
        # }

        # test_cases[10] = {
        #     "number_of_agents": 4,
        #     "number_of_airports": 3,
        #     "number_of_initial_cargo": 12,
        #     "max_cargo_per_episode": 12,
        #     "avg_working_capacity": 3,
        #     "avg_processing_time": 3,
        #     "soft_deadline_multiplier": 8.8169,
        #     "hard_deadline_multiplier": 17.5424,
        #     "min_degree": 2,
        #     "max_degree": 5,
        #     "route_ratio": 2.7085,
        #     "malfunction_rate": 0.0,
        #     "malfunction_min_steps": 0,
        #     "malfunction_max_steps": 0,
        #     "max_soft_deadline": 4654,
        #     "max_hard_deadline": 4604,
        #     "map_type": "plain",
        # }

        # test_cases[11] = {
        #     "number_of_agents": 5,
        #     "number_of_airports": 3,
        #     "number_of_initial_cargo": 13,
        #     "max_cargo_per_episode": 13,
        #     "avg_working_capacity": 3.0814,
        #     "avg_processing_time": 3,
        #     "soft_deadline_multiplier": 8.678,
        #     "hard_deadline_multiplier": 17.2966,
        #     "min_degree": 2,
        #     "max_degree": 5,
        #     "route_ratio": 2.6593,
        #     "malfunction_rate": 0.0,
        #     "malfunction_min_steps": 0,
        #     "malfunction_max_steps": 0,
        #     "max_soft_deadline": 4609,
        #     "max_hard_deadline": 4559,
        #     "map_type": "plain",
        # }

        # test_cases[12] = {
        #     "number_of_agents": 5,
        #     "number_of_airports": 4,
        #     "number_of_initial_cargo": 14,
        #     "max_cargo_per_episode": 14,
        #     "avg_working_capacity": 3.1797,
        #     "avg_processing_time": 3,
        #     "soft_deadline_multiplier": 8.5305,
        #     "hard_deadline_multiplier": 17.0508,
        #     "min_degree": 2,
        #     "max_degree": 6,
        #     "route_ratio": 2.6102,
        #     "malfunction_rate": 0.0,
        #     "malfunction_min_steps": 0,
        #     "malfunction_max_steps": 0,
        #     "max_soft_deadline": 4560,
        #     "max_hard_deadline": 4510,
        #     "map_type": "plain",
        # }

        # test_cases[13] = {
        #     "number_of_agents": 5,
        #     "number_of_airports": 4,
        #     "number_of_initial_cargo": 15,
        #     "max_cargo_per_episode": 15,
        #     "avg_working_capacity": 3.278,
        #     "avg_processing_time": 3,
        #     "soft_deadline_multiplier": 8.422,
        #     "hard_deadline_multiplier": 16.8051,
        #     "min_degree": 2,
        #     "max_degree": 6,
        #     "route_ratio": 2.561,
        #     "malfunction_rate": 0.0,
        #     "malfunction_min_steps": 0,
        #     "malfunction_max_steps": 0,
        #     "max_soft_deadline": 4531,
        #     "max_hard_deadline": 4461,
        #     "map_type": "plain",
        # }

        # test_cases[14] = {
        #     "number_of_agents": 6,
        #     "number_of_airports": 4,
        #     "number_of_initial_cargo": 16,
        #     "max_cargo_per_episode": 16,
        #     "avg_working_capacity": 3.3763,
        #     "avg_processing_time": 3,
        #     "soft_deadline_multiplier": 8.3237,
        #     "hard_deadline_multiplier": 16.5593,
        #     "min_degree": 2,
        #     "max_degree": 6,
        #     "route_ratio": 2.5119,
        #     "malfunction_rate": 0.0,
        #     "malfunction_min_steps": 0,
        #     "malfunction_max_steps": 0,
        #     "max_soft_deadline": 4506,
        #     "max_hard_deadline": 4412,
        #     "map_type": "plain",
        # }

        # test_cases[15] = {
        #     "number_of_agents": 6,
        #     "number_of_airports": 4,
        #     "number_of_initial_cargo": 17,
        #     "max_cargo_per_episode": 17,
        #     "avg_working_capacity": 3.4373,
        #     "avg_processing_time": 3,
        #     "soft_deadline_multiplier": 8.1881,
        #     "hard_deadline_multiplier": 16.3136,
        #     "min_degree": 2,
        #     "max_degree": 6,
        #     "route_ratio": 2.4814,
        #     "malfunction_rate": 0.0,
        #     "malfunction_min_steps": 0,
        #     "malfunction_max_steps": 0,
        #     "max_soft_deadline": 4463,
        #     "max_hard_deadline": 4381,
        #     "map_type": "plain",
        # }

        # test_cases[16] = {
        #     "number_of_agents": 6,
        #     "number_of_airports": 4,
        #     "number_of_initial_cargo": 18,
        #     "max_cargo_per_episode": 18,
        #     "avg_working_capacity": 3.4864,
        #     "avg_processing_time": 3,
        #     "soft_deadline_multiplier": 8.0407,
        #     "hard_deadline_multiplier": 16.0678,
        #     "min_degree": 2,
        #     "max_degree": 7,
        #     "route_ratio": 2.4568,
        #     "malfunction_rate": 0.0,
        #     "malfunction_min_steps": 0,
        #     "malfunction_max_steps": 0,
        #     "max_soft_deadline": 4414,
        #     "max_hard_deadline": 4357,
        #     "map_type": "plain",
        # }

        # test_cases[17] = {
        #     "number_of_agents": 7,
        #     "number_of_airports": 4,
        #     "number_of_initial_cargo": 19,
        #     "max_cargo_per_episode": 19,
        #     "avg_working_capacity": 3.678,
        #     "avg_processing_time": 3,
        #     "soft_deadline_multiplier": 7.9288,
        #     "hard_deadline_multiplier": 15.822,
        #     "min_degree": 2,
        #     "max_degree": 7,
        #     "route_ratio": 2.4322,
        #     "malfunction_rate": 0.0,
        #     "malfunction_min_steps": 0,
        #     "malfunction_max_steps": 0,
        #     "max_soft_deadline": 4364,
        #     "max_hard_deadline": 4314,
        #     "map_type": "plain",
        # }

        # test_cases[18] = {
        #     "number_of_agents": 7,
        #     "number_of_airports": 4,
        #     "number_of_initial_cargo": 20,
        #     "max_cargo_per_episode": 20,
        #     "avg_working_capacity": 3.9237,
        #     "avg_processing_time": 3,
        #     "soft_deadline_multiplier": 7.8305,
        #     "hard_deadline_multiplier": 15.5763,
        #     "min_degree": 3,
        #     "max_degree": 8,
        #     "route_ratio": 2.4076,
        #     "malfunction_rate": 0.0,
        #     "malfunction_min_steps": 0,
        #     "malfunction_max_steps": 0,
        #     "max_soft_deadline": 4315,
        #     "max_hard_deadline": 4265,
        #     "map_type": "plain",
        # }

        # test_cases[19] = {
        #     "number_of_agents": 7,
        #     "number_of_airports": 4,
        #     "number_of_initial_cargo": 21,
        #     "max_cargo_per_episode": 21,
        #     "avg_working_capacity": 4.1695,
        #     "avg_processing_time": 3,
        #     "soft_deadline_multiplier": 7.6983,
        #     "hard_deadline_multiplier": 15.3305,
        #     "min_degree": 3,
        #     "max_degree": 8,
        #     "route_ratio": 2.3661,
        #     "malfunction_rate": 0.0,
        #     "malfunction_min_steps": 0,
        #     "malfunction_max_steps": 0,
        #     "max_soft_deadline": 4266,
        #     "max_hard_deadline": 4216,
        #     "map_type": "plain",
        # }

        # test_cases[20] = {
        #     "number_of_agents": 8,
        #     "number_of_airports": 4,
        #     "number_of_initial_cargo": 22,
        #     "max_cargo_per_episode": 22,
        #     "avg_working_capacity": 4.4153,
        #     "avg_processing_time": 3,
        #     "soft_deadline_multiplier": 7.5508,
        #     "hard_deadline_multiplier": 15.0847,
        #     "min_degree": 3,
        #     "max_degree": 8,
        #     "route_ratio": 2.3169,
        #     "malfunction_rate": 0.0,
        #     "malfunction_min_steps": 0,
        #     "malfunction_max_steps": 0,
        #     "max_soft_deadline": 4217,
        #     "max_hard_deadline": 4167,
        #     "map_type": "plain",
        # }

        # test_cases[21] = {
        #     "number_of_agents": 8,
        #     "number_of_airports": 4,
        #     "number_of_initial_cargo": 23,
        #     "max_cargo_per_episode": 23,
        #     "avg_working_capacity": 4.5,
        #     "avg_processing_time": 3.0644,
        #     "soft_deadline_multiplier": 7.4034,
        #     "hard_deadline_multiplier": 14.839,
        #     "min_degree": 3,
        #     "max_degree": 8,
        #     "route_ratio": 2.2678,
        #     "malfunction_rate": 0.0,
        #     "malfunction_min_steps": 0,
        #     "malfunction_max_steps": 0,
        #     "max_soft_deadline": 4168,
        #     "max_hard_deadline": 4118,
        #     "map_type": "plain",
        # }

        # test_cases[22] = {
        #     "number_of_agents": 9,
        #     "number_of_airports": 4,
        #     "number_of_initial_cargo": 24,
        #     "max_cargo_per_episode": 24,
        #     "avg_working_capacity": 4.5,
        #     "avg_processing_time": 3.1627,
        #     "soft_deadline_multiplier": 7.2559,
        #     "hard_deadline_multiplier": 14.5932,
        #     "min_degree": 3,
        #     "max_degree": 9,
        #     "route_ratio": 2.2186,
        #     "malfunction_rate": 0.0,
        #     "malfunction_min_steps": 0,
        #     "malfunction_max_steps": 0,
        #     "max_soft_deadline": 4119,
        #     "max_hard_deadline": 4069,
        #     "map_type": "plain",
        # }

        # test_cases[23] = {
        #     "number_of_agents": 9,
        #     "number_of_airports": 5,
        #     "number_of_initial_cargo": 26,
        #     "max_cargo_per_episode": 26,
        #     "avg_working_capacity": 4.6525,
        #     "avg_processing_time": 3.2915,
        #     "soft_deadline_multiplier": 7.139,
        #     "hard_deadline_multiplier": 14.3475,
        #     "min_degree": 3,
        #     "max_degree": 9,
        #     "route_ratio": 2.1695,
        #     "malfunction_rate": 0.0,
        #     "malfunction_min_steps": 0,
        #     "malfunction_max_steps": 0,
        #     "max_soft_deadline": 4069,
        #     "max_hard_deadline": 4019,
        #     "map_type": "plain",
        # }

        # test_cases[24] = {
        #     "number_of_agents": 10,
        #     "number_of_airports": 5,
        #     "number_of_initial_cargo": 27,
        #     "max_cargo_per_episode": 27,
        #     "avg_working_capacity": 4.8983,
        #     "avg_processing_time": 3.439,
        #     "soft_deadline_multiplier": 7.0407,
        #     "hard_deadline_multiplier": 14.1017,
        #     "min_degree": 3,
        #     "max_degree": 10,
        #     "route_ratio": 2.1203,
        #     "malfunction_rate": 0.0,
        #     "malfunction_min_steps": 0,
        #     "malfunction_max_steps": 0,
        #     "max_soft_deadline": 4020,
        #     "max_hard_deadline": 3970,
        #     "map_type": "plain",
        # }

        # test_cases[25] = {
        #     "number_of_agents": 10,
        #     "number_of_airports": 5,
        #     "number_of_initial_cargo": 29,
        #     "max_cargo_per_episode": 29,
        #     "avg_working_capacity": 5.1441,
        #     "avg_processing_time": 3.5864,
        #     "soft_deadline_multiplier": 6.9424,
        #     "hard_deadline_multiplier": 13.8559,
        #     "min_degree": 3,
        #     "max_degree": 10,
        #     "route_ratio": 2.0856,
        #     "malfunction_rate": 0.0,
        #     "malfunction_min_steps": 0,
        #     "malfunction_max_steps": 0,
        #     "max_soft_deadline": 3986,
        #     "max_hard_deadline": 3936,
        #     "map_type": "plain",
        # }

        # test_cases[26] = {
        #     "number_of_agents": 11,
        #     "number_of_airports": 6,
        #     "number_of_initial_cargo": 30,
        #     "max_cargo_per_episode": 30,
        #     "avg_working_capacity": 5.3898,
        #     "avg_processing_time": 3.7339,
        #     "soft_deadline_multiplier": 6.8441,
        #     "hard_deadline_multiplier": 13.6102,
        #     "min_degree": 3,
        #     "max_degree": 11,
        #     "route_ratio": 2.061,
        #     "malfunction_rate": 0.0,
        #     "malfunction_min_steps": 0,
        #     "malfunction_max_steps": 0,
        #     "max_soft_deadline": 3961,
        #     "max_hard_deadline": 3911,
        #     "map_type": "plain",
        # }

        # test_cases[27] = {
        #     "number_of_agents": 11,
        #     "number_of_airports": 6,
        #     "number_of_initial_cargo": 31,
        #     "max_cargo_per_episode": 31,
        #     "avg_working_capacity": 5.6356,
        #     "avg_processing_time": 3.8542,
        #     "soft_deadline_multiplier": 6.7458,
        #     "hard_deadline_multiplier": 13.3644,
        #     "min_degree": 3,
        #     "max_degree": 11,
        #     "route_ratio": 2.0364,
        #     "malfunction_rate": 0.0,
        #     "malfunction_min_steps": 0,
        #     "malfunction_max_steps": 0,
        #     "max_soft_deadline": 3936,
        #     "max_hard_deadline": 3886,
        #     "map_type": "plain",
        # }

        # test_cases[28] = {
        #     "number_of_agents": 12,
        #     "number_of_airports": 6,
        #     "number_of_initial_cargo": 32,
        #     "max_cargo_per_episode": 32,
        #     "avg_working_capacity": 5.8814,
        #     "avg_processing_time": 3.9525,
        #     "soft_deadline_multiplier": 6.6475,
        #     "hard_deadline_multiplier": 13.1186,
        #     "min_degree": 3,
        #     "max_degree": 12,
        #     "route_ratio": 2.0119,
        #     "malfunction_rate": 0.0,
        #     "malfunction_min_steps": 0,
        #     "malfunction_max_steps": 0,
        #     "max_soft_deadline": 3912,
        #     "max_hard_deadline": 3862,
        #     "map_type": "plain",
        # }

        # test_cases[29] = {
        #     "number_of_agents": 12,
        #     "number_of_airports": 6,
        #     "number_of_initial_cargo": 33,
        #     "max_cargo_per_episode": 33,
        #     "avg_working_capacity": 6.1271,
        #     "avg_processing_time": 4.0508,
        #     "soft_deadline_multiplier": 6.5492,
        #     "hard_deadline_multiplier": 12.9492,
        #     "min_degree": 3,
        #     "max_degree": 12,
        #     "route_ratio": 2.0,
        #     "malfunction_rate": 0.0,
        #     "malfunction_min_steps": 0,
        #     "malfunction_max_steps": 0,
        #     "max_soft_deadline": 3887,
        #     "max_hard_deadline": 3837,
        #     "map_type": "plain",
        # }

        # test_cases[30] = {
        #     "number_of_agents": 13,
        #     "number_of_airports": 6,
        #     "number_of_initial_cargo": 34,
        #     "max_cargo_per_episode": 34,
        #     "avg_working_capacity": 6.3729,
        #     "avg_processing_time": 4.1492,
        #     "soft_deadline_multiplier": 6.4508,
        #     "hard_deadline_multiplier": 12.8508,
        #     "min_degree": 3,
        #     "max_degree": 13,
        #     "route_ratio": 2.0,
        #     "malfunction_rate": 0.0,
        #     "malfunction_min_steps": 0,
        #     "malfunction_max_steps": 0,
        #     "max_soft_deadline": 3863,
        #     "max_hard_deadline": 3813,
        #     "map_type": "plain",
        # }

        # test_cases[31] = {
        #     "number_of_agents": 13,
        #     "number_of_airports": 7,
        #     "number_of_initial_cargo": 36,
        #     "max_cargo_per_episode": 36,
        #     "avg_working_capacity": 6.6186,
        #     "avg_processing_time": 4.2475,
        #     "soft_deadline_multiplier": 6.3525,
        #     "hard_deadline_multiplier": 12.7525,
        #     "min_degree": 3,
        #     "max_degree": 13,
        #     "route_ratio": 2.0,
        #     "malfunction_rate": 0.0,
        #     "malfunction_min_steps": 0,
        #     "malfunction_max_steps": 0,
        #     "max_soft_deadline": 3838,
        #     "max_hard_deadline": 3788,
        #     "map_type": "plain",
        # }

        # test_cases[32] = {
        #     "number_of_agents": 14,
        #     "number_of_airports": 7,
        #     "number_of_initial_cargo": 37,
        #     "max_cargo_per_episode": 37,
        #     "avg_working_capacity": 6.8644,
        #     "avg_processing_time": 4.3458,
        #     "soft_deadline_multiplier": 6.2542,
        #     "hard_deadline_multiplier": 12.6542,
        #     "min_degree": 3,
        #     "max_degree": 13,
        #     "route_ratio": 2.0,
        #     "malfunction_rate": 0.0,
        #     "malfunction_min_steps": 0,
        #     "malfunction_max_steps": 0,
        #     "max_soft_deadline": 3814,
        #     "max_hard_deadline": 3764,
        #     "map_type": "plain",
        # }

        # test_cases[33] = {
        #     "number_of_agents": 14,
        #     "number_of_airports": 7,
        #     "number_of_initial_cargo": 38,
        #     "max_cargo_per_episode": 38,
        #     "avg_working_capacity": 7.0,
        #     "avg_processing_time": 4.422,
        #     "soft_deadline_multiplier": 6.1559,
        #     "hard_deadline_multiplier": 12.5339,
        #     "min_degree": 3,
        #     "max_degree": 13,
        #     "route_ratio": 2.0,
        #     "malfunction_rate": 0.0,
        #     "malfunction_min_steps": 0,
        #     "malfunction_max_steps": 0,
        #     "max_soft_deadline": 3789,
        #     "max_hard_deadline": 3739,
        #     "map_type": "plain",
        # }

        # test_cases[34] = {
        #     "number_of_agents": 15,
        #     "number_of_airports": 8,
        #     "number_of_initial_cargo": 39,
        #     "max_cargo_per_episode": 39,
        #     "avg_working_capacity": 7.0,
        #     "avg_processing_time": 4.4712,
        #     "soft_deadline_multiplier": 6.0576,
        #     "hard_deadline_multiplier": 12.3864,
        #     "min_degree": 3,
        #     "max_degree": 13,
        #     "route_ratio": 2.0,
        #     "malfunction_rate": 0.0,
        #     "malfunction_min_steps": 0,
        #     "malfunction_max_steps": 0,
        #     "max_soft_deadline": 3764,
        #     "max_hard_deadline": 3714,
        #     "map_type": "plain",
        # }

        # test_cases[35] = {
        #     "number_of_agents": 15,
        #     "number_of_airports": 8,
        #     "number_of_initial_cargo": 41,
        #     "max_cargo_per_episode": 41,
        #     "avg_working_capacity": 7.1017,
        #     "avg_processing_time": 4.5407,
        #     "soft_deadline_multiplier": 5.9593,
        #     "hard_deadline_multiplier": 12.239,
        #     "min_degree": 3,
        #     "max_degree": 13,
        #     "route_ratio": 2.0,
        #     "malfunction_rate": 0.0,
        #     "malfunction_min_steps": 0,
        #     "malfunction_max_steps": 0,
        #     "max_soft_deadline": 3740,
        #     "max_hard_deadline": 3690,
        #     "map_type": "plain",
        # }

        # test_cases[36] = {
        #     "number_of_agents": 16,
        #     "number_of_airports": 8,
        #     "number_of_initial_cargo": 42,
        #     "max_cargo_per_episode": 42,
        #     "avg_working_capacity": 7.3475,
        #     "avg_processing_time": 4.639,
        #     "soft_deadline_multiplier": 5.861,
        #     "hard_deadline_multiplier": 12.0915,
        #     "min_degree": 3,
        #     "max_degree": 14,
        #     "route_ratio": 2.0,
        #     "malfunction_rate": 0.0,
        #     "malfunction_min_steps": 0,
        #     "malfunction_max_steps": 0,
        #     "max_soft_deadline": 3715,
        #     "max_hard_deadline": 3665,
        #     "map_type": "plain",
        # }

        # test_cases[37] = {
        #     "number_of_agents": 16,
        #     "number_of_airports": 9,
        #     "number_of_initial_cargo": 43,
        #     "max_cargo_per_episode": 43,
        #     "avg_working_capacity": 7.5932,
        #     "avg_processing_time": 4.7559,
        #     "soft_deadline_multiplier": 5.7627,
        #     "hard_deadline_multiplier": 11.9627,
        #     "min_degree": 3,
        #     "max_degree": 14,
        #     "route_ratio": 2.0,
        #     "malfunction_rate": 0.0,
        #     "malfunction_min_steps": 0,
        #     "malfunction_max_steps": 0,
        #     "max_soft_deadline": 3691,
        #     "max_hard_deadline": 3641,
        #     "map_type": "plain",
        # }

        # test_cases[38] = {
        #     "number_of_agents": 17,
        #     "number_of_airports": 9,
        #     "number_of_initial_cargo": 44,
        #     "max_cargo_per_episode": 44,
        #     "avg_working_capacity": 7.839,
        #     "avg_processing_time": 4.9034,
        #     "soft_deadline_multiplier": 5.6644,
        #     "hard_deadline_multiplier": 11.8644,
        #     "min_degree": 3,
        #     "max_degree": 14,
        #     "route_ratio": 2.0,
        #     "malfunction_rate": 0.0,
        #     "malfunction_min_steps": 0,
        #     "malfunction_max_steps": 0,
        #     "max_soft_deadline": 3666,
        #     "max_hard_deadline": 3616,
        #     "map_type": "plain",
        # }

        # test_cases[39] = {
        #     "number_of_agents": 17,
        #     "number_of_airports": 9,
        #     "number_of_initial_cargo": 46,
        #     "max_cargo_per_episode": 46,
        #     "avg_working_capacity": 8.0847,
        #     "avg_processing_time": 5.0847,
        #     "soft_deadline_multiplier": 5.5661,
        #     "hard_deadline_multiplier": 11.7492,
        #     "min_degree": 3,
        #     "max_degree": 14,
        #     "route_ratio": 2.0,
        #     "malfunction_rate": 0.0,
        #     "malfunction_min_steps": 0,
        #     "malfunction_max_steps": 0,
        #     "max_soft_deadline": 3642,
        #     "max_hard_deadline": 3592,
        #     "map_type": "plain",
        # }

        # test_cases[40] = {
        #     "number_of_agents": 18,
        #     "number_of_airports": 9,
        #     "number_of_initial_cargo": 47,
        #     "max_cargo_per_episode": 47,
        #     "avg_working_capacity": 8.3305,
        #     "avg_processing_time": 5.3305,
        #     "soft_deadline_multiplier": 5.4678,
        #     "hard_deadline_multiplier": 11.6017,
        #     "min_degree": 3,
        #     "max_degree": 14,
        #     "route_ratio": 2.0,
        #     "malfunction_rate": 0.0,
        #     "malfunction_min_steps": 0,
        #     "malfunction_max_steps": 0,
        #     "max_soft_deadline": 3617,
        #     "max_hard_deadline": 3567,
        #     "map_type": "plain",
        # }

        # test_cases[41] = {
        #     "number_of_agents": 18,
        #     "number_of_airports": 9,
        #     "number_of_initial_cargo": 48,
        #     "max_cargo_per_episode": 48,
        #     "avg_working_capacity": 8.5763,
        #     "avg_processing_time": 5.5763,
        #     "soft_deadline_multiplier": 5.3847,
        #     "hard_deadline_multiplier": 11.4695,
        #     "min_degree": 3,
        #     "max_degree": 14,
        #     "route_ratio": 2.0,
        #     "malfunction_rate": 0.0,
        #     "malfunction_min_steps": 0,
        #     "malfunction_max_steps": 0,
        #     "max_soft_deadline": 3592,
        #     "max_hard_deadline": 3542,
        #     "map_type": "plain",
        # }

        # test_cases[42] = {
        #     "number_of_agents": 19,
        #     "number_of_airports": 10,
        #     "number_of_initial_cargo": 49,
        #     "max_cargo_per_episode": 49,
        #     "avg_working_capacity": 8.822,
        #     "avg_processing_time": 5.822,
        #     "soft_deadline_multiplier": 5.3356,
        #     "hard_deadline_multiplier": 11.3712,
        #     "min_degree": 3,
        #     "max_degree": 15,
        #     "route_ratio": 2.0,
        #     "malfunction_rate": 0.0,
        #     "malfunction_min_steps": 0,
        #     "malfunction_max_steps": 0,
        #     "max_soft_deadline": 3568,
        #     "max_hard_deadline": 3518,
        #     "map_type": "plain",
        # }

        # test_cases[43] = {
        #     "number_of_agents": 19,
        #     "number_of_airports": 10,
        #     "number_of_initial_cargo": 50,
        #     "max_cargo_per_episode": 50,
        #     "avg_working_capacity": 9.0678,
        #     "avg_processing_time": 6.0,
        #     "soft_deadline_multiplier": 5.2864,
        #     "hard_deadline_multiplier": 11.2729,
        #     "min_degree": 3,
        #     "max_degree": 15,
        #     "route_ratio": 2.0,
        #     "malfunction_rate": 0.0,
        #     "malfunction_min_steps": 0,
        #     "malfunction_max_steps": 0,
        #     "max_soft_deadline": 3543,
        #     "max_hard_deadline": 3493,
        #     "map_type": "plain",
        # }

        # test_cases[44] = {
        #     "number_of_agents": 20,
        #     "number_of_airports": 10,
        #     "number_of_initial_cargo": 52,
        #     "max_cargo_per_episode": 52,
        #     "avg_working_capacity": 9.3136,
        #     "avg_processing_time": 6.0,
        #     "soft_deadline_multiplier": 5.2373,
        #     "hard_deadline_multiplier": 11.1746,
        #     "min_degree": 3,
        #     "max_degree": 15,
        #     "route_ratio": 2.0,
        #     "malfunction_rate": 0.0,
        #     "malfunction_min_steps": 0,
        #     "malfunction_max_steps": 0,
        #     "max_soft_deadline": 3519,
        #     "max_hard_deadline": 3469,
        #     "map_type": "plain",
        # }

        # test_cases[45] = {
        #     "number_of_agents": 20,
        #     "number_of_airports": 10,
        #     "number_of_initial_cargo": 53,
        #     "max_cargo_per_episode": 53,
        #     "avg_working_capacity": 9.5356,
        #     "avg_processing_time": 6.0237,
        #     "soft_deadline_multiplier": 5.1881,
        #     "hard_deadline_multiplier": 11.0881,
        #     "min_degree": 3,
        #     "max_degree": 15,
        #     "route_ratio": 2.0,
        #     "malfunction_rate": 0.0,
        #     "malfunction_min_steps": 0,
        #     "malfunction_max_steps": 0,
        #     "max_soft_deadline": 3494,
        #     "max_hard_deadline": 3444,
        #     "map_type": "plain",
        # }

        # test_cases[46] = {
        #     "number_of_agents": 21,
        #     "number_of_airports": 11,
        #     "number_of_initial_cargo": 55,
        #     "max_cargo_per_episode": 55,
        #     "avg_working_capacity": 9.6831,
        #     "avg_processing_time": 6.122,
        #     "soft_deadline_multiplier": 5.139,
        #     "hard_deadline_multiplier": 11.039,
        #     "min_degree": 3,
        #     "max_degree": 16,
        #     "route_ratio": 2.0,
        #     "malfunction_rate": 0.0,
        #     "malfunction_min_steps": 0,
        #     "malfunction_max_steps": 0,
        #     "max_soft_deadline": 3469,
        #     "max_hard_deadline": 3419,
        #     "map_type": "plain",
        # }

        # test_cases[47] = {
        #     "number_of_agents": 21,
        #     "number_of_airports": 11,
        #     "number_of_initial_cargo": 56,
        #     "max_cargo_per_episode": 56,
        #     "avg_working_capacity": 9.8203,
        #     "avg_processing_time": 6.2814,
        #     "soft_deadline_multiplier": 5.0949,
        #     "hard_deadline_multiplier": 10.9797,
        #     "min_degree": 3,
        #     "max_degree": 16,
        #     "route_ratio": 2.0,
        #     "malfunction_rate": 0.0,
        #     "malfunction_min_steps": 0,
        #     "malfunction_max_steps": 0,
        #     "max_soft_deadline": 3440,
        #     "max_hard_deadline": 3390,
        #     "map_type": "plain",
        # }

        # test_cases[48] = {
        #     "number_of_agents": 22,
        #     "number_of_airports": 11,
        #     "number_of_initial_cargo": 58,
        #     "max_cargo_per_episode": 58,
        #     "avg_working_capacity": 9.9186,
        #     "avg_processing_time": 6.6746,
        #     "soft_deadline_multiplier": 5.0703,
        #     "hard_deadline_multiplier": 10.8814,
        #     "min_degree": 3,
        #     "max_degree": 16,
        #     "route_ratio": 2.0,
        #     "malfunction_rate": 0.0,
        #     "malfunction_min_steps": 0,
        #     "malfunction_max_steps": 0,
        #     "max_soft_deadline": 3391,
        #     "max_hard_deadline": 3341,
        #     "map_type": "plain",
        # }

        # test_cases[49] = {
        #     "number_of_agents": 22,
        #     "number_of_airports": 11,
        #     "number_of_initial_cargo": 60,
        #     "max_cargo_per_episode": 60,
        #     "avg_working_capacity": 10.0424,
        #     "avg_processing_time": 7.0847,
        #     "soft_deadline_multiplier": 5.0483,
        #     "hard_deadline_multiplier": 10.7915,
        #     "min_degree": 3,
        #     "max_degree": 16,
        #     "route_ratio": 2.0,
        #     "malfunction_rate": 0.0,
        #     "malfunction_min_steps": 0,
        #     "malfunction_max_steps": 0,
        #     "max_soft_deadline": 3346,
        #     "max_hard_deadline": 3296,
        #     "map_type": "plain",
        # }

        # test_cases[50] = {
        #     "number_of_agents": 23,
        #     "number_of_airports": 11,
        #     "number_of_initial_cargo": 62,
        #     "max_cargo_per_episode": 62,
        #     "avg_working_capacity": 10.2881,
        #     "avg_processing_time": 7.5763,
        #     "soft_deadline_multiplier": 5.0385,
        #     "hard_deadline_multiplier": 10.7424,
        #     "min_degree": 3,
        #     "max_degree": 16,
        #     "route_ratio": 2.0,
        #     "malfunction_rate": 0.0,
        #     "malfunction_min_steps": 0,
        #     "malfunction_max_steps": 0,
        #     "max_soft_deadline": 3321,
        #     "max_hard_deadline": 3271,
        #     "map_type": "plain",
        # }

        # test_cases[51] = {
        #     "number_of_agents": 23,
        #     "number_of_airports": 12,
        #     "number_of_initial_cargo": 63,
        #     "max_cargo_per_episode": 63,
        #     "avg_working_capacity": 10.5339,
        #     "avg_processing_time": 8.0678,
        #     "soft_deadline_multiplier": 5.0293,
        #     "hard_deadline_multiplier": 10.6932,
        #     "min_degree": 3,
        #     "max_degree": 16,
        #     "route_ratio": 2.0,
        #     "malfunction_rate": 0.0,
        #     "malfunction_min_steps": 0,
        #     "malfunction_max_steps": 0,
        #     "max_soft_deadline": 3297,
        #     "max_hard_deadline": 3247,
        #     "map_type": "plain",
        # }

        # test_cases[52] = {
        #     "number_of_agents": 23,
        #     "number_of_airports": 12,
        #     "number_of_initial_cargo": 64,
        #     "max_cargo_per_episode": 64,
        #     "avg_working_capacity": 10.7797,
        #     "avg_processing_time": 8.5593,
        #     "soft_deadline_multiplier": 5.0244,
        #     "hard_deadline_multiplier": 10.6441,
        #     "min_degree": 3,
        #     "max_degree": 16,
        #     "route_ratio": 2.0,
        #     "malfunction_rate": 0.0,
        #     "malfunction_min_steps": 0,
        #     "malfunction_max_steps": 0,
        #     "max_soft_deadline": 3272,
        #     "max_hard_deadline": 3222,
        #     "map_type": "plain",
        # }

        # test_cases[53] = {
        #     "number_of_agents": 23,
        #     "number_of_airports": 12,
        #     "number_of_initial_cargo": 65,
        #     "max_cargo_per_episode": 65,
        #     "avg_working_capacity": 11.0203,
        #     "avg_processing_time": 9.0254,
        #     "soft_deadline_multiplier": 5.0195,
        #     "hard_deadline_multiplier": 10.5975,
        #     "min_degree": 3,
        #     "max_degree": 16,
        #     "route_ratio": 2.0,
        #     "malfunction_rate": 0.0,
        #     "malfunction_min_steps": 0,
        #     "malfunction_max_steps": 0,
        #     "max_soft_deadline": 3247,
        #     "max_hard_deadline": 3197,
        #     "map_type": "plain",
        # }

        # test_cases[54] = {
        #     "number_of_agents": 24,
        #     "number_of_airports": 12,
        #     "number_of_initial_cargo": 66,
        #     "max_cargo_per_episode": 66,
        #     "avg_working_capacity": 11.2169,
        #     "avg_processing_time": 9.2712,
        #     "soft_deadline_multiplier": 5.0146,
        #     "hard_deadline_multiplier": 10.5729,
        #     "min_degree": 3,
        #     "max_degree": 16,
        #     "route_ratio": 2.0,
        #     "malfunction_rate": 0.0,
        #     "malfunction_min_steps": 0,
        #     "malfunction_max_steps": 0,
        #     "max_soft_deadline": 3223,
        #     "max_hard_deadline": 3173,
        #     "map_type": "plain",
        # }

        # test_cases[55] = {
        #     "number_of_agents": 24,
        #     "number_of_airports": 12,
        #     "number_of_initial_cargo": 66,
        #     "max_cargo_per_episode": 66,
        #     "avg_working_capacity": 11.409,
        #     "avg_processing_time": 9.5169,
        #     "soft_deadline_multiplier": 5.0097,
        #     "hard_deadline_multiplier": 10.3958,
        #     "min_degree": 3,
        #     "max_degree": 16,
        #     "route_ratio": 2.0,
        #     "malfunction_rate": 0.0,
        #     "malfunction_min_steps": 0,
        #     "malfunction_max_steps": 0,
        #     "max_soft_deadline": 3187,
        #     "max_hard_deadline": 3142,
        #     "map_type": "plain",
        # }

        # test_cases[56] = {
        #     "number_of_agents": 24,
        #     "number_of_airports": 12,
        #     "number_of_initial_cargo": 67,
        #     "max_cargo_per_episode": 67,
        #     "avg_working_capacity": 11.5401,
        #     "avg_processing_time": 9.7627,
        #     "soft_deadline_multiplier": 5.0047,
        #     "hard_deadline_multiplier": 8.1593,
        #     "min_degree": 2,
        #     "max_degree": 16,
        #     "route_ratio": 2.0,
        #     "malfunction_rate": 0.0,
        #     "malfunction_min_steps": 0,
        #     "malfunction_max_steps": 0,
        #     "max_soft_deadline": 2999,
        #     "max_hard_deadline": 3025,
        #     "map_type": "plain",
        # }

        # test_cases[57] = {
        #     "number_of_agents": 24,
        #     "number_of_airports": 12,
        #     "number_of_initial_cargo": 67,
        #     "max_cargo_per_episode": 67,
        #     "avg_working_capacity": 11.6737,
        #     "avg_processing_time": 10.0,
        #     "soft_deadline_multiplier": 5,
        #     "hard_deadline_multiplier": 6,
        #     "min_degree": 2,
        #     "max_degree": 16,
        #     "route_ratio": 2.0,
        #     "malfunction_rate": 0.0,
        #     "malfunction_min_steps": 0,
        #     "malfunction_max_steps": 0,
        #     "max_soft_deadline": 2817,
        #     "max_hard_deadline": 2911,
        #     "map_type": "plain",
        # }

        # test_cases[58] = {
        #     "number_of_agents": 24,
        #     "number_of_airports": 12,
        #     "number_of_initial_cargo": 70,
        #     "max_cargo_per_episode": 70,
        #     "avg_working_capacity": 11.8785,
        #     "avg_processing_time": 10.0,
        #     "soft_deadline_multiplier": 5,
        #     "hard_deadline_multiplier": 6,
        #     "min_degree": 2,
        #     "max_degree": 15,
        #     "route_ratio": 2.0,
        #     "malfunction_rate": 0.0,
        #     "malfunction_min_steps": 0,
        #     "malfunction_max_steps": 0,
        #     "max_soft_deadline": 2792,
        #     "max_hard_deadline": 2877,
        #     "map_type": "plain",
        # }

        # test_cases[59] = {
        #     "number_of_agents": 24,
        #     "number_of_airports": 12,
        #     "number_of_initial_cargo": 72,
        #     "max_cargo_per_episode": 72,
        #     "avg_working_capacity": 12.0833,
        #     "avg_processing_time": 10.0,
        #     "soft_deadline_multiplier": 5,
        #     "hard_deadline_multiplier": 6,
        #     "min_degree": 2,
        #     "max_degree": 14,
        #     "route_ratio": 2.0,
        #     "malfunction_rate": 0.0,
        #     "malfunction_min_steps": 0,
        #     "malfunction_max_steps": 0,
        #     "max_soft_deadline": 2767,
        #     "max_hard_deadline": 2843,
        #     "map_type": "plain",
        # }

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
            "route_ratio": 3,
            "malfunction_rate": 0,
            "malfunction_min_steps": 0,
            "malfunction_max_steps": 0,
            "max_soft_deadline": 5000,
            "max_hard_deadline": 5000,
            "map_type": "plain",
        }

        test_cases[1] = {
            "number_of_agents": 1,
            "number_of_airports": 3,
            "number_of_initial_cargo": 2,
            "max_cargo_per_episode": 2,
            "avg_working_capacity": 2.0169,
            "avg_processing_time": 2.1356,
            "soft_deadline_multiplier": 9.9153,
            "hard_deadline_multiplier": 19.7627,
            "min_degree": 2,
            "max_degree": 3,
            "route_ratio": 2.9831,
            "malfunction_rate": 0,
            "malfunction_min_steps": 0,
            "malfunction_max_steps": 0,
            "max_soft_deadline": 4962,
            "max_hard_deadline": 4963,
            "map_type": "plain",
        }

        test_cases[2] = {
            "number_of_agents": 1,
            "number_of_airports": 3,
            "number_of_initial_cargo": 2,
            "max_cargo_per_episode": 2,
            "avg_working_capacity": 2.0339,
            "avg_processing_time": 2.2712,
            "soft_deadline_multiplier": 9.8305,
            "hard_deadline_multiplier": 19.5254,
            "min_degree": 2,
            "max_degree": 3,
            "route_ratio": 2.9661,
            "malfunction_rate": 0,
            "malfunction_min_steps": 0,
            "malfunction_max_steps": 0,
            "max_soft_deadline": 4924,
            "max_hard_deadline": 4927,
            "map_type": "plain",
        }

        test_cases[3] = {
            "number_of_agents": 1,
            "number_of_airports": 3,
            "number_of_initial_cargo": 2,
            "max_cargo_per_episode": 3,
            "avg_working_capacity": 2.0508,
            "avg_processing_time": 2.4068,
            "soft_deadline_multiplier": 9.7458,
            "hard_deadline_multiplier": 19.2881,
            "min_degree": 2,
            "max_degree": 3,
            "route_ratio": 2.9492,
            "malfunction_rate": 0,
            "malfunction_min_steps": 0,
            "malfunction_max_steps": 0,
            "max_soft_deadline": 4886,
            "max_hard_deadline": 4890,
            "map_type": "plain",
        }

        test_cases[4] = {
            "number_of_agents": 1,
            "number_of_airports": 3,
            "number_of_initial_cargo": 2,
            "max_cargo_per_episode": 3,
            "avg_working_capacity": 2.0678,
            "avg_processing_time": 2.5424,
            "soft_deadline_multiplier": 9.661,
            "hard_deadline_multiplier": 19.0508,
            "min_degree": 2,
            "max_degree": 3,
            "route_ratio": 2.9322,
            "malfunction_rate": 0,
            "malfunction_min_steps": 0,
            "malfunction_max_steps": 0,
            "max_soft_deadline": 4849,
            "max_hard_deadline": 4854,
            "map_type": "plain",
        }

        test_cases[5] = {
            "number_of_agents": 1,
            "number_of_airports": 3,
            "number_of_initial_cargo": 3,
            "max_cargo_per_episode": 3,
            "avg_working_capacity": 2.0847,
            "avg_processing_time": 2.678,
            "soft_deadline_multiplier": 9.5763,
            "hard_deadline_multiplier": 18.8136,
            "min_degree": 2,
            "max_degree": 3,
            "route_ratio": 2.9153,
            "malfunction_rate": 0,
            "malfunction_min_steps": 0,
            "malfunction_max_steps": 0,
            "max_soft_deadline": 4811,
            "max_hard_deadline": 4817,
            "map_type": "plain",
        }

        test_cases[6] = {
            "number_of_agents": 2,
            "number_of_airports": 3,
            "number_of_initial_cargo": 3,
            "max_cargo_per_episode": 3,
            "avg_working_capacity": 2.1017,
            "avg_processing_time": 2.8136,
            "soft_deadline_multiplier": 9.4915,
            "hard_deadline_multiplier": 18.5763,
            "min_degree": 2,
            "max_degree": 3,
            "route_ratio": 2.8983,
            "malfunction_rate": 0,
            "malfunction_min_steps": 0,
            "malfunction_max_steps": 0,
            "max_soft_deadline": 4773,
            "max_hard_deadline": 4781,
            "map_type": "plain",
        }

        test_cases[7] = {
            "number_of_agents": 2,
            "number_of_airports": 3,
            "number_of_initial_cargo": 3,
            "max_cargo_per_episode": 4,
            "avg_working_capacity": 2.1186,
            "avg_processing_time": 2.9492,
            "soft_deadline_multiplier": 9.4068,
            "hard_deadline_multiplier": 18.339,
            "min_degree": 2,
            "max_degree": 3,
            "route_ratio": 2.8814,
            "malfunction_rate": 0,
            "malfunction_min_steps": 0,
            "malfunction_max_steps": 0,
            "max_soft_deadline": 4735,
            "max_hard_deadline": 4744,
            "map_type": "plain",
        }

        test_cases[8] = {
            "number_of_agents": 2,
            "number_of_airports": 3,
            "number_of_initial_cargo": 3,
            "max_cargo_per_episode": 4,
            "avg_working_capacity": 2.1356,
            "avg_processing_time": 3.0847,
            "soft_deadline_multiplier": 9.322,
            "hard_deadline_multiplier": 18.1017,
            "min_degree": 2,
            "max_degree": 3,
            "route_ratio": 2.8644,
            "malfunction_rate": 0,
            "malfunction_min_steps": 0,
            "malfunction_max_steps": 0,
            "max_soft_deadline": 4697,
            "max_hard_deadline": 4708,
            "map_type": "plain",
        }

        test_cases[9] = {
            "number_of_agents": 2,
            "number_of_airports": 3,
            "number_of_initial_cargo": 3,
            "max_cargo_per_episode": 4,
            "avg_working_capacity": 2.1525,
            "avg_processing_time": 3.2203,
            "soft_deadline_multiplier": 9.2373,
            "hard_deadline_multiplier": 17.8644,
            "min_degree": 2,
            "max_degree": 3,
            "route_ratio": 2.8475,
            "malfunction_rate": 0,
            "malfunction_min_steps": 0,
            "malfunction_max_steps": 0,
            "max_soft_deadline": 4659,
            "max_hard_deadline": 4671,
            "map_type": "plain",
        }

        test_cases[10] = {
            "number_of_agents": 2,
            "number_of_airports": 4,
            "number_of_initial_cargo": 3,
            "max_cargo_per_episode": 4,
            "avg_working_capacity": 2.1695,
            "avg_processing_time": 3.3559,
            "soft_deadline_multiplier": 9.1525,
            "hard_deadline_multiplier": 17.6271,
            "min_degree": 2,
            "max_degree": 3,
            "route_ratio": 2.8305,
            "malfunction_rate": 0,
            "malfunction_min_steps": 0,
            "malfunction_max_steps": 0,
            "max_soft_deadline": 4622,
            "max_hard_deadline": 4634,
            "map_type": "plain",
        }

        test_cases[11] = {
            "number_of_agents": 2,
            "number_of_airports": 4,
            "number_of_initial_cargo": 3,
            "max_cargo_per_episode": 4,
            "avg_working_capacity": 2.1864,
            "avg_processing_time": 3.4915,
            "soft_deadline_multiplier": 9.0678,
            "hard_deadline_multiplier": 17.3898,
            "min_degree": 2,
            "max_degree": 3,
            "route_ratio": 2.8136,
            "malfunction_rate": 0,
            "malfunction_min_steps": 0,
            "malfunction_max_steps": 0,
            "max_soft_deadline": 4584,
            "max_hard_deadline": 4598,
            "map_type": "plain",
        }

        test_cases[12] = {
            "number_of_agents": 2,
            "number_of_airports": 4,
            "number_of_initial_cargo": 3,
            "max_cargo_per_episode": 5,
            "avg_working_capacity": 2.2034,
            "avg_processing_time": 3.6271,
            "soft_deadline_multiplier": 8.9831,
            "hard_deadline_multiplier": 17.1525,
            "min_degree": 2,
            "max_degree": 3,
            "route_ratio": 2.7966,
            "malfunction_rate": 0,
            "malfunction_min_steps": 0,
            "malfunction_max_steps": 0,
            "max_soft_deadline": 4546,
            "max_hard_deadline": 4561,
            "map_type": "plain",
        }

        test_cases[13] = {
            "number_of_agents": 2,
            "number_of_airports": 4,
            "number_of_initial_cargo": 3,
            "max_cargo_per_episode": 5,
            "avg_working_capacity": 2.2203,
            "avg_processing_time": 3.7627,
            "soft_deadline_multiplier": 8.8983,
            "hard_deadline_multiplier": 16.9153,
            "min_degree": 2,
            "max_degree": 3,
            "route_ratio": 2.7797,
            "malfunction_rate": 0,
            "malfunction_min_steps": 0,
            "malfunction_max_steps": 0,
            "max_soft_deadline": 4508,
            "max_hard_deadline": 4525,
            "map_type": "plain",
        }

        test_cases[14] = {
            "number_of_agents": 2,
            "number_of_airports": 4,
            "number_of_initial_cargo": 3,
            "max_cargo_per_episode": 5,
            "avg_working_capacity": 2.2373,
            "avg_processing_time": 3.8983,
            "soft_deadline_multiplier": 8.8136,
            "hard_deadline_multiplier": 16.678,
            "min_degree": 2,
            "max_degree": 3,
            "route_ratio": 2.7627,
            "malfunction_rate": 0,
            "malfunction_min_steps": 0,
            "malfunction_max_steps": 0,
            "max_soft_deadline": 4470,
            "max_hard_deadline": 4488,
            "map_type": "plain",
        }

        test_cases[15] = {
            "number_of_agents": 2,
            "number_of_airports": 4,
            "number_of_initial_cargo": 4,
            "max_cargo_per_episode": 5,
            "avg_working_capacity": 2.2542,
            "avg_processing_time": 4.0339,
            "soft_deadline_multiplier": 8.7288,
            "hard_deadline_multiplier": 16.4407,
            "min_degree": 2,
            "max_degree": 3,
            "route_ratio": 2.7458,
            "malfunction_rate": 0,
            "malfunction_min_steps": 0,
            "malfunction_max_steps": 0,
            "max_soft_deadline": 4432,
            "max_hard_deadline": 4452,
            "map_type": "plain",
        }

        test_cases[16] = {
            "number_of_agents": 2,
            "number_of_airports": 4,
            "number_of_initial_cargo": 4,
            "max_cargo_per_episode": 6,
            "avg_working_capacity": 2.2712,
            "avg_processing_time": 4.1695,
            "soft_deadline_multiplier": 8.6441,
            "hard_deadline_multiplier": 16.2034,
            "min_degree": 2,
            "max_degree": 3,
            "route_ratio": 2.7288,
            "malfunction_rate": 0,
            "malfunction_min_steps": 0,
            "malfunction_max_steps": 0,
            "max_soft_deadline": 4394,
            "max_hard_deadline": 4415,
            "map_type": "plain",
        }

        test_cases[17] = {
            "number_of_agents": 2,
            "number_of_airports": 4,
            "number_of_initial_cargo": 4,
            "max_cargo_per_episode": 6,
            "avg_working_capacity": 2.2881,
            "avg_processing_time": 4.3051,
            "soft_deadline_multiplier": 8.5593,
            "hard_deadline_multiplier": 15.9661,
            "min_degree": 2,
            "max_degree": 3,
            "route_ratio": 2.7119,
            "malfunction_rate": 0,
            "malfunction_min_steps": 0,
            "malfunction_max_steps": 0,
            "max_soft_deadline": 4357,
            "max_hard_deadline": 4378,
            "map_type": "plain",
        }

        test_cases[18] = {
            "number_of_agents": 3,
            "number_of_airports": 4,
            "number_of_initial_cargo": 4,
            "max_cargo_per_episode": 6,
            "avg_working_capacity": 2.3051,
            "avg_processing_time": 4.4407,
            "soft_deadline_multiplier": 8.4746,
            "hard_deadline_multiplier": 15.7288,
            "min_degree": 2,
            "max_degree": 3,
            "route_ratio": 2.6949,
            "malfunction_rate": 0,
            "malfunction_min_steps": 0,
            "malfunction_max_steps": 0,
            "max_soft_deadline": 4319,
            "max_hard_deadline": 4342,
            "map_type": "plain",
        }

        test_cases[19] = {
            "number_of_agents": 3,
            "number_of_airports": 4,
            "number_of_initial_cargo": 4,
            "max_cargo_per_episode": 6,
            "avg_working_capacity": 2.322,
            "avg_processing_time": 4.5763,
            "soft_deadline_multiplier": 8.3898,
            "hard_deadline_multiplier": 15.4915,
            "min_degree": 2,
            "max_degree": 3,
            "route_ratio": 2.678,
            "malfunction_rate": 0,
            "malfunction_min_steps": 0,
            "malfunction_max_steps": 0,
            "max_soft_deadline": 4281,
            "max_hard_deadline": 4305,
            "map_type": "plain",
        }

        test_cases[20] = {
            "number_of_agents": 3,
            "number_of_airports": 4,
            "number_of_initial_cargo": 4,
            "max_cargo_per_episode": 6,
            "avg_working_capacity": 2.339,
            "avg_processing_time": 4.7119,
            "soft_deadline_multiplier": 8.3051,
            "hard_deadline_multiplier": 15.2542,
            "min_degree": 2,
            "max_degree": 3,
            "route_ratio": 2.661,
            "malfunction_rate": 0,
            "malfunction_min_steps": 0,
            "malfunction_max_steps": 0,
            "max_soft_deadline": 4243,
            "max_hard_deadline": 4269,
            "map_type": "plain",
        }

        test_cases[21] = {
            "number_of_agents": 3,
            "number_of_airports": 4,
            "number_of_initial_cargo": 4,
            "max_cargo_per_episode": 7,
            "avg_working_capacity": 2.3559,
            "avg_processing_time": 4.8475,
            "soft_deadline_multiplier": 8.2203,
            "hard_deadline_multiplier": 15.0169,
            "min_degree": 2,
            "max_degree": 3,
            "route_ratio": 2.6441,
            "malfunction_rate": 0,
            "malfunction_min_steps": 0,
            "malfunction_max_steps": 0,
            "max_soft_deadline": 4205,
            "max_hard_deadline": 4232,
            "map_type": "plain",
        }

        test_cases[22] = {
            "number_of_agents": 3,
            "number_of_airports": 4,
            "number_of_initial_cargo": 4,
            "max_cargo_per_episode": 7,
            "avg_working_capacity": 2.3729,
            "avg_processing_time": 4.9831,
            "soft_deadline_multiplier": 8.1356,
            "hard_deadline_multiplier": 14.7797,
            "min_degree": 2,
            "max_degree": 3,
            "route_ratio": 2.6271,
            "malfunction_rate": 0,
            "malfunction_min_steps": 0,
            "malfunction_max_steps": 0,
            "max_soft_deadline": 4167,
            "max_hard_deadline": 4196,
            "map_type": "plain",
        }

        test_cases[23] = {
            "number_of_agents": 3,
            "number_of_airports": 4,
            "number_of_initial_cargo": 4,
            "max_cargo_per_episode": 7,
            "avg_working_capacity": 2.3898,
            "avg_processing_time": 5.1186,
            "soft_deadline_multiplier": 8.0508,
            "hard_deadline_multiplier": 14.5424,
            "min_degree": 2,
            "max_degree": 3,
            "route_ratio": 2.6102,
            "malfunction_rate": 0,
            "malfunction_min_steps": 0,
            "malfunction_max_steps": 0,
            "max_soft_deadline": 4130,
            "max_hard_deadline": 4159,
            "map_type": "plain",
        }

        test_cases[24] = {
            "number_of_agents": 3,
            "number_of_airports": 4,
            "number_of_initial_cargo": 4,
            "max_cargo_per_episode": 7,
            "avg_working_capacity": 2.4068,
            "avg_processing_time": 5.2542,
            "soft_deadline_multiplier": 7.9661,
            "hard_deadline_multiplier": 14.3051,
            "min_degree": 2,
            "max_degree": 3,
            "route_ratio": 2.5932,
            "malfunction_rate": 0,
            "malfunction_min_steps": 0,
            "malfunction_max_steps": 0,
            "max_soft_deadline": 4092,
            "max_hard_deadline": 4123,
            "map_type": "plain",
        }

        test_cases[25] = {
            "number_of_agents": 3,
            "number_of_airports": 4,
            "number_of_initial_cargo": 5,
            "max_cargo_per_episode": 8,
            "avg_working_capacity": 2.4237,
            "avg_processing_time": 5.3898,
            "soft_deadline_multiplier": 7.8814,
            "hard_deadline_multiplier": 14.0678,
            "min_degree": 2,
            "max_degree": 3,
            "route_ratio": 2.5763,
            "malfunction_rate": 0,
            "malfunction_min_steps": 0,
            "malfunction_max_steps": 0,
            "max_soft_deadline": 4054,
            "max_hard_deadline": 4086,
            "map_type": "plain",
        }

        test_cases[26] = {
            "number_of_agents": 3,
            "number_of_airports": 4,
            "number_of_initial_cargo": 5,
            "max_cargo_per_episode": 8,
            "avg_working_capacity": 2.4407,
            "avg_processing_time": 5.5254,
            "soft_deadline_multiplier": 7.7966,
            "hard_deadline_multiplier": 13.8305,
            "min_degree": 2,
            "max_degree": 3,
            "route_ratio": 2.5593,
            "malfunction_rate": 0,
            "malfunction_min_steps": 0,
            "malfunction_max_steps": 0,
            "max_soft_deadline": 4016,
            "max_hard_deadline": 4049,
            "map_type": "plain",
        }

        test_cases[27] = {
            "number_of_agents": 3,
            "number_of_airports": 4,
            "number_of_initial_cargo": 5,
            "max_cargo_per_episode": 8,
            "avg_working_capacity": 2.4576,
            "avg_processing_time": 5.661,
            "soft_deadline_multiplier": 7.7119,
            "hard_deadline_multiplier": 13.5932,
            "min_degree": 2,
            "max_degree": 3,
            "route_ratio": 2.5424,
            "malfunction_rate": 0,
            "malfunction_min_steps": 0,
            "malfunction_max_steps": 0,
            "max_soft_deadline": 3978,
            "max_hard_deadline": 4013,
            "map_type": "plain",
        }

        test_cases[28] = {
            "number_of_agents": 3,
            "number_of_airports": 4,
            "number_of_initial_cargo": 5,
            "max_cargo_per_episode": 8,
            "avg_working_capacity": 2.4746,
            "avg_processing_time": 5.7966,
            "soft_deadline_multiplier": 7.6271,
            "hard_deadline_multiplier": 13.3559,
            "min_degree": 2,
            "max_degree": 3,
            "route_ratio": 2.5254,
            "malfunction_rate": 0,
            "malfunction_min_steps": 0,
            "malfunction_max_steps": 0,
            "max_soft_deadline": 3940,
            "max_hard_deadline": 3976,
            "map_type": "plain",
        }

        test_cases[29] = {
            "number_of_agents": 3,
            "number_of_airports": 4,
            "number_of_initial_cargo": 5,
            "max_cargo_per_episode": 8,
            "avg_working_capacity": 2.4915,
            "avg_processing_time": 5.9322,
            "soft_deadline_multiplier": 7.5424,
            "hard_deadline_multiplier": 13.1186,
            "min_degree": 2,
            "max_degree": 3,
            "route_ratio": 2.5085,
            "malfunction_rate": 0,
            "malfunction_min_steps": 0,
            "malfunction_max_steps": 0,
            "max_soft_deadline": 3902,
            "max_hard_deadline": 3940,
            "map_type": "plain",
        }

        test_cases[30] = {
            "number_of_agents": 4,
            "number_of_airports": 5,
            "number_of_initial_cargo": 5,
            "max_cargo_per_episode": 9,
            "avg_working_capacity": 2.5085,
            "avg_processing_time": 6.0678,
            "soft_deadline_multiplier": 7.4576,
            "hard_deadline_multiplier": 12.8814,
            "min_degree": 2,
            "max_degree": 3,
            "route_ratio": 2.4915,
            "malfunction_rate": 0,
            "malfunction_min_steps": 0,
            "malfunction_max_steps": 0,
            "max_soft_deadline": 3865,
            "max_hard_deadline": 3903,
            "map_type": "plain",
        }

        test_cases[31] = {
            "number_of_agents": 4,
            "number_of_airports": 5,
            "number_of_initial_cargo": 5,
            "max_cargo_per_episode": 9,
            "avg_working_capacity": 2.5254,
            "avg_processing_time": 6.2034,
            "soft_deadline_multiplier": 7.3729,
            "hard_deadline_multiplier": 12.6441,
            "min_degree": 2,
            "max_degree": 3,
            "route_ratio": 2.4746,
            "malfunction_rate": 0,
            "malfunction_min_steps": 0,
            "malfunction_max_steps": 0,
            "max_soft_deadline": 3827,
            "max_hard_deadline": 3867,
            "map_type": "plain",
        }

        test_cases[32] = {
            "number_of_agents": 4,
            "number_of_airports": 5,
            "number_of_initial_cargo": 5,
            "max_cargo_per_episode": 9,
            "avg_working_capacity": 2.5424,
            "avg_processing_time": 6.339,
            "soft_deadline_multiplier": 7.2881,
            "hard_deadline_multiplier": 12.4068,
            "min_degree": 2,
            "max_degree": 3,
            "route_ratio": 2.4576,
            "malfunction_rate": 0,
            "malfunction_min_steps": 0,
            "malfunction_max_steps": 0,
            "max_soft_deadline": 3789,
            "max_hard_deadline": 3830,
            "map_type": "plain",
        }

        test_cases[33] = {
            "number_of_agents": 4,
            "number_of_airports": 5,
            "number_of_initial_cargo": 5,
            "max_cargo_per_episode": 9,
            "avg_working_capacity": 2.5593,
            "avg_processing_time": 6.4746,
            "soft_deadline_multiplier": 7.2034,
            "hard_deadline_multiplier": 12.1695,
            "min_degree": 2,
            "max_degree": 3,
            "route_ratio": 2.4407,
            "malfunction_rate": 0,
            "malfunction_min_steps": 0,
            "malfunction_max_steps": 0,
            "max_soft_deadline": 3751,
            "max_hard_deadline": 3794,
            "map_type": "plain",
        }

        test_cases[34] = {
            "number_of_agents": 4,
            "number_of_airports": 5,
            "number_of_initial_cargo": 5,
            "max_cargo_per_episode": 9,
            "avg_working_capacity": 2.5763,
            "avg_processing_time": 6.6102,
            "soft_deadline_multiplier": 7.1186,
            "hard_deadline_multiplier": 11.9322,
            "min_degree": 2,
            "max_degree": 3,
            "route_ratio": 2.4237,
            "malfunction_rate": 0,
            "malfunction_min_steps": 0,
            "malfunction_max_steps": 0,
            "max_soft_deadline": 3713,
            "max_hard_deadline": 3757,
            "map_type": "plain",
        }

        test_cases[35] = {
            "number_of_agents": 4,
            "number_of_airports": 5,
            "number_of_initial_cargo": 6,
            "max_cargo_per_episode": 10,
            "avg_working_capacity": 2.5932,
            "avg_processing_time": 6.7458,
            "soft_deadline_multiplier": 7.0339,
            "hard_deadline_multiplier": 11.6949,
            "min_degree": 2,
            "max_degree": 3,
            "route_ratio": 2.4068,
            "malfunction_rate": 0,
            "malfunction_min_steps": 0,
            "malfunction_max_steps": 0,
            "max_soft_deadline": 3675,
            "max_hard_deadline": 3720,
            "map_type": "plain",
        }

        test_cases[36] = {
            "number_of_agents": 4,
            "number_of_airports": 5,
            "number_of_initial_cargo": 6,
            "max_cargo_per_episode": 10,
            "avg_working_capacity": 2.6102,
            "avg_processing_time": 6.8814,
            "soft_deadline_multiplier": 6.9492,
            "hard_deadline_multiplier": 11.4576,
            "min_degree": 2,
            "max_degree": 3,
            "route_ratio": 2.3898,
            "malfunction_rate": 0,
            "malfunction_min_steps": 0,
            "malfunction_max_steps": 0,
            "max_soft_deadline": 3637,
            "max_hard_deadline": 3684,
            "map_type": "plain",
        }

        test_cases[37] = {
            "number_of_agents": 4,
            "number_of_airports": 5,
            "number_of_initial_cargo": 6,
            "max_cargo_per_episode": 10,
            "avg_working_capacity": 2.6271,
            "avg_processing_time": 7.0169,
            "soft_deadline_multiplier": 6.8644,
            "hard_deadline_multiplier": 11.2203,
            "min_degree": 2,
            "max_degree": 3,
            "route_ratio": 2.3729,
            "malfunction_rate": 0,
            "malfunction_min_steps": 0,
            "malfunction_max_steps": 0,
            "max_soft_deadline": 3600,
            "max_hard_deadline": 3647,
            "map_type": "plain",
        }

        test_cases[38] = {
            "number_of_agents": 4,
            "number_of_airports": 5,
            "number_of_initial_cargo": 6,
            "max_cargo_per_episode": 10,
            "avg_working_capacity": 2.6441,
            "avg_processing_time": 7.1525,
            "soft_deadline_multiplier": 6.7797,
            "hard_deadline_multiplier": 10.9831,
            "min_degree": 2,
            "max_degree": 3,
            "route_ratio": 2.3559,
            "malfunction_rate": 0,
            "malfunction_min_steps": 0,
            "malfunction_max_steps": 0,
            "max_soft_deadline": 3562,
            "max_hard_deadline": 3611,
            "map_type": "plain",
        }

        test_cases[39] = {
            "number_of_agents": 4,
            "number_of_airports": 5,
            "number_of_initial_cargo": 6,
            "max_cargo_per_episode": 11,
            "avg_working_capacity": 2.661,
            "avg_processing_time": 7.2881,
            "soft_deadline_multiplier": 6.6949,
            "hard_deadline_multiplier": 10.7458,
            "min_degree": 2,
            "max_degree": 3,
            "route_ratio": 2.339,
            "malfunction_rate": 0,
            "malfunction_min_steps": 0,
            "malfunction_max_steps": 0,
            "max_soft_deadline": 3524,
            "max_hard_deadline": 3574,
            "map_type": "plain",
        }

        test_cases[40] = {
            "number_of_agents": 4,
            "number_of_airports": 5,
            "number_of_initial_cargo": 6,
            "max_cargo_per_episode": 11,
            "avg_working_capacity": 2.678,
            "avg_processing_time": 7.4237,
            "soft_deadline_multiplier": 6.6102,
            "hard_deadline_multiplier": 10.5085,
            "min_degree": 2,
            "max_degree": 3,
            "route_ratio": 2.322,
            "malfunction_rate": 0,
            "malfunction_min_steps": 0,
            "malfunction_max_steps": 0,
            "max_soft_deadline": 3486,
            "max_hard_deadline": 3538,
            "map_type": "plain",
        }

        test_cases[41] = {
            "number_of_agents": 4,
            "number_of_airports": 5,
            "number_of_initial_cargo": 6,
            "max_cargo_per_episode": 11,
            "avg_working_capacity": 2.6949,
            "avg_processing_time": 7.5593,
            "soft_deadline_multiplier": 6.5254,
            "hard_deadline_multiplier": 10.2712,
            "min_degree": 2,
            "max_degree": 3,
            "route_ratio": 2.3051,
            "malfunction_rate": 0.0079,
            "malfunction_min_steps": 0,
            "malfunction_max_steps": 1,
            "max_soft_deadline": 3448,
            "max_hard_deadline": 3501,
            "map_type": "plain",
        }

        test_cases[42] = {
            "number_of_agents": 5,
            "number_of_airports": 5,
            "number_of_initial_cargo": 6,
            "max_cargo_per_episode": 11,
            "avg_working_capacity": 2.7119,
            "avg_processing_time": 7.6949,
            "soft_deadline_multiplier": 6.4407,
            "hard_deadline_multiplier": 10.0339,
            "min_degree": 2,
            "max_degree": 3,
            "route_ratio": 2.2881,
            "malfunction_rate": 0.0158,
            "malfunction_min_steps": 1,
            "malfunction_max_steps": 3,
            "max_soft_deadline": 3410,
            "max_hard_deadline": 3465,
            "map_type": "plain",
        }

        test_cases[43] = {
            "number_of_agents": 5,
            "number_of_airports": 5,
            "number_of_initial_cargo": 6,
            "max_cargo_per_episode": 11,
            "avg_working_capacity": 2.7288,
            "avg_processing_time": 7.8305,
            "soft_deadline_multiplier": 6.3559,
            "hard_deadline_multiplier": 9.7966,
            "min_degree": 2,
            "max_degree": 3,
            "route_ratio": 2.2712,
            "malfunction_rate": 0.0237,
            "malfunction_min_steps": 1,
            "malfunction_max_steps": 4,
            "max_soft_deadline": 3373,
            "max_hard_deadline": 3428,
            "map_type": "plain",
        }

        test_cases[44] = {
            "number_of_agents": 5,
            "number_of_airports": 5,
            "number_of_initial_cargo": 6,
            "max_cargo_per_episode": 12,
            "avg_working_capacity": 2.7458,
            "avg_processing_time": 7.9661,
            "soft_deadline_multiplier": 6.2712,
            "hard_deadline_multiplier": 9.5593,
            "min_degree": 2,
            "max_degree": 3,
            "route_ratio": 2.2542,
            "malfunction_rate": 0.0316,
            "malfunction_min_steps": 1,
            "malfunction_max_steps": 6,
            "max_soft_deadline": 3335,
            "max_hard_deadline": 3391,
            "map_type": "plain",
        }

        test_cases[45] = {
            "number_of_agents": 5,
            "number_of_airports": 5,
            "number_of_initial_cargo": 7,
            "max_cargo_per_episode": 12,
            "avg_working_capacity": 2.7627,
            "avg_processing_time": 8.1017,
            "soft_deadline_multiplier": 6.1864,
            "hard_deadline_multiplier": 9.322,
            "min_degree": 2,
            "max_degree": 3,
            "route_ratio": 2.2373,
            "malfunction_rate": 0.0395,
            "malfunction_min_steps": 2,
            "malfunction_max_steps": 7,
            "max_soft_deadline": 3297,
            "max_hard_deadline": 3355,
            "map_type": "plain",
        }

        test_cases[46] = {
            "number_of_agents": 5,
            "number_of_airports": 5,
            "number_of_initial_cargo": 7,
            "max_cargo_per_episode": 12,
            "avg_working_capacity": 2.7797,
            "avg_processing_time": 8.2373,
            "soft_deadline_multiplier": 6.1017,
            "hard_deadline_multiplier": 9.0847,
            "min_degree": 2,
            "max_degree": 3,
            "route_ratio": 2.2203,
            "malfunction_rate": 0.0474,
            "malfunction_min_steps": 2,
            "malfunction_max_steps": 9,
            "max_soft_deadline": 3259,
            "max_hard_deadline": 3318,
            "map_type": "plain",
        }

        test_cases[47] = {
            "number_of_agents": 5,
            "number_of_airports": 5,
            "number_of_initial_cargo": 7,
            "max_cargo_per_episode": 12,
            "avg_working_capacity": 2.7966,
            "avg_processing_time": 8.3729,
            "soft_deadline_multiplier": 6.0169,
            "hard_deadline_multiplier": 8.8475,
            "min_degree": 2,
            "max_degree": 3,
            "route_ratio": 2.2034,
            "malfunction_rate": 0.0553,
            "malfunction_min_steps": 3,
            "malfunction_max_steps": 10,
            "max_soft_deadline": 3221,
            "max_hard_deadline": 3282,
            "map_type": "plain",
        }

        test_cases[48] = {
            "number_of_agents": 5,
            "number_of_airports": 5,
            "number_of_initial_cargo": 7,
            "max_cargo_per_episode": 13,
            "avg_working_capacity": 2.8136,
            "avg_processing_time": 8.5085,
            "soft_deadline_multiplier": 5.9322,
            "hard_deadline_multiplier": 8.6102,
            "min_degree": 2,
            "max_degree": 3,
            "route_ratio": 2.1864,
            "malfunction_rate": 0.0632,
            "malfunction_min_steps": 3,
            "malfunction_max_steps": 12,
            "max_soft_deadline": 3183,
            "max_hard_deadline": 3245,
            "map_type": "plain",
        }

        test_cases[49] = {
            "number_of_agents": 5,
            "number_of_airports": 5,
            "number_of_initial_cargo": 7,
            "max_cargo_per_episode": 13,
            "avg_working_capacity": 2.8305,
            "avg_processing_time": 8.6441,
            "soft_deadline_multiplier": 5.8475,
            "hard_deadline_multiplier": 8.3729,
            "min_degree": 2,
            "max_degree": 3,
            "route_ratio": 2.1695,
            "malfunction_rate": 0.0711,
            "malfunction_min_steps": 3,
            "malfunction_max_steps": 13,
            "max_soft_deadline": 3145,
            "max_hard_deadline": 3209,
            "map_type": "plain",
        }

        test_cases[50] = {
            "number_of_agents": 5,
            "number_of_airports": 6,
            "number_of_initial_cargo": 7,
            "max_cargo_per_episode": 13,
            "avg_working_capacity": 2.8475,
            "avg_processing_time": 8.7797,
            "soft_deadline_multiplier": 5.7627,
            "hard_deadline_multiplier": 8.1356,
            "min_degree": 2,
            "max_degree": 3,
            "route_ratio": 2.1525,
            "malfunction_rate": 0.0789,
            "malfunction_min_steps": 4,
            "malfunction_max_steps": 15,
            "max_soft_deadline": 3108,
            "max_hard_deadline": 3172,
            "map_type": "plain",
        }

        test_cases[51] = {
            "number_of_agents": 5,
            "number_of_airports": 6,
            "number_of_initial_cargo": 7,
            "max_cargo_per_episode": 13,
            "avg_working_capacity": 2.8644,
            "avg_processing_time": 8.9153,
            "soft_deadline_multiplier": 5.678,
            "hard_deadline_multiplier": 7.8983,
            "min_degree": 2,
            "max_degree": 3,
            "route_ratio": 2.1356,
            "malfunction_rate": 0.0868,
            "malfunction_min_steps": 4,
            "malfunction_max_steps": 16,
            "max_soft_deadline": 3070,
            "max_hard_deadline": 3135,
            "map_type": "plain",
        }

        test_cases[52] = {
            "number_of_agents": 5,
            "number_of_airports": 6,
            "number_of_initial_cargo": 7,
            "max_cargo_per_episode": 13,
            "avg_working_capacity": 2.8814,
            "avg_processing_time": 9.0508,
            "soft_deadline_multiplier": 5.5932,
            "hard_deadline_multiplier": 7.661,
            "min_degree": 2,
            "max_degree": 3,
            "route_ratio": 2.1186,
            "malfunction_rate": 0.0947,
            "malfunction_min_steps": 4,
            "malfunction_max_steps": 18,
            "max_soft_deadline": 3032,
            "max_hard_deadline": 3099,
            "map_type": "plain",
        }

        test_cases[53] = {
            "number_of_agents": 5,
            "number_of_airports": 6,
            "number_of_initial_cargo": 7,
            "max_cargo_per_episode": 14,
            "avg_working_capacity": 2.8983,
            "avg_processing_time": 9.1864,
            "soft_deadline_multiplier": 5.5085,
            "hard_deadline_multiplier": 7.4237,
            "min_degree": 2,
            "max_degree": 3,
            "route_ratio": 2.1017,
            "malfunction_rate": 0.1026,
            "malfunction_min_steps": 5,
            "malfunction_max_steps": 19,
            "max_soft_deadline": 2994,
            "max_hard_deadline": 3062,
            "map_type": "plain",
        }

        test_cases[54] = {
            "number_of_agents": 6,
            "number_of_airports": 6,
            "number_of_initial_cargo": 7,
            "max_cargo_per_episode": 14,
            "avg_working_capacity": 2.9153,
            "avg_processing_time": 9.322,
            "soft_deadline_multiplier": 5.4237,
            "hard_deadline_multiplier": 7.1864,
            "min_degree": 2,
            "max_degree": 3,
            "route_ratio": 2.0847,
            "malfunction_rate": 0.1105,
            "malfunction_min_steps": 5,
            "malfunction_max_steps": 21,
            "max_soft_deadline": 2956,
            "max_hard_deadline": 3026,
            "map_type": "plain",
        }

        test_cases[55] = {
            "number_of_agents": 6,
            "number_of_airports": 6,
            "number_of_initial_cargo": 8,
            "max_cargo_per_episode": 14,
            "avg_working_capacity": 2.9322,
            "avg_processing_time": 9.4576,
            "soft_deadline_multiplier": 5.339,
            "hard_deadline_multiplier": 6.9492,
            "min_degree": 2,
            "max_degree": 3,
            "route_ratio": 2.0678,
            "malfunction_rate": 0.1184,
            "malfunction_min_steps": 6,
            "malfunction_max_steps": 22,
            "max_soft_deadline": 2918,
            "max_hard_deadline": 2989,
            "map_type": "plain",
        }

        test_cases[56] = {
            "number_of_agents": 6,
            "number_of_airports": 6,
            "number_of_initial_cargo": 8,
            "max_cargo_per_episode": 14,
            "avg_working_capacity": 2.9492,
            "avg_processing_time": 9.5932,
            "soft_deadline_multiplier": 5.2542,
            "hard_deadline_multiplier": 6.7119,
            "min_degree": 2,
            "max_degree": 3,
            "route_ratio": 2.0508,
            "malfunction_rate": 0.1263,
            "malfunction_min_steps": 6,
            "malfunction_max_steps": 24,
            "max_soft_deadline": 2881,
            "max_hard_deadline": 2953,
            "map_type": "plain",
        }

        test_cases[57] = {
            "number_of_agents": 6,
            "number_of_airports": 6,
            "number_of_initial_cargo": 8,
            "max_cargo_per_episode": 15,
            "avg_working_capacity": 2.9661,
            "avg_processing_time": 9.7288,
            "soft_deadline_multiplier": 5.1695,
            "hard_deadline_multiplier": 6.4746,
            "min_degree": 2,
            "max_degree": 3,
            "route_ratio": 2.0339,
            "malfunction_rate": 0.1342,
            "malfunction_min_steps": 6,
            "malfunction_max_steps": 25,
            "max_soft_deadline": 2843,
            "max_hard_deadline": 2916,
            "map_type": "plain",
        }

        test_cases[58] = {
            "number_of_agents": 6,
            "number_of_airports": 6,
            "number_of_initial_cargo": 8,
            "max_cargo_per_episode": 15,
            "avg_working_capacity": 2.9831,
            "avg_processing_time": 9.8644,
            "soft_deadline_multiplier": 5.0847,
            "hard_deadline_multiplier": 6.2373,
            "min_degree": 2,
            "max_degree": 3,
            "route_ratio": 2.0169,
            "malfunction_rate": 0.1421,
            "malfunction_min_steps": 7,
            "malfunction_max_steps": 27,
            "max_soft_deadline": 2805,
            "max_hard_deadline": 2880,
            "map_type": "plain",
        }

        test_cases[59] = {
            "number_of_agents": 6,
            "number_of_airports": 6,
            "number_of_initial_cargo": 8,
            "max_cargo_per_episode": 15,
            "avg_working_capacity": 3,
            "avg_processing_time": 10.0,
            "soft_deadline_multiplier": 5,
            "hard_deadline_multiplier": 6,
            "min_degree": 2,
            "max_degree": 3,
            "route_ratio": 2,
            "malfunction_rate": 0.15,
            "malfunction_min_steps": 7,
            "malfunction_max_steps": 28,
            "max_soft_deadline": 2767,
            "max_hard_deadline": 2843,
            "map_type": "plain",
        }


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
            PlaneType(id=0, model="A0", max_range=1.42, speed=0.21, max_weight=15),
            PlaneType(id=1, model="B1", max_range=0.9,  speed=0.90, max_weight=10),
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
