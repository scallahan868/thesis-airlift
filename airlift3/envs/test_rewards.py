from airlift.envs.airlift_env_rl import AirliftEnv
from airlift.envs.generators.world_generators import WorldGenerator
from airlift.envs.generators.airport_generators import RandomAirportGenerator
from airlift.envs.generators.route_generators import RouteByDistanceGenerator
from airlift.envs.generators.cargo_generators import StaticCargoGenerator
from airlift.envs.generators.airplane_generators import AirplaneGenerator
from airlift.envs.plane_types import PlaneType

def test_calculate_rewards():
    wg = WorldGenerator(
        plane_types=[PlaneType(id=0, model='A0', max_range=2.0, speed=0.1, max_weight=5)],
        airport_generator=RandomAirportGenerator(3),
        route_generator=RouteByDistanceGenerator(),
        cargo_generator=StaticCargoGenerator(1),
        airplane_generator=AirplaneGenerator(1),
    )
    env = AirliftEnv(world_generator=wg)
    obs = env.reset()

    actions = env.sample_valid_actions()

    for _ in range(3):
        obs, rewards, dones, infos = env.step(actions)
        print("STEP REWARDS:", rewards)
        prev_cargo_delivered = {c.id for c in env.cargo if env.cargo_delivered(c)}
        calc_rewards = env.calculate_rewards(actions, prev_cargo_delivered, dones, infos)
        print("CALCULATE_REWARDS:", calc_rewards)

if __name__ == "__main__":
    print("In __main__ block")
    test_calculate_rewards()