from airlift.solutions import Solution
from airlift.envs import ActionHelper
from solution.working_policy_fixed import WorkingPolicy


class MySolution(Solution):
    """
    Utilizing this class for your solution is required for your submission. The primary solution algorithm will go inside the
    policy function.
    """
    def __init__(self):
        super().__init__()
        
        # Initialize the trained policy
        checkpoint_path = r"C:\Users\SCallahan\Desktop\airlift-starter-kit\solution\checkpoint_000149\learner_group\learner\rl_module\shared_policy"
        try:
            self.policy = WorkingPolicy(checkpoint_path)
            print(f"✅ Successfully loaded trained policy from checkpoint")
        except Exception as e:
            print(f"❌ Failed to load trained policy: {e}")
            print("   Falling back to random actions")
            self.policy = None

    def reset(self, obs, observation_spaces=None, action_spaces=None, seed=None):
        # Currently, the evaluator will NOT pass in an observation space or action space (they will be set to None)
        super().reset(obs, observation_spaces, action_spaces, seed)

        # Create an action helper using our random number generator
        self._action_helper = ActionHelper(self._np_random)

    def policies(self, obs, dones, infos):
        # Print agent observations and actions for debugging
        # print("\n[MySolution] Step Debug Info:")
        # for agent_id, agent_obs in obs.items():
        #     state = agent_obs.get('state')
        #     airport = agent_obs.get('current_airport')
        #     dest = agent_obs.get('destination')
        #     cargo_at_airport = agent_obs.get('cargo_at_current_airport', [])
        #     cargo_onboard = agent_obs.get('cargo_onboard', [])
        #     available_routes = agent_obs.get('available_routes', [])
        #     print(f"  {agent_id}: state={state}, airport={airport}, dest={dest}, cargo_at_airport={cargo_at_airport}, cargo_onboard={cargo_onboard}, available_routes={available_routes}")

        # Use trained policy if available, otherwise fall back to random actions
        if self.policy is not None:
            try:
                actions = {}
                for agent_id, agent_obs in obs.items():
                    action = self.policy.forward_inference(agent_obs, deterministic=True)
                    print(f"    Action for {agent_id}: {action}")
                    actions[agent_id] = action
                return actions
            except Exception as e:
                print(f"❌ Policy inference failed: {e}")
                print("   Falling back to random actions")

        # Use the action helper to generate an action (fallback or if no policy loaded)
        return self._action_helper.sample_valid_actions(obs)