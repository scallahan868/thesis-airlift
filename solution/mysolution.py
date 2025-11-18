# from airlift.solutions import Solution
# from airlift.envs import ActionHelper

# import pickle
# import torch
# import numpy as np

# from airlift.simple_flat_wrapper import AirliftSimpleFlattenWrapper
# from airlift.centralized_critic_model import CentralizedCriticModel
# from airlift.envs.generators.map_generators import PlainMapGenerator

# from airlift.envs.airlift_env import AirliftEnv
# from airlift.curriculum import create_curriculum_world_generator


# class MySolution(Solution):
#     """
#     Utilizing this class for your solution is required for your submission. The primary solution algorithm will go inside the
#     policy function.
#     """
#     def __init__(self):
#         super().__init__()
#         self.wrapper = None
#         self.model = None

#     def reset(self, obs, observation_spaces=None, action_spaces=None, seed=None):
#         # Currently, the evaluator will NOT pass in an observation space or action space (they will be set to None)
#         super().reset(obs, observation_spaces, action_spaces, seed)

#         # Step : Load the model
#         if self.model is None:
#             file_path = "/opt/project/airlift/rllib_logs/PPO_2025-10-15_17-41-09/PPO_AirliftEnv-v0_229cb_00000_0_2025-10-15_17-41-09/checkpoint_000149/policies/shared_policy/policy_state.pkl"
#             with open(file_path, "rb") as f:
#                 policy_state = pickle.load(f)

#             agent_id = list(obs.keys())[0]  # or any valid agent_id
#             obs_space = self.wrapper.observation_space(agent_id)
#             action_space = self.wrapper.action_space(agent_id)
#             num_outputs = 31
#             model_config = {}
#             name = "centralized_critic_model"

#             self.model = CentralizedCriticModel(obs_space, action_space, num_outputs, model_config, name)
#             self.model.load_state_dict(policy_state['model'])
#             self.model.eval()

#     def policies(self, obs, dones, infos):

#         # Step 2: Flatten the obs exactly how they are flattened for training in simple_flat_wrapper.py and centralized_critic_model.py
#         flat_obs_dict = self.wrapper._transform_all(obs)
#         flat_obs = {agent_id: np.concatenate(list(agent_obs.values())) for agent_id, agent_obs in flat_obs_dict.items()}

#         # Step 3: Perform a forward pass of the flattened observations
#         obs_tensor = torch.tensor(list(flat_obs.values()), dtype=torch.float32)
#         with torch.no_grad():
#             action_logits, _ = self.model(obs_tensor, [], None)

#         # Step 4: Unflatten the result in accordance with the action unflattening in simple_flat_wrapper.py
#         from airlift.simple_flat_wrapper import decode_action_from_flat
#         actions = {}
#         for agent_id, logits in zip(flat_obs.keys(), action_logits):
#             actions[agent_id] = decode_action_from_flat(logits.numpy())

#         # Step 5: Return the correctly formatted observations
#         return actions
    
#         # Use the action helper to generate an action
#         # return self._action_helper.sample_valid_actions(obs)
from airlift.solutions import Solution
from airlift.envs import ActionHelper

class MySolution(Solution):
    """
    Utilizing this class for your solution is required for your submission. The primary solution algorithm will go inside the
    policy function.
    """
    def __init__(self):
        super().__init__()

    def reset(self, obs, observation_spaces=None, action_spaces=None, seed=None):
        # Currently, the evaluator will NOT pass in an observation space or action space (they will be set to None)
        super().reset(obs, observation_spaces, action_spaces, seed)

        # Create an action helper using our random number generator
        self._action_helper = ActionHelper(self._np_random)

    def policies(self, obs, dones, infos):
        # Use the acion helper to generate an action
        return self._action_helper.sample_valid_actions(obs)