"""
Fixed working policy that properly handles real environment observations
"""
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Union

class ManualPolicy(nn.Module):
    """
    Manual policy model reconstructed from checkpoint weights
    """
    def __init__(self, state_dict):
        super().__init__()
        
        # Build the actor encoder: 330 -> 256 -> 256
        self.actor_encoder = nn.Sequential(
            nn.Linear(330, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # Build the policy head: 256 -> 601
        self.policy_head = nn.Linear(256, 601)
        
        # Load weights into our model
        self._load_weights(state_dict)
        self.eval()  # Set to evaluation mode
        
    def _load_weights(self, state_dict):
        """Load weights from the checkpoint into our manual model"""
        
        # Actor encoder weights
        self.actor_encoder[0].weight.data = torch.from_numpy(state_dict['encoder.actor_encoder.net.mlp.0.weight']).float()
        self.actor_encoder[0].bias.data = torch.from_numpy(state_dict['encoder.actor_encoder.net.mlp.0.bias']).float()
        
        self.actor_encoder[2].weight.data = torch.from_numpy(state_dict['encoder.actor_encoder.net.mlp.2.weight']).float()
        self.actor_encoder[2].bias.data = torch.from_numpy(state_dict['encoder.actor_encoder.net.mlp.2.bias']).float()
        
        # Policy head weights
        self.policy_head.weight.data = torch.from_numpy(state_dict['pi.net.mlp.0.weight']).float()
        self.policy_head.bias.data = torch.from_numpy(state_dict['pi.net.mlp.0.bias']).float()
        
    def forward(self, obs):
        """Forward pass through the manual model"""
        # Encode observation
        encoded = self.actor_encoder(obs)
        
        # Get policy logits
        policy_logits = self.policy_head(encoded)
        
        return policy_logits

class WorkingPolicy:
    """
    Working policy that uses manual model reconstruction
    """
    def __init__(self, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_path = os.path.join("solution", "checkpoint_000000", "learner_group", "learner", "rl_module", "shared_policy")
        
        self.checkpoint_path = checkpoint_path
        self.model = None
        self._load_manual_model()
        
    def _load_manual_model(self):
        """Load the manual model from checkpoint"""
        try:
            # print(f"🔍 [WORKING_POLICY] Loading manual model from: {self.checkpoint_path}")
            
            # Load the state dict
            module_state_path = os.path.join(self.checkpoint_path, "module_state.pkl")
            if not os.path.exists(module_state_path):
                print(f"❌ [WORKING_POLICY] Module state not found at {module_state_path}")
                return
                
            with open(module_state_path, 'rb') as f:
                state_dict = pickle.load(f)
            
            # Create and load the manual model
            self.model = ManualPolicy(state_dict)
            # print(f"✅ [WORKING_POLICY] Manual model loaded successfully!")
            
        except Exception as e:
            print(f"❌ [WORKING_POLICY] Failed to load manual model: {e}")
            self.model = None

    def flatten_observation_for_training(self, observation: Dict[str, Any]) -> np.ndarray:
        """
        Flatten observation dictionary into vector for model input.
        Handles real environment observations with proper type checking.
        """
        try:
            features = []
            
            if not isinstance(observation, dict):
                print(f"⚠️ [WORKING_POLICY] Unexpected observation type: {type(observation)}")
                return np.zeros(330, dtype=np.float32)
            
            # Helper function to safely extract scalar values
            def safe_scalar(value, default=0.0):
                if value is None:
                    return default
                    
                # If it's already a number
                if isinstance(value, (int, float)):
                    return float(value)
                    
                # If it's a list/array, take first element
                if hasattr(value, '__len__') and not isinstance(value, (str, bytes)):
                    if len(value) > 0:
                        first_elem = value[0]
                        if isinstance(first_elem, (int, float)):
                            return float(first_elem)
                        # Recursively handle nested structures
                        return safe_scalar(first_elem, default)
                    else:
                        return default
                        
                # Try to convert to float
                try:
                    return float(value)
                except (ValueError, TypeError):
                    # If it's a complex object, try to get a numeric property
                    if hasattr(value, 'value'):
                        return safe_scalar(value.value, default)
                    elif hasattr(value, 'id'):
                        return safe_scalar(value.id, default)
                    elif hasattr(value, '__int__'):
                        return float(int(value))
                    else:
                        return default
            
            # Extract scalar values with robust handling
            # Check that these values are to be expected
            features.append(safe_scalar(observation.get('state', 0)))
            features.append(safe_scalar(observation.get('current_airport', 0)))
            features.append(safe_scalar(observation.get('destination', 0)))
            features.append(safe_scalar(observation.get('current_weight', 0.0)))
            features.append(safe_scalar(observation.get('plane_type', 0)))
            features.append(safe_scalar(observation.get('max_weight', 0.0)))
            
            # Handle array/list fields with robust processing
            def extract_list_features(field_name, max_size):
                field_data = observation.get(field_name, [])
                float_data = []
                
                if field_data is None:
                    return [0.0] * max_size
                    
                # Handle different types of field_data
                if hasattr(field_data, '__iter__') and not isinstance(field_data, (str, bytes)):
                    # It's iterable - process each element
                    for item in field_data:
                        try:
                            float_val = safe_scalar(item, 0.0)
                            float_data.append(float_val)
                        except Exception:
                            float_data.append(0.0)
                else:
                    # Single value
                    float_data.append(safe_scalar(field_data, 0.0))
                
                # Pad or truncate to max_size
                padded = (float_data + [0.0] * max_size)[:max_size]
                return padded
            
            # Add cargo arrays (100 + 100 + 120 = 320, plus 6 scalars = 326)
            cargo_onboard_features = extract_list_features('cargo_onboard', 100)
            features.extend(cargo_onboard_features)
            
            cargo_at_airport_features = extract_list_features('cargo_at_current_airport', 100)
            features.extend(cargo_at_airport_features)
            
            available_routes_features = extract_list_features('available_routes', 120)
            features.extend(available_routes_features)
            
            # Handle additional fields if present (to reach 330)
            if 'next_action' in observation:
                next_action_features = extract_list_features('next_action', 2)
                features.extend(next_action_features)
            else:
                features.extend([0.0, 0.0])
            
            if 'globalstate' in observation:
                globalstate_features = extract_list_features('globalstate', 2)
                features.extend(globalstate_features)
            else:
                features.extend([0.0, 0.0])
            
            # Pad to exactly 330 features
            while len(features) < 330:
                features.append(0.0)
            
            features = features[:330]  # Truncate if too long
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            print(f"❌ [WORKING_POLICY] Flattening error: {e}")
            print(f"❌ [WORKING_POLICY] Observation type: {type(observation)}")
            if isinstance(observation, dict):
                print(f"❌ [WORKING_POLICY] Observation keys: {list(observation.keys())}")
                for key, value in list(observation.items())[:3]:
                    print(f"  {key}: {type(value)} = {str(value)[:100]}")
            # Return zero observation if there's an error
            return np.zeros(330, dtype=np.float32)

    def forward_inference_manual(self, obs, deterministic=False):
        """
        Run inference using the manual model
        """
        if self.model is None:
            return self.forward_inference_random(obs, deterministic)

        try:
            # print(f"🎯 [WORKING_POLICY] Running manual inference...")
            
            # Extract current airport for destination selection
            current_airport = obs.get('current_airport', None)
            
            # This is a single agent's observation - flatten it directly
            obs_flat = self.flatten_observation_for_training(obs)
            
            # Convert to tensor and add batch dimension
            obs_tensor = torch.from_numpy(obs_flat).unsqueeze(0)  # Shape: (1, 330)
            
            # Run model inference
            with torch.no_grad():
                logits = self.model(obs_tensor)  # Shape: (1, 601)
            
            # print(f"Logits shape: {logits.shape}, range: [{logits.min():.3f}, {logits.max():.3f}]")
            
            # Convert to proper action format for single agent
            action = self.convert_logits_to_action(logits[0], obs, deterministic)
            
            # print(f"✅ [WORKING_POLICY] Manual inference successful: {action}")
            return action
            
        except Exception as e:
            print(f"❌ [WORKING_POLICY] Manual inference failed: {e}")
            import traceback
            traceback.print_exc()
            return self.forward_inference_random(obs, deterministic)
    
    def convert_logits_to_action(self, logits_flat, obs, deterministic=False):
        """
        Convert model logits to proper action format respecting observation constraints.
        Only one action is performed per step, in priority order: drop off > pick up > fly > change priority.
        """
        try:
            # For MultiDiscrete [2, 100, 100, 13], use first 215 outputs
            logits = logits_flat[:215]

            # Split logits for each action component
            priority_logits = logits[:2]  # 2 classes for priority
            cargo_load_logits = logits[2:102]  # 100 classes for cargo to load
            cargo_unload_logits = logits[102:202]  # 100 classes for cargo to unload
            destination_logits = logits[202:215]  # 13 classes for destination (airports 0-12)

            if deterministic:
                # Take argmax for deterministic actions
                priority = int(torch.argmax(priority_logits))
                cargo_load_idx = int(torch.argmax(cargo_load_logits))
                cargo_unload_idx = int(torch.argmax(cargo_unload_logits))
            else:
                # Sample from distribution for stochastic actions
                priority_dist = torch.softmax(priority_logits, dim=0)
                priority = int(torch.multinomial(priority_dist, 1))

                cargo_load_dist = torch.softmax(cargo_load_logits, dim=0)
                cargo_load_idx = int(torch.multinomial(cargo_load_dist, 1))

                cargo_unload_dist = torch.softmax(cargo_unload_logits, dim=0)
                cargo_unload_idx = int(torch.multinomial(cargo_unload_dist, 1))

            # Get available cargo and routes
            cargo_at_airport = obs.get('cargo_at_current_airport', [])
            cargo_onboard = obs.get('cargo_onboard', [])
            available_routes = obs.get('available_routes', [])
            current_airport = obs.get('current_airport', None)

            # Enforce one action per step, in priority order:
            # 1. Drop off cargo
            if cargo_onboard and cargo_unload_idx in cargo_onboard:
                action = {
                    'priority': priority,
                    'cargo_to_load': [],
                    'cargo_to_unload': [cargo_unload_idx],
                    'destination': current_airport  # stay
                }
                return action

            # 2. Pick up cargo
            if cargo_at_airport and cargo_load_idx in cargo_at_airport:
                action = {
                    'priority': priority,
                    'cargo_to_load': [cargo_load_idx],
                    'cargo_to_unload': [],
                    'destination': current_airport  # stay
                }
                return action

            # 3. Fly to a new destination (if possible)
            destination = self.select_most_likely_legal_destination(destination_logits, available_routes, current_airport, deterministic)
            if destination != current_airport:
                action = {
                    'priority': priority,
                    'cargo_to_load': [],
                    'cargo_to_unload': [],
                    'destination': destination
                }
                return action

            # 4. Change priority (if nothing else to do)
            action = {
                'priority': priority,
                'cargo_to_load': [],
                'cargo_to_unload': [],
                'destination': current_airport  # stay
            }
            return action

        except Exception as e:
            print(f"❌ [WORKING_POLICY] Action conversion failed: {e}")
            return self.forward_inference_random({}, deterministic)

    def select_most_likely_legal_destination(self, destination_logits, available_routes, current_airport=None, deterministic=False):
        """
        Select the most likely legal destination from the model's output, restricted to available_routes.
        If no available_routes, stay at current airport (or pick 0 as fallback).
        """
        try:
            if not available_routes:
                # No available routes, stay at current airport or fallback
                return current_airport if current_airport is not None else 0

            destination_probs = torch.softmax(destination_logits, dim=0)
            # Get destinations ranked by probability (highest first)
            ranked_destinations = torch.argsort(destination_probs, descending=True)
            for dest_tensor in ranked_destinations:
                destination = int(dest_tensor)
                if destination in available_routes:
                    return destination
            # Fallback: pick the first available route
            return available_routes[0]
        except Exception as e:
            print(f"❌ [WORKING_POLICY] Legal destination selection failed: {e}")
            return 0
    
    def select_destination_from_model(self, destination_logits, deterministic=False):
        """
        Select destination directly from model's output, trusting the trained behavior.
        Only ensure it fits within the action space bounds.
        """
        try:
            if deterministic:
                # Take argmax for deterministic actions
                destination = int(torch.argmax(destination_logits))
            else:
                # Sample from distribution for stochastic actions
                destination_dist = torch.softmax(destination_logits, dim=0)
                destination = int(torch.multinomial(destination_dist, 1))
            
            # Ensure destination is within valid bounds (0-12 for the airlift environment)
            destination = max(0, min(destination, 12))
            
            return destination
            
        except Exception as e:
            print(f"❌ [WORKING_POLICY] Model destination selection failed: {e}")
            return 0  # NOAIRPORT_ID as safe fallback

    def select_valid_destination(self, destination_logits, current_airport=None, deterministic=False):
        """
        Select destination from model's ranked preferences, ensuring validity.
        Try top destinations in order until finding a valid one.
        """
        try:
            # Convert logits to probabilities and get ranked preferences
            destination_probs = torch.softmax(destination_logits, dim=0)
            
            if deterministic:
                # Get destinations ranked by probability (highest first)
                ranked_destinations = torch.argsort(destination_probs, descending=True)
            else:
                # For stochastic, still use preference ranking but add some randomness
                # Sample from top 5 destinations with weighted probability
                top_k = min(5, len(destination_probs))
                top_destinations = torch.topk(destination_probs, top_k)
                
                # Re-normalize top destinations and sample
                renorm_probs = top_destinations.values / top_destinations.values.sum()
                selected_idx = torch.multinomial(renorm_probs, 1)
                selected_dest = top_destinations.indices[selected_idx]
                
                # Still get full ranking for fallback
                ranked_destinations = torch.argsort(destination_probs, descending=True)
                
                # Try the sampled destination first, then fall back to ranking
                ranked_destinations = torch.cat([selected_dest, ranked_destinations])
                ranked_destinations = torch.unique(ranked_destinations, sorted=False)
            
            # Try destinations in preference order
            for dest_tensor in ranked_destinations:
                destination = int(dest_tensor)
                
                # Basic validity checks
                if destination < 0:
                    continue
                    
                # Don't go to same airport (unless no other choice)
                if current_airport is not None and destination == current_airport:
                    continue
                    
                # Conservative upper bound - airlift environment has destination:Discrete(13) = 0-12
                if destination >= 13:
                    continue
                
                # Found a valid destination
                return destination
            
            # Fallback: if no valid destination found, use a safe default
            # Try destinations 0-5 as they're most likely to exist (within 0-12 range)
            for safe_dest in [0, 1, 2, 3, 4, 5]:
                if current_airport is None or safe_dest != current_airport:
                    print(f"🔄 [WORKING_POLICY] Using fallback destination: {safe_dest}")
                    return safe_dest
            
            # Last resort
            print(f"⚠️ [WORKING_POLICY] Using last resort destination: 0")
            return 0
            
        except Exception as e:
            print(f"❌ [WORKING_POLICY] Destination selection failed: {e}")
            return 0
    
    def forward_inference_random(self, obs, deterministic=False):
        """Fallback random action"""
        import random
        return {
            'priority': random.randint(0, 1),
            'cargo_to_load': [random.randint(0, 99)] if random.random() > 0.5 else [],
            'cargo_to_unload': [random.randint(0, 99)] if random.random() > 0.5 else [],
            'destination': random.randint(0, 19)
        }
    
    def forward_inference(self, obs, deterministic=False):
        """Main inference method"""
        return self.forward_inference_manual(obs, deterministic)


# Test the fixed policy
if __name__ == "__main__":
    print("Testing Fixed Working Policy...")
    
    policy = WorkingPolicy()
    
    # Test with simple observation (like our test case)
    test_obs = {
        'state': 1,
        'current_airport': 0,
        'destination': 5,
        'current_weight': 10.5,
        'plane_type': 0,
        'max_weight': 100.0,
        'cargo_onboard': [1, 2, 3],
        'cargo_at_current_airport': [4, 5],
        'available_routes': [0, 1, 2, 3, 4]
    }
    
    print("\\nTesting inference...")
    action = policy.forward_inference(test_obs)
    print(f"Action: {action}")
    print(f"Action type: {type(action)}")
