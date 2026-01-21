import numpy as np
from ray.rllib.callbacks.callbacks import RLlibCallback
import ray


class MissedDeliveriesCallback(RLlibCallback):
    """Logs missed deliveries percentage as a custom metric."""

    def on_episode_end(self, *, episode, metrics_logger, **kwargs):
        print(ray.__version__)
        info_list = episode.get_infos()
        for agent, agent_infos in info_list.items():
            for info in reversed(agent_infos):
                if isinstance(info, dict) and "episode_metrics" in info:
                    metrics = info["episode_metrics"]
                    if "missed_deliveries_percentage" in metrics:
                        metrics_logger.log_value(
                            "missed_deliveries_percentage",
                            metrics["missed_deliveries_percentage"],
                            reduce="mean",
                            window=50,
                        )
                    break


class TotalRewardsCallback(RLlibCallback):
    """Logs total rewards for all agents as a custom metric."""

    def on_episode_end(self, *, episode, metrics_logger, **kwargs):
        info_list = episode.get_infos()
        for agent, agent_infos in info_list.items():
            for info in reversed(agent_infos):
                if isinstance(info, dict) and "episode_metrics" in info:
                    metrics = info["episode_metrics"]
                    if "total_rewards_for_all_agents" in metrics:
                        metrics_logger.log_value(
                            "total_rewards_all_agents",
                            metrics["total_rewards_for_all_agents"],
                            reduce="mean",
                            window=50,
                        )
                    break


class AgentRewardsCallback(RLlibCallback):
    """Logs individual agent rewards as custom metrics."""

    def on_episode_end(self, *, episode, metrics_logger, **kwargs):
        info_list = episode.get_infos()
        for agent, agent_infos in info_list.items():
            for info in reversed(agent_infos):
                if isinstance(info, dict) and "episode_metrics" in info:
                    metrics = info["episode_metrics"]
                    if "agent_rewards" in metrics:
                        for agent_id, reward in metrics["agent_rewards"].items():
                            metrics_logger.log_value(
                                f"agent_{agent_id}_total_reward",
                                reward,
                                reduce="mean",
                                window=50,
                            )
                    break


class CargoMetricsCallback(RLlibCallback):
    """Logs cargo-related metrics."""

    def on_episode_end(self, *, episode, metrics_logger, **kwargs):
        info_list = episode.get_infos()
        for agent, agent_infos in info_list.items():
            for info in reversed(agent_infos):
                if isinstance(info, dict) and "episode_metrics" in info:
                    metrics = info["episode_metrics"]
                    if "total_cargo_generated" in metrics:
                        metrics_logger.log_value(
                            "total_cargo_generated",
                            metrics["total_cargo_generated"],
                            reduce="mean",
                            window=50,
                        )
                    if "missed_deliveries" in metrics:
                        metrics_logger.log_value(
                            "missed_deliveries_count",
                            metrics["missed_deliveries"],
                            reduce="mean",
                            window=50,
                        )
                    break


class CurriculumCallback(RLlibCallback):
    """Logs curriculum information."""

    def on_episode_end(self, *, episode, metrics_logger, **kwargs):
        info_list = episode.get_infos()
        for agent, agent_infos in info_list.items():
            for info in reversed(agent_infos):
                if isinstance(info, dict) and "curriculum_info" in info:
                    curriculum_info = info["curriculum_info"]
                    if "testid" in curriculum_info:
                        metrics_logger.log_value(
                            "curriculum_testid",
                            curriculum_info["testid"],
                            reduce="mean",
                            window=10,
                        )
                    if "episode_count" in curriculum_info:
                        metrics_logger.log_value(
                            "curriculum_episode_count",
                            curriculum_info["episode_count"],
                            reduce="mean",
                            window=10,
                        )
                    break
