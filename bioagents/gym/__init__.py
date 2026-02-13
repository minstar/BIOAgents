"""Gymnasium-compatible environment interface for BIOAgents."""

from bioagents.gym.agent_env import BioAgentGymEnv, BIOAGENT_ENV_ID, register_bioagent_gym

__all__ = [
    "BioAgentGymEnv", "BIOAGENT_ENV_ID", "register_bioagent_gym",
    # Lazy imports for gym_coach and training_memory
]


def get_gym_coach():
    """Lazy import for GymCoach."""
    from bioagents.gym.gym_coach import GymCoach, CoachConfig
    return GymCoach, CoachConfig


def get_training_memory():
    """Lazy import for TrainingMemory."""
    from bioagents.gym.training_memory import TrainingMemory
    return TrainingMemory
