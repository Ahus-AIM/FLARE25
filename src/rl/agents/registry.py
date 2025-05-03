from typing import Type

from src.rl.agents import ThresholdAgent
from src.rl.agents.attention_based.ppo import AttentionPPOThresholdAgent

THRESHOLD_AGENT_REGISTRY: dict[str, Type[ThresholdAgent]] = {
    # "PPO": PPOThresholdAgent,
    # "attention_ddpg": AttentionDDPGThresholdAgent,
    "attention_ppo": AttentionPPOThresholdAgent,
}
