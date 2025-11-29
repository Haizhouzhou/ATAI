from __future__ import annotations

"""
Simple configuration container for the agent.

Right now this is very small, but keeping it in a dedicated module
makes it easy to add flags later (e.g. to toggle debug behaviour).
"""

from dataclasses import dataclass


@dataclass
class RuntimeConfig:
    max_recommendations: int = 10
    max_images_per_entity: int = 1


# Global default config instance (can be imported if needed)
DEFAULT_CONFIG = RuntimeConfig()
