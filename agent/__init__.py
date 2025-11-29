from __future__ import annotations

"""
Agent package init.

For convenience we re-export the main entrypoint that most callers
care about: AgentCore and get_global_composer().
"""

from .composer import AgentCore, get_global_composer

__all__ = ["AgentCore", "get_global_composer"]
