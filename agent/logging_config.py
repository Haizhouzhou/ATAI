from __future__ import annotations

"""
Simple logging configuration helper.

All modules in the agent call get_logger() instead of configuring
logging on their own. This keeps log output consistent and avoids
duplicate handlers when the package is imported multiple times.
"""

import logging
from typing import Optional

# Guard to ensure we configure logging only once
_LOGGING_CONFIGURED: bool = False


def _configure_root_logger() -> None:
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    _LOGGING_CONFIGURED = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Return a logger. On first call, configure a simple global setup.

    Parameters
    ----------
    name:
        Optional logger name. If None, use 'atai_agent' as default.
    """
    _configure_root_logger()
    if not name:
        name = "atai_agent"
    return logging.getLogger(name)
