from __future__ import annotations

"""
Application-level package.

External code (including your Speakeasy bot) can simply import
`answer_question` from here.
"""

from .main import answer_question

__all__ = ["answer_question"]
