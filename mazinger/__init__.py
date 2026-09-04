"""
Mazinger Dubber -- End-to-end video dubbing pipeline.

Transcribe, translate, and voice-clone audio from any video URL.
Each stage can be used independently or chained through the unified
``MazingerDubber`` pipeline class.
"""

from importlib.metadata import PackageNotFoundError, version as _installed_version

from mazinger.pipeline import MazingerDubber
from mazinger.paths import ProjectPaths
from mazinger.utils import LLMUsageTracker

__all__ = ["MazingerDubber", "ProjectPaths", "LLMUsageTracker"]

# Single source of truth is ``[project] version`` in pyproject.toml; read it
# back from the installed distribution so the two can never drift apart.
try:
    __version__ = _installed_version("mazinger")
except PackageNotFoundError:  # running straight from a source checkout
    __version__ = "0.0.0.dev0"
