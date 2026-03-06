"""
Mazinger Dubber -- End-to-end video dubbing pipeline.

Transcribe, translate, and voice-clone audio from any video URL.
Each stage can be used independently or chained through the unified
``MazingerDubber`` pipeline class.
"""

from mazinger_dubber.pipeline import MazingerDubber
from mazinger_dubber.paths import ProjectPaths

__all__ = ["MazingerDubber", "ProjectPaths"]
__version__ = "0.1.0"
