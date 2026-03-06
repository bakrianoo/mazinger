"""Allow ``python -m mazinger_dubber`` invocation."""

import os
import warnings

# Remove Jupyter/Colab-specific matplotlib backend that may not be available
# in this virtual environment, causing an import error in downstream libs.
os.environ.pop("MPLBACKEND", None)

# ── Silence noisy third-party warnings ──────────────────────────────────────
# torchcodec FFmpeg compatibility warning from pyannote
warnings.filterwarnings("ignore", message=".*torchcodec is not installed correctly.*")
# pyannote TF32 reproducibility warning
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
# Lightning checkpoint auto-upgrade info
warnings.filterwarnings("ignore", message=".*Lightning automatically upgraded.*")

# Suppress Lightning upgrade log at the logging level too
import logging
logging.getLogger("lightning.pytorch.utilities.migration").setLevel(logging.WARNING)

from mazinger_dubber.cli import main

if __name__ == "__main__":
    main()
