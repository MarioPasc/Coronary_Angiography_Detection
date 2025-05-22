# optimization/__init__.py

import logging

# Global logger for the entire optimization module
LOGGER = logging.getLogger("ica.optimization")
LOGGER.setLevel(logging.INFO)

# Configure a default stream handler (can be overridden in cli or main)
_handler = logging.StreamHandler()
_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
_handler.setFormatter(_formatter)
LOGGER.addHandler(_handler)
