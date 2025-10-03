import sys
from pathlib import Path

import pytz
from loguru import logger

# Path to logs directory relative to project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
LOG_DIR = PROJECT_ROOT / "data" / "logs"

# Ensure directory exists
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Timezone setup
TZ = pytz.timezone("Europe/Moscow")

logger.remove()

# Info level log with daily rotation and 14 days retention
logger.add(
    sink=str(LOG_DIR / "info.log"),
    level="INFO",
    rotation="1 day",
    retention="14 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    serialize=False,
    enqueue=True,
    colorize=False,
)

# Debug level log with file size rotation
logger.add(
    sink=str(LOG_DIR / "debug.log"),
    level="DEBUG",
    rotation="100 MB",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    serialize=False,
    enqueue=True,
    colorize=False,
)

# Error level log with file size rotation
logger.add(
    sink=str(LOG_DIR / "error.log"),
    level="ERROR",
    rotation="100 MB",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    serialize=False,
    enqueue=True,
    colorize=False,
)

# Console output for DEBUG level
logger.add(
    sink=sys.stderr,
    level="DEBUG",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{message}</cyan>",
    enqueue=True,
    colorize=True,
)

# Console output for ERROR level (duplication, might be unnecessary)
logger.add(
    sink=sys.stderr,
    level="ERROR",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{message}</cyan>",
    enqueue=True,
    colorize=True,
)
