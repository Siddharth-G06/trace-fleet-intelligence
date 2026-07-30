"""
Configuration loader for the TRACE fleet intelligence system.

Reads ``config/config.yaml`` and exposes it as a plain Python dict.
Environment variables take precedence over YAML values for any top-level
key, enabling twelve-factor-style configuration in production.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Default path relative to the project root
_DEFAULT_CONFIG_PATH = "config/config.yaml"


@lru_cache(maxsize=1)
def _load_yaml(path: str) -> dict[str, Any]:
    """Read and parse the YAML file exactly once.

    Args:
        path: Absolute or relative path to the YAML config file.

    Returns:
        Parsed YAML content as a nested dict.

    Raises:
        FileNotFoundError: When *path* does not point to an existing file.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: '{config_path.resolve()}'. "
            "Ensure config/config.yaml exists at the project root."
        )
    with config_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    logger.info("Config loaded from '%s'", config_path.resolve())
    return data


def _apply_env_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """Overlay top-level keys with matching environment variables.

    For each top-level key in *config*, the function checks whether an
    environment variable of the same name (upper-cased) exists and, if so,
    replaces the config value with the env value.

    Args:
        config: Base configuration dict parsed from YAML.

    Returns:
        A new dict with environment-variable overrides applied.
    """
    result: dict[str, Any] = dict(config)
    for key in list(result.keys()):
        env_key = key.upper()
        if env_key in os.environ:
            logger.debug(
                "Config key '%s' overridden by environment variable '%s'",
                key,
                env_key,
            )
            result[key] = os.environ[env_key]
    return result


def load_config(path: str = _DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load the TRACE configuration, with env-variable overrides.

    The YAML file is read **only once** per unique *path*; subsequent calls
    return the cached result instantly.  Environment variables whose names
    match any top-level YAML key (case-insensitive) take precedence.

    Args:
        path: Path to the YAML config file.  Defaults to
            ``config/config.yaml`` relative to the working directory.

    Returns:
        Flat-nested dict of configuration values.

    Raises:
        FileNotFoundError: If the YAML file cannot be found at *path*.

    Example::

        cfg = load_config()
        window_size = cfg["data"]["window_size"]   # 30
    """
    raw = _load_yaml(path)
    return _apply_env_overrides(raw)
