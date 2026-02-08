"""
Configuration management for SplunkResearch.

This module loads configuration from YAML files and provides typed access
to configuration values throughout the application.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional


class Config:
    """
    Configuration loader and accessor for SplunkResearch.

    Loads configuration from:
    1. config/default.yaml - default values and non-sensitive settings
    2. config/secrets.yaml - sensitive credentials (gitignored)

    Access config values using dot notation:
        config.get('reward.beta')
        config.get('splunk.port')
    """

    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize configuration loader.

        Args:
            config_dir: Optional path to config directory.
                       If not provided, uses ../config relative to this file.
        """
        if config_dir is None:
            # Default to ../config relative to this file
            src_dir = Path(__file__).parent
            config_dir = src_dir.parent / 'config'
        else:
            config_dir = Path(config_dir)

        self.config_dir = config_dir
        self._config: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self):
        """Load configuration from YAML files."""
        # Load default configuration
        default_path = self.config_dir / 'default.yaml'
        if default_path.exists():
            with open(default_path, 'r') as f:
                self._config = yaml.safe_load(f) or {}
        else:
            raise FileNotFoundError(
                f"Default configuration file not found: {default_path}\n"
                "Please ensure config/default.yaml exists."
            )

        # Load secrets configuration (optional, overrides defaults)
        secrets_path = self.config_dir / 'secrets.yaml'
        if secrets_path.exists():
            with open(secrets_path, 'r') as f:
                secrets = yaml.safe_load(f) or {}
                self._deep_merge(self._config, secrets)
        else:
            import warnings
            warnings.warn(
                f"Secrets file not found: {secrets_path}. "
                f"Copy config/secrets.yaml.example to config/secrets.yaml "
                f"and fill in your credentials.",
                UserWarning, stacklevel=2
            )

    def _deep_merge(self, base: Dict, update: Dict):
        """
        Deep merge update dict into base dict.

        Args:
            base: Base dictionary to merge into
            update: Dictionary with updates to merge
        """
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key_path: Dot-separated path to config value (e.g., 'reward.beta')
            default: Default value if key not found

        Returns:
            Configuration value or default if not found

        Examples:
            config.get('reward.beta')  # Returns 0.33
            config.get('splunk.port')  # Returns 8089
            config.get('paths.random_forest_model')  # Returns model path
        """
        keys = key_path.split('.')
        value = self._config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def get_required(self, key_path: str) -> Any:
        """
        Get configuration value, raise error if not found.

        Args:
            key_path: Dot-separated path to config value

        Returns:
            Configuration value

        Raises:
            KeyError: If configuration key not found
        """
        value = self.get(key_path)
        if value is None:
            raise KeyError(
                f"Required configuration key not found: {key_path}\n"
                f"Please check your config files."
            )
        return value

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section.

        Args:
            section: Top-level section name (e.g., 'reward', 'splunk')

        Returns:
            Dictionary containing all values in the section
        """
        return self.get(section, {})

    def reload(self):
        """Reload configuration from files."""
        self._config = {}
        self._load_config()


# Global configuration instance
# Import this in other modules: from src.config import config
_config_instance: Optional[Config] = None


def get_config() -> Config:
    """
    Get or create the global configuration instance.

    Returns:
        Global Config instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance


# Convenience alias
config = get_config()
