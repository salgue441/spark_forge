# sparkforge/config/__init__.py

"""
SparkForge Configuration Module

This module provides configuration management for the SparkForge framework. It
handles loading, validating, and managing configuration settings from YAML files
"""

import os
import yaml
from typing import Dict, Any, Optional, Union
from pathlib import Path
import logging


class ConfigurationError(Exception):
    """
    Custom exception for configuration-related errors
    """

    pass


class SparkForgeConfig:
    """
    Configuration manager for SparkForge framework

    Handles loading configuration from YAML files, environment variables,
    and provides easy access to configuration parameters.
    """

    def __init__(self, config_path: Optional[str] = None, environment: str = "default"):
        """
        Initialize the configuration manager.

        Args:
          config_path (Optional, str): Path to the configuration file. If None,
                                       uses default config.
          environment (str): Configuration environment (default, dev, prod, etc)
        """

        self.environment = environment
        self.logger = logging.getLogger(__name__)

        self._config = self._load_config(config_path)
        self._apply_environment_overrides()
        self._validate_config()

        self.logger.info(f"Configuration loaded for environment: {environment}")

    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration from YAML file

        Args:
          config_path (Optional, str): Path to the configuration file. If None,
                                       uses default config.

        Returns:
          Dict[str, Any]: A dictionary containing the config provided

        Raises:
          ConfigurationError: If the file is not found
          ConfigurationError: If there's an error parsing the yaml file
          ConfigurationError: If there's an error loading the configuration
        """

        if config_path is None:
            config_dir = Path(__file__).parent
            config_path = config_dir / "default_config.yaml"

        else:
            config_path = Path(config_path)

        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, "r") as file:
                config = yaml.safe_load(file)

            return config

        except yaml.YAMLError as e:
            raise ConfigurationError(f"Error parsing configuration file: {e}")

        except Exception as e:
            raise ConfigurationError(f"Error loading configuration: {e}")

    def _apply_environment_overrides(self):
        """
        Apply environment-specific configuration overrides
        """

        env_overrides = {
            "SPARK_MASTER": ["spark", "master"],
            "SPARK_APP_NAME": ["spark", "app_name"],
            "SPARKFORGE_LOG_LEVEL": ["logging", "level"],
            "SPARKFORGE_OUTPUT_PATH": ["output", "results", "output_path"],
            "SPARKFORGE_MODEL_PATH": ["output", "model_persistence", "base_path"],
        }

        for env_var, config_path in env_overrides.items():
            env_value = os.getenv(env_var)
            if env_value:
                self._set_nested_config(config_path, env_value)
                self.logger.info(f"Configuration override from {env_var}: {env_value}")

        env_config_path = Path(__file__).parent / f"{self.environment}_config.yaml"
        if env_config_path.exists():
            try:
                with open(env_config_path, "r") as file:
                    env_config = yaml.safe_load(file)

                self._deep_merge(self._config, env_config)
                self.logger.info(f"Loaded environment config: {env_config_path}")

            except Exception as e:
                self.logger.warning(f"Error loading environment config: {e}")

    def _set_nested_config(self, path: list, value: Any):
        """
        Set a nested configuration value using a path

        Args:
          path (list): Path where the value is stored
          value (Any): Value to be set for the path
        """

        config = self._config
        for key in path[:-1]:
            if key not in config:
                config[key] = {}

            config = config[key]

        config[path[-1]] = value

    def _deep_merge(self, base: Dict, override: Dict):
        """
        Deep merge two dictionaries

        Args:
          base (dict): Base dictionary to be overriden
          override (dict): Dictionary with new values to be set
        """

        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)

            else:
                base[key] = value

    def _validate_config(self):
        """
        Validate configuration for required fields and consistency

        Raises:
          ConfigurationError: If a required section is missing
          ConfigurationError: If the data split ratio is not 1.0
          ConfigurationError: If the time series window size is not a list
        """

        required_sections = ["spark", "feature_engineering", "models", "evaluation"]

        for section in required_sections:
            if section not in self._config:
                raise ConfigurationError(
                    f"Required configuration section missing: {section}"
                )

        # Validate Spark configuration
        spark_config = self._config.get("spark", {})
        if "app_name" not in spark_config:
            self._config["spark"]["app_name"] = "SparkForge_ML_Pipeline"

        data_config = self._config.get("data", {}).get("splitting", {})
        ratios = [
            data_config.get("train_ratio", 0.7),
            data_config.get("validation_ratio", 0.15),
            data_config.get("test_ratio", 0.15),
        ]

        if abs(sum(ratios) - 1.0) > 0.01:
            raise ConfigurationError(
                f"Data splitting ratios must sum to 1.0, got: {sum(ratios)}"
            )

        fe_config = self._config.get("feature_engineering", {})
        if "time_series" in fe_config:
            ts_config = fe_config["time_series"]

            if not isinstance(ts_config.get("default_window_sizes", []), list):
                raise ConfigurationError(
                    "time_series.default_window_sizes must be a list"
                )

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
          key (str): Configuration key in dot notation (e.g., 'spark.app_name')
          default (Any): Default value if key is not found

        Returns:
          The found value if the key is found and is valid, otherwise, the
          provided default value
        """

        keys = key.split(".")
        value = self._config

        try:
            for k in keys:
                value = value[k]

        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any):
        """
        Set configuration value using dot notation

        Args:
          key (str): Configuration key in dot notation
          value (Any): Value to be set
        """

        keys = key.split(".")
        self._set_nested_config(keys, value)

    def get_spark_config(self) -> Dict[str, Any]:
        """
        Get Spark-specific configuration
        """

        return self._config.get("spark", {})

    def get_feature_engineering_config(self) -> Dict[str, Any]:
        """
        Get feature engineering configuration
        """

        return self._config.get("feature_engineering", {})

    def get_model_config(self) -> Dict[str, Any]:
        """
        Get model training configuration
        """

        return self._config.get("models", {})

    def get_ensemble_config(self) -> Dict[str, Any]:
        """
        Get ensemble configuration
        """

        return self._config.get("ensemble", {})

    def get_evaluation_config(self) -> Dict[str, Any]:
        """
        Get evaluation configuration
        """

        return self._config.get("evaluation", {})

    def get_data_config(self) -> Dict[str, Any]:
        """
        Get data processing configuration
        """

        return self._config.get("data", {})

    def get_output_config(self) -> Dict[str, Any]:
        """
        Get output configuration
        """

        return self._config.get("output", {})

    def get_performance_config(self) -> Dict[str, Any]:
        """
        Get performance configuration
        """

        return self._config.get("performance", {})

    def to_dict(self) -> Dict[str, Any]:
        """
        Return configuration as dictionary
        """

        return self._config.copy()

    def save_config(self, path: str):
        """
        Save current configuration to a new file
        """

        try:
            with open(path, "w") as file:
                yaml.dump(self._config, file, default_flow_style=False, indent=2)

            self.logger.info(f"Configuration saved to: {path}")

        except Exception as e:
            raise ConfigurationError(f"Error saving configuration: {e}")

    def __str__(self) -> str:
        """
        String representation of configuration
        """

        return f"SparkForgeConfig(environment={self.environment})"

    def __repr__(self) -> str:
        return self.__str__()


# Global configuration instance
_global_config: Optional[SparkForgeConfig] = None


def get_config(
    config_path: Optional[str] = None,
    environment: str = "default",
    force_reload: bool = False,
) -> SparkForgeConfig:
    """
    Get global configuration instance

    Args:
      config_path (Optional, str): Path to the configuration file
      environment (str): Configuration environment
      force_reload (bool): Force reload of configuration

    Returns:
      SparkForgeConfig instance
    """

    global _global_config
    if _global_config is None or force_reload:
        _global_config = SparkForgeConfig(config_path, environment)

    return _global_config


def set_config(config: SparkForgeConfig):
    """
    Set global configuration instance
    """

    global _global_config
    _global_config = config


def validate_model_config(config: Dict[str, Any], problem_type: str) -> bool:
    """
    Validate model configuration for specific problem type

    Args:
      config (Dict[str, Any]): Config in dictionary format
      problem_type (str): Problem to be solved (e.g., classification)

    Returns:
      True if the config matches the model configuration and problem type, false otherwise.

    Raises:
      ConfigurationError: If the problem stated is not supported or invalid
      ConfigurationError: If there's no configuration for the problem type
      ConfigurationError: If there's no specified algorithm for the problem
      ConfigurationError: If there's no default algorithm for the problem type
    """

    if problem_type not in ["classification", "regression"]:
        raise ConfigurationError(f"Invalid problem type: {problem_type}")

    if problem_type not in config:
        raise ConfigurationError(f"No configuration for problem type: {problem_type}")

    problem_config = config[problem_type]

    if "algorithms" not in problem_config:
        raise ConfigurationError(f"No algorithms specified for {problem_type}")

    if "default_algorithms" not in problem_config:
        raise ConfigurationError(f"No default algorithms specified for {problem_type}")

    return True


def validate_feature_config(config: Dict[str, Any]) -> bool:
    """
    Validate feature engineering configuration

    Args:
      config (Dict[str, Any]): Config in dictionary format

    Returns:
      True if the config is valid, false otherwise

    Raises:
      ConfigurationError: If there's missing feature engineering section
    """

    required_sections = ["numerical", "categorical", "text", "time_series"]
    for section in required_sections:
        if section not in config:
            raise ConfigurationError(f"Missing feature engineering section: {section}")

    return True


# Export main classes and functions
__all__ = [
    "SparkForgeConfig",
    "ConfigurationError",
    "get_config",
    "set_config",
    "validate_model_config",
    "validate_feature_config",
]
