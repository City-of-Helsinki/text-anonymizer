import os
import json
import logging
from typing import List, Dict, Optional

from presidio_analyzer import Pattern

logger = logging.getLogger(__name__)

# Module-level cache for regex patterns
_regex_patterns_cache: Optional[Dict[str, List[Pattern]]] = None


def get_config_file_path(file_name: str) -> str:
    """Get the path to a configuration file in the config directory."""
    config_dir = os.getenv("CONFIG_DIR", os.path.join(os.path.dirname(__file__), "..", "config"))
    return os.path.join(config_dir, file_name)


def _convert_json_to_patterns(config: Dict) -> Dict[str, List[Pattern]]:
    """Convert JSON configuration to Pattern objects.

    Expected JSON format:
    {
        "entity_type_1": [
            {
                "name": "pattern_name",
                "pattern": "regex_pattern",
                "score": 0.85
            },
            ...
        ],
        "entity_type_2": [...]
    }

    :param config: Dictionary loaded from JSON
    :return: Dictionary mapping entity types to lists of Pattern objects
    :raises ValueError: If pattern configuration is invalid
    """
    patterns_dict = {}

    for entity_type, patterns_config in config.items():
        patterns_list = []
        for pattern_config in patterns_config:
            pattern = Pattern(
                name=pattern_config.get("name", "unnamed"),
                regex=pattern_config.get("pattern", ""),
                score=float(pattern_config.get("score", 0.85))
            )
            patterns_list.append(pattern)
        patterns_dict[entity_type] = patterns_list

    return patterns_dict


def load_regex_patterns_from_json(file_path: str) -> Dict[str, List[Pattern]]:
    """Load regex patterns from JSON configuration file.

    Expected JSON format:
    {
        "entity_type_1": [
            {
                "name": "pattern_name",
                "pattern": "regex_pattern",
                "score": 0.85
            },
            ...
        ],
        "entity_type_2": [...]
    }

    :param file_path: Path to the JSON configuration file
    :return: Dictionary mapping entity types to lists of Pattern objects
    """
    patterns_dict = {}

    if not os.path.exists(file_path):
        logger.debug(f"Regex patterns file not found: {file_path}")
        return patterns_dict

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            config = json.load(file)

        patterns_dict = _convert_json_to_patterns(config)
        logger.info(f"Loaded {len(patterns_dict)} entity types from regex patterns file")

    except (json.JSONDecodeError, IOError, KeyError, ValueError) as e:
        logger.error(f"Error loading regex patterns from {file_path}: {str(e)}")

    return patterns_dict


def get_regex_patterns(use_cache: bool = True) -> Dict[str, List[Pattern]]:
    """Get all configured regex patterns with optional caching.

    :param use_cache: If True, use cached patterns. If False, reload from file.
    :return: Dictionary mapping entity types to lists of Pattern objects
    """
    global _regex_patterns_cache

    if use_cache and _regex_patterns_cache is not None:
        logger.debug("Using cached regex patterns")
        return _regex_patterns_cache

    config_file = get_config_file_path("regex_patterns.json")
    patterns = load_regex_patterns_from_json(config_file)
    _regex_patterns_cache = patterns

    return patterns


def clear_regex_patterns_cache() -> None:
    """Clear the regex patterns cache to force reload on next access."""
    global _regex_patterns_cache
    _regex_patterns_cache = None
    logger.info("Regex patterns cache cleared")

