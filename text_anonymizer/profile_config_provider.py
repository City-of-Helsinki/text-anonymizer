"""
Profile-Based Configuration Manager.

This module loads profile-based configurations including:
- Custom regex patterns
- Block lists (words to anonymize)
- Grant lists (words to keep)

Each profile can have its own set of configurations to customize anonymization
behavior for different material types and use cases.
"""

import logging
import os
from typing import List, Dict, Set

from presidio_analyzer import Pattern

from text_anonymizer.config_cache import ConfigCache

logger = logging.getLogger(__name__)


class ProfileConfigProvider:
    """Manages profile-based configurations for anonymization."""

    def __init__(self):
        """Initialize the profile configuration manager."""
        self._profiles_cache: Dict[str, Dict] = {}

    @staticmethod
    def get_profile_config_dir(profile_name: str) -> str:
        """
        Get the path to the profile-specific config directory.

        Uses CONFIG_DIR environment variable if set, otherwise defaults to text_anonymizer/config.
        Returns the path: {config_dir}/{profile_name}

        :param profile_name: Name of the profile (e.g., "palautteet")
        :return: Path to profile-specific config directory
        """

        # Default should point to text_anonymizer/config (same directory as this file's parent)
        config_dir = os.getenv("CONFIG_DIR", os.path.join(os.path.dirname(__file__), "config"))
        profile_dir = os.path.join(config_dir, profile_name)
        os.makedirs(profile_dir, exist_ok=True)
        return profile_dir

    def list_profiles(self) -> List[str]:
        """
        List all available profiles.

        :return: List of profile names
        """
        config_dir = os.getenv("CONFIG_DIR", os.path.join(os.path.dirname(__file__), "config"))
        profiles = []

        if os.path.exists(config_dir):
            for item in os.listdir(config_dir):
                item_path = os.path.join(config_dir, item)
                if os.path.isdir(item_path) and not item.startswith("_"):
                    profiles.append(item)

        return sorted(profiles)

    # ============================================================================
    # Block List Management
    # ============================================================================

    def load_profile_blocklist(self, profile_name: str) -> Set[str]:
        """
        Load block list for a profile.

        Block list contains words/phrases that should be anonymized.

        :param profile_name: Name of the profile
        :return: Set of blocked words/phrases
        """
        return ConfigCache.instance().get_profile_blocklist(profile_name)

    # ============================================================================
    # Grant List
    # ============================================================================

    def load_profile_grantlist(self, profile_name: str) -> Set[str]:
        """
        Load grant list for a profile.

        Grant list contains words/phrases that should NOT be anonymized.

        :param profile_name: Name of the profile
        :return: Set of granted words/phrases
        """
        return ConfigCache.instance().get_profile_grantlist(profile_name)

    # ============================================================================
    # Regex Pattern
    # ============================================================================

    def load_profile_regex_patterns(self, profile_name: str) -> Dict[str, List[Pattern]]:
        """
        Load regex patterns for a profile.

        :param profile_name: Name of the profile
        :return: Dictionary mapping entity types to Pattern lists
        """
        return ConfigCache.instance().get_profile_regex_patterns(profile_name)
