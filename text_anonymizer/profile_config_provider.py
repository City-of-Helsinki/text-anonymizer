"""
Profile-Based Configuration Manager.

This module loads profile-based configurations including:
- Custom regex patterns
- Block lists (words to anonymize)
- Grant lists (words to keep)

Each profile can have its own set of configurations to customize anonymization
behavior for different material types and use cases.
"""

import os
import json
import logging
from typing import List, Dict, Set

from presidio_analyzer import Pattern
from text_anonymizer.regex_config_provider import _convert_json_to_patterns

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
        profile_dir = self.get_profile_config_dir(profile_name)
        blocklist_file = os.path.join(profile_dir, f"blocklist.txt")

        if not os.path.exists(blocklist_file):
            logger.debug(f"Block list not found for profile: {profile_name}")
            return set()

        try:
            with open(blocklist_file, 'r', encoding='utf-8') as file:
                items = {line.strip().lower() for line in file if line.strip() and not line.startswith("#")}

            logger.info(f"Loaded {len(items)} items from block list: {profile_name}")
            return items
        except IOError as e:
            logger.error(f"Error loading block list: {str(e)}")
            return set()



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
        profile_dir = self.get_profile_config_dir(profile_name)
        grantlist_file = os.path.join(profile_dir, f"grantlist.txt")

        if not os.path.exists(grantlist_file):
            logger.debug(f"Grant list not found for profile: {profile_name}")
            return set()

        try:
            with open(grantlist_file, 'r', encoding='utf-8') as file:
                items = {line.strip().lower() for line in file if line.strip() and not line.startswith("#")}

            logger.info(f"Loaded {len(items)} items from grant list: {profile_name}")
            return items
        except IOError as e:
            logger.error(f"Error loading grant list: {str(e)}")
            return set()


    # ============================================================================
    # Regex Pattern
    # ============================================================================

    def load_profile_regex_patterns(self, profile_name: str) -> Dict[str, List[Pattern]]:
        """
        Load regex patterns for a profile.

        :param profile_name: Name of the profile
        :return: Dictionary mapping entity types to Pattern lists
        """
        profile_dir = self.get_profile_config_dir(profile_name)
        regex_file = os.path.join(profile_dir, "regex_patterns.json")

        if not os.path.exists(regex_file):
            logger.debug(f"Regex patterns not found for profile: {profile_name}")
            return {}

        try:
            with open(regex_file, 'r', encoding='utf-8') as file:
                config = json.load(file)

            patterns = _convert_json_to_patterns(config)
            logger.info(f"Loaded {len(patterns)} entity types from regex patterns for profile: {profile_name}")
            return patterns
        except (json.JSONDecodeError, IOError, KeyError, ValueError) as e:
            logger.error(f"Error loading regex patterns for profile {profile_name}: {str(e)}")
            return {}



