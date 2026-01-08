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
import re
from pathlib import Path
from typing import List, Dict, Set

from presidio_analyzer import Pattern

from text_anonymizer.config_cache import ConfigCache

logger = logging.getLogger(__name__)


class InvalidProfileNameError(ValueError):
    """Raised when a profile name is invalid or contains path traversal attempts."""
    pass


class ProfileConfigProvider:
    """Manages profile-based configurations for anonymization."""

    # Maximum allowed length for profile names
    MAX_PROFILE_NAME_LENGTH = 50

    def __init__(self):
        """Initialize the profile configuration manager."""
        self._profiles_cache: Dict[str, Dict] = {}

    @staticmethod
    def validate_profile_name(profile_name: str) -> str:
        """
        Validate and sanitize profile name to prevent path traversal attacks.

        Security checks:
        - Rejects empty or None profile names
        - Only allows alphanumeric characters, hyphens, and underscores
        - Prevents path traversal attempts (.., /, \\)
        - Enforces maximum length limit

        :param profile_name: Profile name from user input
        :return: Validated profile name
        :raises InvalidProfileNameError: If profile name is invalid or contains path traversal attempts
        """
        if not profile_name or not isinstance(profile_name, str):
            raise InvalidProfileNameError("Profile name cannot be empty or None")

        # Strip whitespace
        profile_name = profile_name.strip()

        # Check length
        if len(profile_name) > ProfileConfigProvider.MAX_PROFILE_NAME_LENGTH:
            raise InvalidProfileNameError(
                f"Profile name too long (max {ProfileConfigProvider.MAX_PROFILE_NAME_LENGTH} characters): {profile_name}"
            )

        # Only allow alphanumeric characters, hyphens, and underscores
        if not re.match(r'^[a-zA-Z0-9_-]+$', profile_name):
            raise InvalidProfileNameError(
                f"Invalid profile name: '{profile_name}'. "
                "Only alphanumeric characters, hyphens, and underscores are allowed."
            )

        # Explicitly check for path traversal attempts
        if '..' in profile_name or '/' in profile_name or '\\' in profile_name:
            raise InvalidProfileNameError(
                f"Profile name cannot contain path separators or '..' sequences: {profile_name}"
            )

        # Prevent hidden files/directories (starting with .)
        if profile_name.startswith('.'):
            raise InvalidProfileNameError(
                f"Profile name cannot start with '.': {profile_name}"
            )

        return profile_name

    @staticmethod
    def get_safe_profile_path(profile_name: str, base_config_dir: str) -> Path:
        """
        Safely construct and validate profile directory path.

        This method ensures that the resulting path is within the base config directory
        and cannot escape through path traversal attacks.

        :param profile_name: Profile name (will be validated)
        :param base_config_dir: Base configuration directory
        :return: Absolute path to profile directory
        :raises InvalidProfileNameError: If profile name is invalid or path escapes base directory
        """
        # Validate the profile name first
        validated_profile = ProfileConfigProvider.validate_profile_name(profile_name)

        # Resolve base path to absolute path
        base_path = Path(base_config_dir).resolve()

        # Construct profile path
        profile_path = (base_path / validated_profile).resolve()

        # Critical security check: ensure profile path is within base directory
        try:
            profile_path.relative_to(base_path)
        except ValueError:
            raise InvalidProfileNameError(
                f"Profile path escapes base directory: {profile_name}"
            )

        return profile_path

    @staticmethod
    def get_profile_config_dir(profile_name: str) -> str:
        """
        Get the path to the profile-specific config directory.

        Uses CONFIG_DIR environment variable if set, otherwise defaults to text_anonymizer/config.
        Returns the path: {config_dir}/{profile_name}

        SECURITY: Profile directories are NOT auto-created. They must exist and be created by administrators.
        This prevents attackers from creating arbitrary directories through user input.

        :param profile_name: Name of the profile (e.g., "palautteet")
        :return: Path to profile-specific config directory
        :raises InvalidProfileNameError: If profile name is invalid or contains path traversal attempts
        :raises FileNotFoundError: If profile directory does not exist
        """
        # Default should point to text_anonymizer/config (same directory as this file's parent)
        config_dir = os.getenv("CONFIG_DIR", os.path.join(os.path.dirname(__file__), "config"))

        # Use the safe path construction method with validation
        profile_path = ProfileConfigProvider.get_safe_profile_path(profile_name, config_dir)

        # Security: Do NOT auto-create directories from user input
        # Profile directories should be created by administrators only
        if not profile_path.exists():
            logger.warning("Profile directory does not exist: %s", profile_name)
            raise FileNotFoundError(
                f"Profile '{profile_name}' not found. Profile directories must be created by administrators."
            )

        if not profile_path.is_dir():
            raise InvalidProfileNameError(
                f"Profile path exists but is not a directory: {profile_name}"
            )

        return str(profile_path)

    def list_profiles(self) -> List[str]:
        """
        List all available profiles.

        Only returns valid profile subdirectories that:
        - Are actual directories
        - Don't start with '_' (special/private directories)
        - Don't start with '.' (hidden directories)
        - Pass validation (alphanumeric, hyphens, underscores only)

        :return: List of valid profile names
        """
        config_dir = os.getenv("CONFIG_DIR", os.path.join(os.path.dirname(__file__), "config"))
        profiles = []

        if os.path.exists(config_dir):
            for item in os.listdir(config_dir):
                # Skip hidden and private directories
                if item.startswith("_") or item.startswith("."):
                    continue

                item_path = os.path.join(config_dir, item)

                # Only include directories
                if not os.path.isdir(item_path):
                    continue

                # Validate the profile name for security
                try:
                    ProfileConfigProvider.validate_profile_name(item)
                    profiles.append(item)
                except InvalidProfileNameError:
                    logger.warning("Skipping invalid profile directory: %s", item)
                    continue

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
