from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Set

from presidio_analyzer import Pattern

logger = logging.getLogger(__name__)


def _convert_json_to_patterns(config: Dict) -> Dict[str, List[Pattern]]:
    """Convert JSON configuration to Pattern objects."""
    patterns_dict = {}
    for entity_type, patterns_config in config.items():
        pattern_list = []
        for pattern_config in patterns_config:
            pattern_list.append(
                Pattern(
                    name=pattern_config.get("name", "unnamed"),
                    regex=pattern_config.get("pattern", ""),
                    score=float(pattern_config.get("score", 0.85))
                )
            )
        patterns_dict[entity_type] = pattern_list
    return patterns_dict


class ConfigCache:
    """Lightweight config cache with mtime-based invalidation."""

    _instance: Optional["ConfigCache"] = None

    def __init__(self) -> None:
        default_dir = Path(__file__).resolve().parent / "config"
        self._config_dir = Path(os.getenv("CONFIG_DIR", default_dir)).resolve()
        self._config_dir.mkdir(parents=True, exist_ok=True)

        self._default_blocklist: Optional[List[str]] = None
        self._default_blocklist_mtime: Optional[float] = None
        self._default_grantlist: Optional[List[str]] = None
        self._default_grantlist_mtime: Optional[float] = None
        self._default_regex_patterns: Optional[Dict[str, List[Pattern]]] = None
        self._default_regex_mtime: Optional[float] = None

        self._profile_blocklists: Dict[str, Optional[Set[str]]] = {}
        self._profile_blocklists_mtime: Dict[str, Optional[float]] = {}
        self._profile_grantlists: Dict[str, Optional[Set[str]]] = {}
        self._profile_grantlists_mtime: Dict[str, Optional[float]] = {}
        self._profile_regex_patterns: Dict[str, Optional[Dict[str, List[Pattern]]]] = {}
        self._profile_regex_mtime: Dict[str, Optional[float]] = {}

    @classmethod
    def instance(cls) -> "ConfigCache":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (test-only helper)."""
        cls._instance = None

    @property
    def config_dir(self) -> str:
        return str(self._config_dir)

    def invalidate_all(self) -> None:
        self._default_blocklist = None
        self._default_blocklist_mtime = None
        self._default_grantlist = None
        self._default_grantlist_mtime = None
        self._default_regex_patterns = None
        self._default_regex_mtime = None
        self._profile_blocklists.clear()
        self._profile_blocklists_mtime.clear()
        self._profile_grantlists.clear()
        self._profile_grantlists_mtime.clear()
        self._profile_regex_patterns.clear()
        self._profile_regex_mtime.clear()
        logger.info("All caches invalidated")

    def notify_path_changed(self, path: Optional[str]) -> None:
        if not path:
            return
        path_obj = Path(path).resolve()
        try:
            rel_path = path_obj.relative_to(self._config_dir)
        except ValueError:
            return

        filename = rel_path.name
        profile = rel_path.parts[0] if len(rel_path.parts) > 1 else None

        if profile:
            self._profile_blocklists.pop(profile, None)
            self._profile_blocklists_mtime.pop(profile, None)
            self._profile_grantlists.pop(profile, None)
            self._profile_grantlists_mtime.pop(profile, None)
            self._profile_regex_patterns.pop(profile, None)
            self._profile_regex_mtime.pop(profile, None)
            logger.debug("Invalidated profile '%s' cache", profile)
        else:
            if filename == "blocklist.txt":
                self._default_blocklist = None
                self._default_blocklist_mtime = None
                logger.debug("Invalidated default blocklist cache")
            if filename == "grantlist.txt":
                self._default_grantlist = None
                self._default_grantlist_mtime = None
                logger.debug("Invalidated default grantlist cache")
            if filename == "regex_patterns.json":
                self._default_regex_patterns = None
                self._default_regex_mtime = None
                logger.debug("Invalidated default regex patterns cache")

    def get_default_blocklist(self) -> List[str]:
        path = self._config_dir / "blocklist.txt"
        if self._needs_reload(path, self._default_blocklist_mtime, self._default_blocklist):
            self._default_blocklist = self._read_list_file(path)
            self._default_blocklist_mtime = self._safe_mtime(path)
        return self._default_blocklist or []

    def get_default_grantlist(self) -> List[str]:
        path = self._config_dir / "grantlist.txt"
        if self._needs_reload(path, self._default_grantlist_mtime, self._default_grantlist):
            self._default_grantlist = self._read_list_file(path)
            self._default_grantlist_mtime = self._safe_mtime(path)
        return self._default_grantlist or []

    def get_default_regex_patterns(self) -> Dict[str, List[Pattern]]:
        path = self._config_dir / "regex_patterns.json"
        if self._needs_reload(path, self._default_regex_mtime, self._default_regex_patterns):
            self._default_regex_patterns = self._read_regex_patterns(path)
            self._default_regex_mtime = self._safe_mtime(path)
        return self._default_regex_patterns or {}

    def get_profile_blocklist(self, profile: str) -> Set[str]:
        path = self._config_dir / profile / "blocklist.txt"
        mtime = self._safe_mtime(path)
        cached = self._profile_blocklists.get(profile)
        if cached is None or self._profile_blocklists_mtime.get(profile) != mtime:
            self._profile_blocklists[profile] = set(self._read_list_file(path))
            self._profile_blocklists_mtime[profile] = mtime
        return self._profile_blocklists.get(profile) or set()

    def get_profile_grantlist(self, profile: str) -> Set[str]:
        path = self._config_dir / profile / "grantlist.txt"
        mtime = self._safe_mtime(path)
        cached = self._profile_grantlists.get(profile)
        if cached is None or self._profile_grantlists_mtime.get(profile) != mtime:
            self._profile_grantlists[profile] = set(self._read_list_file(path))
            self._profile_grantlists_mtime[profile] = mtime
        return self._profile_grantlists.get(profile) or set()

    def get_profile_regex_patterns(self, profile: str) -> Dict[str, List[Pattern]]:
        path = self._config_dir / profile / "regex_patterns.json"
        mtime = self._safe_mtime(path)
        cached = self._profile_regex_patterns.get(profile)
        if cached is None or self._profile_regex_mtime.get(profile) != mtime:
            self._profile_regex_patterns[profile] = self._read_regex_patterns(path)
            self._profile_regex_mtime[profile] = mtime
        return self._profile_regex_patterns.get(profile) or {}

    @staticmethod
    def _read_list_file(file_path: Path) -> List[str]:
        if not file_path.exists():
            logger.debug("List file not found: %s", file_path)
            return []
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return [
                    line.strip().lower()
                    for line in file
                    if line.strip() and not line.strip().startswith("#")
                ]
        except IOError as exc:
            logger.error("Failed to read list file %s: %s", file_path, exc)
            return []

    def _read_regex_patterns(self, file_path: Path) -> Dict[str, List[Pattern]]:
        if not file_path.exists():
            logger.debug("Regex pattern file not found: %s", file_path)
            return {}
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                config = json.load(file)
            return _convert_json_to_patterns(config)
        except (json.JSONDecodeError, IOError, ValueError) as exc:
            logger.error("Failed to read regex patterns from %s: %s", file_path, exc)
            return {}

    @staticmethod
    def _needs_reload(path: Path, cached_mtime: Optional[float], cached_value: Optional[object]) -> bool:
        mtime = ConfigCache._safe_mtime(path)
        return cached_value is None or mtime != cached_mtime

    @staticmethod
    def _safe_mtime(path: Path) -> Optional[float]:
        try:
            return path.stat().st_mtime
        except FileNotFoundError:
            return None
