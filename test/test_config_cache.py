#!/usr/bin/env python
"""
Simple integration test for ConfigCache and dynamic reloading.

Verifies:
1. ConfigCache loads and caches lists/regex patterns
2. notify_path_changed() invalidates specific entries
3. invalidate_all() clears everything
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from text_anonymizer.config_cache import ConfigCache


def test_default_blocklist_loading():
    """Test loading default blocklist."""
    cache = ConfigCache.instance()
    blocklist = cache.get_default_blocklist()
    assert isinstance(blocklist, list), "Blocklist should be a list"
    print("✓ Default blocklist loaded")


def test_default_grantlist_loading():
    """Test loading default grantlist."""
    cache = ConfigCache.instance()
    grantlist = cache.get_default_grantlist()
    assert isinstance(grantlist, list), "Grantlist should be a list"
    print("✓ Default grantlist loaded")


def test_default_regex_patterns_loading():
    """Test loading default regex patterns."""
    cache = ConfigCache.instance()
    patterns = cache.get_default_regex_patterns()
    assert isinstance(patterns, dict), "Regex patterns should be a dict"
    print("✓ Default regex patterns loaded")


def test_cache_invalidation():
    """Test cache invalidation via notify_path_changed()."""
    cache = ConfigCache.instance()

    # Load blocklist
    blocklist1 = cache.get_default_blocklist()
    mtime1 = cache._default_blocklist_mtime

    # Invalidate
    cache.notify_path_changed(str(Path(cache.config_dir) / "blocklist.txt"))

    # Cache should be None
    assert cache._default_blocklist is None, "Blocklist should be invalidated"
    assert cache._default_blocklist_mtime is None, "Blocklist mtime should be None"
    print("✓ Cache invalidation works")


def test_profile_blocklist_loading():
    """Test loading profile-specific blocklist."""
    cache = ConfigCache.instance()
    blocklist = cache.get_profile_blocklist("example")
    assert isinstance(blocklist, set), "Profile blocklist should be a set"
    print("✓ Profile blocklist loaded")


def test_profile_grantlist_loading():
    """Test loading profile-specific grantlist."""
    cache = ConfigCache.instance()
    grantlist = cache.get_profile_grantlist("example")
    assert isinstance(grantlist, set), "Profile grantlist should be a set"
    print("✓ Profile grantlist loaded")


def test_invalidate_all():
    """Test invalidating all caches."""
    cache = ConfigCache.instance()

    # Load everything
    _ = cache.get_default_blocklist()
    _ = cache.get_default_grantlist()
    _ = cache.get_default_regex_patterns()
    _ = cache.get_profile_blocklist("example")

    # Invalidate all
    cache.invalidate_all()

    assert cache._default_blocklist is None
    assert cache._default_grantlist is None
    assert cache._default_regex_patterns is None
    assert len(cache._profile_blocklists) == 0
    assert len(cache._profile_grantlists) == 0
    print("✓ Invalidate all works")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("ConfigCache Integration Tests")
    print("="*70 + "\n")

    try:
        test_default_blocklist_loading()
        test_default_grantlist_loading()
        test_default_regex_patterns_loading()
        test_cache_invalidation()
        test_profile_blocklist_loading()
        test_profile_grantlist_loading()
        test_invalidate_all()

        print("\n" + "="*70)
        print("All tests passed!")
        print("="*70 + "\n")
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

