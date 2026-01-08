#!/usr/bin/env python
"""
Comprehensive Profile Integration Tests.

This test suite combines:
1. Profile functionality tests (regex patterns, caching, NLP engine reuse)
2. Security validation tests (path traversal prevention, input validation)

Verifies that:
- Profile-specific regex patterns are loaded correctly
- Regex recognizers are added to profile registries
- Pattern matching works as expected
- NLP engine is reused (memory efficient)
- Profile names are validated to prevent path traversal attacks
- Invalid profile names are rejected
- Non-existent profiles don't break anonymization
"""

import sys
import os
import tempfile
import unittest
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

import logging

# Configure logging to see debug messages
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

from text_anonymizer import TextAnonymizer
from text_anonymizer.profile_config_provider import ProfileConfigProvider, InvalidProfileNameError
from text_anonymizer.config_cache import ConfigCache


# ============================================================================
# PROFILE FUNCTIONALITY TESTS
# ============================================================================

class TestProfileRegexPatterns(unittest.TestCase):
    """Test profile-specific regex pattern loading and detection."""

    def setUp(self):
        """Initialize test fixtures."""
        self.anonymizer = TextAnonymizer(debug_mode=False)

    def test_profile_regex_patterns_loaded(self):
        """Test that profile regex patterns are loaded correctly."""
        profile_manager = ProfileConfigProvider()
        regex_patterns = profile_manager.load_profile_regex_patterns("example")

        self.assertIsNotNone(regex_patterns)
        self.assertIsInstance(regex_patterns, dict)
        logger.info(f"Loaded regex patterns for profile 'example': {list(regex_patterns.keys())}")

    def test_example_pattern_detection(self):
        """Test that EXAMPLE entity patterns are detected."""
        test_cases = [
            ("ABCxyz123", "Simple pattern without context"),
            ("Please process ABCxyz123 immediately.", "Pattern in context"),
            ("ABCxyz123 department", "Pattern with trailing word"),
        ]

        for text, description in test_cases:
            with self.subTest(text=text, description=description):
                result = self.anonymizer.anonymize(text, profile="example")

                self.assertIsNotNone(result)
                self.assertIsNotNone(result.anonymized_text)
                logger.info(f"{description}: '{text}' -> '{result.anonymized_text}'")


class TestProfileCaching(unittest.TestCase):
    """Test profile analyzer caching."""

    def setUp(self):
        """Initialize test fixtures."""
        self.anonymizer = TextAnonymizer(debug_mode=False)

    def test_profile_analyzers_are_cached(self):
        """Verify that profile analyzers are cached."""
        # First call to profile
        analyzer1 = self.anonymizer._get_analyzer_for_profile("example")

        # Second call to same profile
        analyzer2 = self.anonymizer._get_analyzer_for_profile("example")

        # Verify they're the same object (cached)
        self.assertIs(analyzer1, analyzer2,
                     "Profile analyzers should be cached and return same object")
        logger.info(f"Profile analyzer cached correctly (ID: {id(analyzer1)})")

    def test_nlp_engine_reuse(self):
        """Verify that NLP engine is reused across profile analyzers."""
        # Get default analyzer NLP engine
        default_nlp = self.anonymizer.nlp_engine

        # Get profile-specific analyzer
        profile_analyzer = self.anonymizer._get_analyzer_for_profile("example")
        profile_nlp = profile_analyzer.nlp_engine

        # Verify NLP engine is the same object (reused)
        self.assertIs(default_nlp, profile_nlp,
                     "NLP engine should be reused (same object) for memory efficiency")
        logger.info(f"NLP engine reused correctly (ID: {id(default_nlp)})")

    def test_profile_registry_has_recognizers(self):
        """Verify that profile registry contains recognizers."""
        profile_analyzer = self.anonymizer._get_analyzer_for_profile("example")
        profile_registry = profile_analyzer.registry

        self.assertIsNotNone(profile_registry)

        # Get all recognizers
        all_recognizers = profile_registry.recognizers
        self.assertGreater(len(all_recognizers), 0,
                          "Profile registry should have at least one recognizer")
        logger.info(f"Profile registry has {len(all_recognizers)} recognizers")


# ============================================================================
# SECURITY VALIDATION TESTS
# ============================================================================

class TestProfileNameValidation(unittest.TestCase):
    """Test profile name security validation."""

    def test_valid_profile_names_accepted(self):
        """Test that valid profile names are accepted."""
        valid_names = [
            "example",
            "palautteet",
            "asiakaspalvelu",
            "test-profile",
            "test_profile",
            "Profile123",
            "a",
            "a" * 50,  # Max length
        ]

        for name in valid_names:
            with self.subTest(name=name):
                result = ProfileConfigProvider.validate_profile_name(name)
                self.assertEqual(result, name.strip())
                logger.debug(f"Valid profile name accepted: '{name}'")

    def test_invalid_profile_names_rejected(self):
        """Test that invalid profile names are rejected."""
        invalid_names = [
            ("", "Empty string"),
            ("   ", "Whitespace only"),
            (".hidden", "Starts with dot"),
            ("..parent", "Path traversal (..)"),
            ("../../../etc/passwd", "Path traversal attack"),
            ("profile/subdir", "Contains slash"),
            ("profile\\subdir", "Contains backslash"),
            ("profile..name", "Contains .."),
            ("a" * 51, "Too long"),
            ("profile name", "Contains space"),
            ("profile@name", "Contains @"),
            ("profile$name", "Contains $"),
            ("profile#name", "Contains #"),
        ]

        for name, description in invalid_names:
            with self.subTest(name=name, description=description):
                with self.assertRaises(InvalidProfileNameError,
                                      msg=f"{description} should raise InvalidProfileNameError"):
                    ProfileConfigProvider.validate_profile_name(name)
                logger.debug(f"Invalid profile name rejected: {description}")

    def test_whitespace_trimming(self):
        """Test that whitespace is trimmed from profile names."""
        result = ProfileConfigProvider.validate_profile_name("  example  ")
        self.assertEqual(result, "example")


class TestPathTraversalPrevention(unittest.TestCase):
    """Test path traversal attack prevention."""

    def test_path_traversal_attempts_blocked(self):
        """Test that path traversal attempts are blocked."""
        traversal_attempts = [
            ("..", "Parent directory"),
            ("../", "Parent directory with slash"),
            ("../../", "Multiple parent directories"),
            ("../../../etc/passwd", "Unix path traversal"),
            ("..\\..\\windows\\system32", "Windows path traversal"),
            ("valid../bad", "Hidden traversal"),
            ("valid/../../bad", "Mixed traversal"),
        ]

        for attempt, description in traversal_attempts:
            with self.subTest(attempt=attempt, description=description):
                with self.assertRaises(InvalidProfileNameError,
                                      msg=f"{description} should be blocked"):
                    ProfileConfigProvider.validate_profile_name(attempt)
                logger.debug(f"Path traversal blocked: {description}")

    def test_safe_profile_path_construction(self):
        """Test that safe profile path construction prevents escaping."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a valid profile directory
            profile_dir = Path(tmpdir) / "valid_profile"
            profile_dir.mkdir()

            # Test valid profile
            safe_path = ProfileConfigProvider.get_safe_profile_path("valid_profile", tmpdir)

            self.assertEqual(safe_path, profile_dir.resolve())

            # Verify path is within tmpdir
            self.assertTrue(str(safe_path).startswith(str(Path(tmpdir).resolve())),
                          "Profile path should be within base directory")

    def test_safe_profile_path_prevents_escape(self):
        """Test that safe path construction prevents directory escape."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Attempt to escape should be blocked by validation
            with self.assertRaises(InvalidProfileNameError):
                ProfileConfigProvider.get_safe_profile_path("../escape", tmpdir)


class TestProfileListFiltering(unittest.TestCase):
    """Test that list_profiles excludes hidden and invalid directories."""

    def test_list_profiles_excludes_invalid_directories(self):
        """Test that list_profiles excludes hidden and invalid directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            old_config_dir = os.environ.get("CONFIG_DIR")
            os.environ["CONFIG_DIR"] = tmpdir

            try:
                # Create various directories
                (Path(tmpdir) / "valid_profile").mkdir()
                (Path(tmpdir) / ".hidden").mkdir()
                (Path(tmpdir) / "_private").mkdir()
                (Path(tmpdir) / "another-valid").mkdir()
                (Path(tmpdir) / "test_profile").mkdir()
                (Path(tmpdir) / "invalid@profile").mkdir()

                # Create a file (should be ignored)
                (Path(tmpdir) / "not_a_dir.txt").touch()

                provider = ProfileConfigProvider()
                profiles = provider.list_profiles()

                # Should include valid profiles
                self.assertIn("valid_profile", profiles)
                self.assertIn("another-valid", profiles)
                self.assertIn("test_profile", profiles)

                # Should exclude hidden, private, invalid, and files
                self.assertNotIn(".hidden", profiles)
                self.assertNotIn("_private", profiles)
                self.assertNotIn("invalid@profile", profiles)
                self.assertNotIn("not_a_dir.txt", profiles)

                logger.info(f"Profile list correctly filtered: {profiles}")

            finally:
                if old_config_dir is None:
                    os.environ.pop("CONFIG_DIR", None)
                else:
                    os.environ["CONFIG_DIR"] = old_config_dir


class TestNonexistentProfileHandling(unittest.TestCase):
    """Test handling of non-existent and invalid profiles."""

    def test_nonexistent_profile_returns_empty_collections(self):
        """Test that non-existent but valid profiles don't break anonymization."""
        cache = ConfigCache.instance()

        # Test with valid but non-existent profile name
        blocklist = cache.get_profile_blocklist("nonexistent_valid_profile")
        grantlist = cache.get_profile_grantlist("nonexistent_valid_profile")
        regex_patterns = cache.get_profile_regex_patterns("nonexistent_valid_profile")

        # Should return empty collections, not raise exceptions
        self.assertEqual(len(blocklist), 0, "Non-existent profile should return empty blocklist")
        self.assertEqual(len(grantlist), 0, "Non-existent profile should return empty grantlist")
        self.assertEqual(len(regex_patterns), 0, "Non-existent profile should return empty regex patterns")

        logger.info("Non-existent profile handled gracefully with empty collections")

    def test_invalid_profile_name_falls_back_to_default(self):
        """Test that invalid profile names fall back to default analyzer without crashing."""
        anonymizer = TextAnonymizer(debug_mode=False)

        # Test with path traversal attempt
        text = "Puhelinnumero 040-1234567"
        result = anonymizer.anonymize(text, profile="../example")

        # Should NOT raise an exception, should use default analyzer
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.anonymized_text)

        # Should anonymize using default recognizers (phone number should be detected)
        self.assertIn("<PUHELIN>", result.anonymized_text)

        logger.info("Invalid profile name handled gracefully - fell back to default analyzer")

    def test_invalid_profile_raises_exception_in_cache(self):
        """Test that ConfigCache still validates profile names (for direct cache access)."""
        cache = ConfigCache.instance()

        # Direct cache access should still raise InvalidProfileNameError
        with self.assertRaises(InvalidProfileNameError,
                              msg="Invalid profile name should raise InvalidProfileNameError in cache"):
            cache.get_profile_blocklist("../../../etc/passwd")

        logger.info("Invalid profile name properly rejected in direct cache access")


class TestProfileConfigDirectory(unittest.TestCase):
    """Test profile config directory behavior."""

    def test_get_profile_config_dir_requires_existing_directory(self):
        """Test that get_profile_config_dir doesn't auto-create directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            old_config_dir = os.environ.get("CONFIG_DIR")
            os.environ["CONFIG_DIR"] = tmpdir

            try:
                # Attempting to get a non-existent profile should raise FileNotFoundError
                with self.assertRaises(FileNotFoundError,
                                      msg="Non-existent profile directory should raise FileNotFoundError"):
                    ProfileConfigProvider.get_profile_config_dir("nonexistent")

                # Verify the directory was NOT created
                self.assertFalse((Path(tmpdir) / "nonexistent").exists(),
                               "Profile directory should NOT be auto-created")

            finally:
                if old_config_dir is None:
                    os.environ.pop("CONFIG_DIR", None)
                else:
                    os.environ["CONFIG_DIR"] = old_config_dir

    def test_get_profile_config_dir_returns_existing_directory(self):
        """Test that get_profile_config_dir returns existing directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            old_config_dir = os.environ.get("CONFIG_DIR")
            os.environ["CONFIG_DIR"] = tmpdir

            try:
                # Create a profile directory
                profile_dir = Path(tmpdir) / "existing_profile"
                profile_dir.mkdir()

                # Should successfully return the path
                result = ProfileConfigProvider.get_profile_config_dir("existing_profile")
                self.assertEqual(result, str(profile_dir.resolve()))

            finally:
                if old_config_dir is None:
                    os.environ.pop("CONFIG_DIR", None)
                else:
                    os.environ["CONFIG_DIR"] = old_config_dir


# ============================================================================
# TEST SUITE
# ============================================================================

def suite():
    """Create test suite."""
    test_suite = unittest.TestSuite()

    # Profile functionality tests
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestProfileRegexPatterns))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestProfileCaching))

    # Security validation tests
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestProfileNameValidation))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestPathTraversalPrevention))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestProfileListFiltering))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestNonexistentProfileHandling))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestProfileConfigDirectory))

    return test_suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite())
    sys.exit(0 if result.wasSuccessful() else 1)

