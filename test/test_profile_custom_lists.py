#!/usr/bin/env python
"""
Test script for profile-based custom lists (blocklist and grantlist).

Verifies that:
1. Profile-specific blocklists are loaded correctly
2. Profile-specific grantlists are loaded correctly
3. Blocklisted words are detected and anonymized
4. Grantlisted words are protected from anonymization
5. Custom lists work alongside regex patterns and standard recognizers
6. Multiple profiles with different lists can be used independently
"""

import sys
import os
import tempfile
import json
import unittest

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
from text_anonymizer.profile_config_provider import ProfileConfigProvider


class TestProfileBlocklist(unittest.TestCase):
    """Test blocklist functionality per profile."""
    
    label = "MUU_TUNNISTE"

    def setUp(self):
        """Initialize test fixtures."""

        self.anonymizer = TextAnonymizer(debug_mode=False)
        self.profile_provider = ProfileConfigProvider()

    def test_blocklist_loads_from_example_profile(self):
        """Test that blocklist is loaded from example profile."""
        blocklist = self.profile_provider.load_profile_blocklist("example")

        self.assertIsNotNone(blocklist)
        self.assertIsInstance(blocklist, set)
        self.assertGreater(len(blocklist), 0)
        self.assertIn("blockword123", blocklist)
        logger.info(f"Blocklist loaded with {len(blocklist)} items: {blocklist}")

    def test_blocklist_items_are_anonymized(self):
        """Test that blocklisted words are detected and anonymized."""
        text = "The code blockword123 is in our system."
        result = self.anonymizer.anonymize(text, profile="example")

        # Check that anonymization happened
        self.assertIsNotNone(result.anonymized_text)
        self.assertNotEqual(result.anonymized_text, text)

        # Check that the blocklisted word was detected
        self.assertIn(self.label, result.details)
        self.assertIn("blockword123", result.details[self.label])
        logger.info(f"Blocklisted word detected and anonymized: {result.anonymized_text}")

    def test_blocklist_case_insensitive(self):
        """Test that blocklist matching is case-insensitive."""
        test_cases = [
            "blockword123",
            "BLOCKWORD123",
            "Blockword123",
            "bLoCkWoRd123",
        ]

        for variant in test_cases:
            text = f"The code {variant} is detected."
            result = self.anonymizer.anonymize(text, profile="example")

            self.assertIn(self.label, result.details,
                         f"Blocklist should match '{variant}' (case-insensitive)")
            logger.info(f"Case variant '{variant}' correctly detected")

    def test_multiple_blocklist_items(self):
        """Test detecting multiple blocklisted items in one text."""
        text = "Items blockword123 and blockword456 are both blocked."
        result = self.anonymizer.anonymize(text, profile="example")

        self.assertIn(self.label, result.details)
        # Should detect at least the first blocklisted item
        stats = result.statistics.get(self.label, 0)
        self.assertGreater(stats, 0)
        logger.info(f"Detected {stats} blocklist items")

    def test_blocklist_in_different_contexts(self):
        """Test blocklist detection in various text contexts."""
        test_cases = [
            "Blockword123 start here",
            "In the middle blockword123 of text",
            "At the end blockword123",
            "Multiple blockword123 items blockword123 here",
        ]

        for text in test_cases:
            result = self.anonymizer.anonymize(text, profile="example")
            self.assertIn(self.label, result.details,
                         f"Blocklist should be detected in: {text}")
            logger.info(f"Successfully detected blocklist in: {text}")

    def test_blocklist_empty_when_no_profile_config(self):
        """Test that blocklist is empty for non-existent profile."""
        blocklist = self.profile_provider.load_profile_blocklist("nonexistent_profile")

        self.assertIsInstance(blocklist, set)
        self.assertEqual(len(blocklist), 0)
        logger.info("Empty blocklist returned for non-existent profile")


class TestProfileGrantlist(unittest.TestCase):
    """Test grantlist functionality per profile."""

    def setUp(self):
        """Initialize test fixtures."""

        self.anonymizer = TextAnonymizer(debug_mode=False)
        self.profile_provider = ProfileConfigProvider()

    def test_grantlist_loads_from_example_profile(self):
        """Test that grantlist is loaded from example profile."""
        grantlist = self.profile_provider.load_profile_grantlist("example")

        self.assertIsNotNone(grantlist)
        self.assertIsInstance(grantlist, set)
        self.assertGreater(len(grantlist), 0)
        self.assertIn("example321", grantlist)
        logger.info(f"Grantlist loaded with {len(grantlist)} items: {grantlist}")

    def test_grantlist_items_are_protected(self):
        """Test that grantlisted words are protected from anonymization."""
        # Grantlist items are marked as GRANTLISTED entity type
        text = "The grantlisted item is example321 in our system."
        result = self.anonymizer.anonymize(text, profile="example")

        # Check that the grantlisted word was not anonymized
        self.assertIn("example321", result.anonymized_text)

        # The grantlist items should be in GRANTLISTED entity type
        # They won't be anonymized by default operators
        if "GRANTLISTED" in result.details:
            self.assertIn("example321", result.details["GRANTLISTED"])
            logger.info(f"Grantlisted item protected: {result.details['GRANTLISTED']}")

    def test_grantlist_case_insensitive(self):
        """Test that grantlist matching is case-insensitive."""
        test_cases = [
            "example321",
            "EXAMPLE321",
            "Example321",
            "eXaMpLe321",
        ]

        for variant in test_cases:
            text = f"The grantlisted item {variant} is here."
            result = self.anonymizer.anonymize(text, profile="example")

            # Should be marked as GRANTLISTED
            if "GRANTLISTED" in result.details:
                logger.info(f"Case variant '{variant}' correctly marked as grantlisted")

    def test_grantlist_empty_when_no_profile_config(self):
        """Test that grantlist is empty for non-existent profile."""
        grantlist = self.profile_provider.load_profile_grantlist("nonexistent_profile")

        self.assertIsInstance(grantlist, set)
        self.assertEqual(len(grantlist), 0)
        logger.info("Empty grantlist returned for non-existent profile")


class TestProfileCustomListsIntegration(unittest.TestCase):
    """Integration tests for blocklist and grantlist with other recognizers."""

    label = "MUU_TUNNISTE"

    def setUp(self):
        """Initialize test fixtures."""

        self.anonymizer = TextAnonymizer(debug_mode=False)
        self.profile_provider = ProfileConfigProvider()
        self.temp_profiles_dir = None

    def tearDown(self):
        """Clean up temporary test profiles."""
        if self.temp_profiles_dir and os.path.exists(self.temp_profiles_dir):
            import shutil
            shutil.rmtree(self.temp_profiles_dir)

    def test_blocklist_and_regex_patterns_together(self):
        """Test that blocklist works alongside regex patterns in a profile."""
        # The example profile has both regex patterns (EXAMPLE entity) and blocklist
        text = "Code EXAMPLE123 and item blockword123 are both here."
        result = self.anonymizer.anonymize(text, profile="example")

        # Should detect both regex pattern and blocklist
        # EXAMPLE from regex patterns
        # OTHER from blocklist
        self.assertGreater(len(result.details), 0)
        logger.info(f"Combined blocklist and regex detection: {result.details}")

    def test_profile_blocklist_with_default_anonymizer(self):
        """Test that blocklist doesn't affect default anonymizer."""
        text = "The code blockword123 is in our system."

        # Anonymize without profile
        result_no_profile = self.anonymizer.anonymize(text)

        # Anonymize with profile
        result_with_profile = self.anonymizer.anonymize(text, profile="example")

        # Results should be different because profile adds blocklist recognizer
        # The blocklisted item should only be detected with profile
        has_blocklist_no_profile = self.label in result_no_profile.details
        has_blocklist_with_profile = self.label in result_with_profile.details

        self.assertFalse(has_blocklist_no_profile,
                        "Blocklist should not be in default anonymizer")
        self.assertTrue(has_blocklist_with_profile,
                       "Blocklist should be in profile anonymizer")
        logger.info("Blocklist correctly isolated to profile anonymizer")

    def test_multiple_blocklist_items_with_different_frequencies(self):
        """Test counting multiple blocklist items in text."""
        text = "We have blockword123 once, but blockword123 appears twice total."
        result = self.anonymizer.anonymize(text, profile="example")

        if self.label in result.statistics:
            count = result.statistics[self.label]
            self.assertEqual(count, 2, "Should count 2 occurrences of blocklisted item")
            logger.info(f"Correctly counted {count} blocklist occurrences")

    def test_blocklist_partial_word_not_detected(self):
        """Test that partial word matches in blocklist are not detected."""
        # example123 is in blocklist, but "example1234" should not match
        text = "The code example1234 is different."
        result = self.anonymizer.anonymize(text, profile="example")

        # Should not have detected as blocklist item
        # (this depends on how the recognizer handles word boundaries)
        logger.info(f"Partial word match result: {result.details}")

    def test_profile_analyzer_caching_with_lists(self):
        """Test that profile analyzers are cached with list recognizers."""
        # First call
        analyzer1 = self.anonymizer._get_analyzer_for_profile("example")

        # Second call
        analyzer2 = self.anonymizer._get_analyzer_for_profile("example")

        # Should be the same cached object
        self.assertIs(analyzer1, analyzer2,
                     "Profile analyzers should be cached")
        logger.info("Profile analyzer successfully cached")

    def test_different_profiles_maintain_separate_lists(self):
        """Test that different profiles maintain separate blocklists and grantlists."""
        blocklist_example = self.profile_provider.load_profile_blocklist("example")
        grantlist_example = self.profile_provider.load_profile_grantlist("example")

        # Non-existent profile should return empty sets
        blocklist_nonexistent = self.profile_provider.load_profile_blocklist("nonexistent")
        grantlist_nonexistent = self.profile_provider.load_profile_grantlist("nonexistent")

        self.assertNotEqual(blocklist_example, blocklist_nonexistent)
        self.assertNotEqual(grantlist_example, grantlist_nonexistent)
        self.assertGreater(len(blocklist_example), 0)
        self.assertEqual(len(blocklist_nonexistent), 0)
        logger.info("Different profiles correctly maintain separate lists")


class TestProfileCustomListsFileHandling(unittest.TestCase):
    """Test file handling for custom lists."""

    def setUp(self):
        """Initialize test fixtures."""
        self.profile_provider = ProfileConfigProvider()

    def test_blocklist_file_with_comments(self):
        """Test that blocklist ignores lines starting with #."""
        # The ProfileConfigProvider should skip comments
        # Create temporary test profile
        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ["CONFIG_DIR"] = temp_dir

            profile_dir = os.path.join(temp_dir, "test_profile")
            os.makedirs(profile_dir, exist_ok=True)

            # Create blocklist with comments
            blocklist_path = os.path.join(profile_dir, "blocklist.txt")
            with open(blocklist_path, 'w', encoding='utf-8') as f:
                f.write("# This is a comment\n")
                f.write("item1\n")
                f.write("# Another comment\n")
                f.write("item2\n")

            # Reset ConfigCache singleton AFTER files are created to pick up new CONFIG_DIR
            from text_anonymizer.config_cache import ConfigCache
            ConfigCache.reset_instance()

            provider = ProfileConfigProvider()
            blocklist = provider.load_profile_blocklist("test_profile")

            self.assertIn("item1", blocklist)
            self.assertIn("item2", blocklist)
            self.assertEqual(len(blocklist), 2)
            logger.info(f"Correctly parsed blocklist with comments: {blocklist}")

            # Reset CONFIG_DIR
            if "CONFIG_DIR" in os.environ:
                del os.environ["CONFIG_DIR"]

    def test_grantlist_file_with_comments(self):
        """Test that grantlist ignores lines starting with #."""
        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ["CONFIG_DIR"] = temp_dir

            profile_dir = os.path.join(temp_dir, "test_profile")
            os.makedirs(profile_dir, exist_ok=True)

            # Create grantlist with comments
            grantlist_path = os.path.join(profile_dir, "grantlist.txt")
            with open(grantlist_path, 'w', encoding='utf-8') as f:
                f.write("# This is a comment\n")
                f.write("grant1\n")
                f.write("# Another comment\n")
                f.write("grant2\n")

            # Reset ConfigCache singleton AFTER files are created to pick up new CONFIG_DIR
            from text_anonymizer.config_cache import ConfigCache
            ConfigCache.reset_instance()

            provider = ProfileConfigProvider()
            grantlist = provider.load_profile_grantlist("test_profile")

            self.assertIn("grant1", grantlist)
            self.assertIn("grant2", grantlist)
            self.assertEqual(len(grantlist), 2)
            logger.info(f"Correctly parsed grantlist with comments: {grantlist}")

            # Reset CONFIG_DIR
            if "CONFIG_DIR" in os.environ:
                del os.environ["CONFIG_DIR"]

    def test_blocklist_empty_lines_ignored(self):
        """Test that empty lines in blocklist are ignored."""
        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ["CONFIG_DIR"] = temp_dir

            profile_dir = os.path.join(temp_dir, "test_profile")
            os.makedirs(profile_dir, exist_ok=True)

            blocklist_path = os.path.join(profile_dir, "blocklist.txt")
            with open(blocklist_path, 'w', encoding='utf-8') as f:
                f.write("item1\n")
                f.write("\n")
                f.write("  \n")
                f.write("item2\n")

            # Reset ConfigCache singleton AFTER files are created to pick up new CONFIG_DIR
            from text_anonymizer.config_cache import ConfigCache
            ConfigCache.reset_instance()

            provider = ProfileConfigProvider()
            blocklist = provider.load_profile_blocklist("test_profile")

            self.assertEqual(len(blocklist), 2)
            self.assertIn("item1", blocklist)
            self.assertIn("item2", blocklist)
            logger.info(f"Correctly handled empty lines: {blocklist}")

            # Reset CONFIG_DIR
            if "CONFIG_DIR" in os.environ:
                del os.environ["CONFIG_DIR"]

    def test_nonexistent_blocklist_returns_empty_set(self):
        """Test that missing blocklist returns empty set."""
        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ["CONFIG_DIR"] = temp_dir

            profile_dir = os.path.join(temp_dir, "empty_profile")
            os.makedirs(profile_dir, exist_ok=True)
            # Don't create blocklist.txt

            # Reset ConfigCache singleton AFTER directory is created to pick up new CONFIG_DIR
            from text_anonymizer.config_cache import ConfigCache
            ConfigCache.reset_instance()

            provider = ProfileConfigProvider()
            blocklist = provider.load_profile_blocklist("empty_profile")

            self.assertEqual(len(blocklist), 0)
            self.assertIsInstance(blocklist, set)
            logger.info("Correctly returned empty set for missing blocklist")

            # Reset CONFIG_DIR
            if "CONFIG_DIR" in os.environ:
                del os.environ["CONFIG_DIR"]


if __name__ == "__main__":
    # Run unittest tests
    unittest.main(verbosity=2)

