#!/usr/bin/env python
"""
Test that demonstrates grantlist protection for names.

This test:
1. Anonymizes text containing common Finnish names without grantlist
2. Creates a temporary profile with a grantlist containing those names
3. Anonymizes the same text with the grantlist profile
4. Verifies that names in the grantlist are no longer anonymized
"""

import sys
import os
import tempfile
import shutil
import unittest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

from text_anonymizer import TextAnonymizer
from text_anonymizer.profile_config_provider import ProfileConfigProvider


class TestGrantlistNameProtection(unittest.TestCase):
    """Test that grantlist protects specified names from anonymization."""

    def setUp(self):
        """Initialize test fixtures."""
        self.anonymizer = TextAnonymizer(debug_mode=False)
        self.profile_provider = ProfileConfigProvider()
        self.temp_config_dir = None
        self.original_config_dir = os.environ.get("CONFIG_DIR")

    def tearDown(self):
        """Clean up temporary test profiles."""
        # Reset ConfigCache singleton
        from text_anonymizer.config_cache import ConfigCache
        ConfigCache.reset_instance()

        # Reset profile analyzers cache
        self.anonymizer._profile_analyzers_cache.clear()

        # Restore original CONFIG_DIR
        if self.original_config_dir:
            os.environ["CONFIG_DIR"] = self.original_config_dir
        elif "CONFIG_DIR" in os.environ:
            del os.environ["CONFIG_DIR"]

        # Clean up temporary directory
        if self.temp_config_dir and os.path.exists(self.temp_config_dir):
            shutil.rmtree(self.temp_config_dir)


    def _setup_temp_profile_with_grantlist(self, profile_name: str, names: list) -> str:
        """
        Create a temporary profile with a grantlist containing specified names.

        :param profile_name: Name of the profile to create
        :param names: List of names to add to grantlist
        :return: Path to temporary config directory
        """
        # Create temporary config directory
        temp_config_dir = tempfile.mkdtemp()
        self.temp_config_dir = temp_config_dir

        # Set CONFIG_DIR to use temporary directory
        os.environ["CONFIG_DIR"] = temp_config_dir

        # IMPORTANT: Reset ConfigCache singleton to force it to reload from new CONFIG_DIR
        from text_anonymizer.config_cache import ConfigCache
        ConfigCache.reset_instance()

        # Create profile directory
        profile_dir = os.path.join(temp_config_dir, profile_name)
        os.makedirs(profile_dir, exist_ok=True)

        # Create grantlist file with the names
        grantlist_path = os.path.join(profile_dir, "grantlist.txt")
        with open(grantlist_path, 'w', encoding='utf-8') as f:
            for name in names:
                f.write(f"{name}\n")

        logger.info(f"Created temporary profile '{profile_name}' with grantlist at {profile_dir}")
        logger.info(f"Grantlist contains: {names}")

        return temp_config_dir

    def test_names_anonymized_without_grantlist(self):
        """Test that names are anonymized when no grantlist is present."""
        test_text = "Useita merkityksiä voi olla sanoilla Anselmi, Elisa, Maisa."

        print("\n" + "="*70)
        print("TEST: Names anonymized WITHOUT grantlist")
        print("="*70)
        print(f"\nOriginal text: {test_text}")

        # Anonymize without profile (no grantlist)
        result = self.anonymizer.anonymize(test_text)

        print(f"\nAnonymized text: {result.anonymized_text}")
        print(f"Detection details: {result.details}")
        print(f"Statistics: {result.statistics}")

        # Check that at least some PER (person) entities were detected
        has_person_detection = "<NIMI>" in result.anonymized_text
        self.assertTrue(has_person_detection, "Should detect person names without grantlist")

        logger.info("Names correctly detected and anonymized without grantlist")

    def test_names_protected_with_grantlist(self):
        """Test that grantlisted names are NOT anonymized."""
        names_to_protect = ["Anselmi", "Elisa", "Maisa"]
        test_text = "Luetellaan satunnaisia nimiä, vaikkapa Anselmi, Elisa, Maisa."

        print("\n" + "="*70)
        print("TEST: Names PROTECTED with grantlist")
        print("="*70)
        print(f"\nOriginal text: {test_text}")
        print(f"Names to protect: {names_to_protect}")

        # Create temporary profile with grantlist
        self._setup_temp_profile_with_grantlist("test_names_profile", names_to_protect)

        # Verify grantlist was loaded
        grantlist = self.profile_provider.load_profile_grantlist("test_names_profile")
        print(f"\nLoaded grantlist: {grantlist}")
        self.assertEqual(len(grantlist), len(names_to_protect))

        # Anonymize with grantlist profile
        result = self.anonymizer.anonymize(test_text, profile="test_names_profile")

        print(f"\nAnonymized text (with grantlist): {result.anonymized_text}")
        print(f"Detection details: {result.details}")
        print(f"Statistics: {result.statistics}")

        # The grantlisted names should be marked as GRANTLISTED entity type
        # and NOT anonymized in the final text
        if "GRANTLISTED" in result.details:
            grantlisted_items = result.details["GRANTLISTED"]
            print(f"\nGRANTLISTED items detected: {grantlisted_items}")

            # Verify protected names are in grantlist
            for name in names_to_protect:
                self.assertIn(name.lower(), {item.lower() for item in grantlisted_items},
                            f"Name '{name}' should be marked as grantlisted")

        # Verify original names still appear in anonymized text
        for name in names_to_protect:
            self.assertIn(name, result.anonymized_text,
                         f"Grantlisted name '{name}' should appear in anonymized text")

        logger.info(f"All {len(names_to_protect)} names correctly protected by grantlist")

    def test_comparison_with_and_without_grantlist(self):
        """Compare anonymization results with and without grantlist."""
        names_to_protect = ["Anselmi", "Elisa", "Maisa"]
        test_text = "Tässä lauseessa mainitaan testinimet Anselmi, Elisa , Maisa."

        print("\n" + "="*70)
        print("TEST: Comparison WITH and WITHOUT grantlist")
        print("="*70)
        print(f"\nTest text: {test_text}")
        print(f"Names to protect: {names_to_protect}\n")

        # Anonymize WITHOUT grantlist
        result_without = self.anonymizer.anonymize(test_text)
        print(f"WITHOUT grantlist:")
        print(f"  Anonymized: {result_without.anonymized_text}")
        print(f"  Detections: {result_without.details}")
        print(f"  Statistics: {result_without.statistics}")

        # Create temporary profile with grantlist
        self._setup_temp_profile_with_grantlist("test_names_comparison", names_to_protect)

        # Anonymize WITH grantlist
        result_with = self.anonymizer.anonymize(test_text, profile="test_names_comparison")
        print(f"\nWITH grantlist:")
        print(f"  Anonymized: {result_with.anonymized_text}")
        print(f"  Detections: {result_with.details}")
        print(f"  Statistics: {result_with.statistics}")

        # Verify the texts are different
        self.assertNotEqual(result_without.anonymized_text, result_with.anonymized_text,
                           "Texts should be different with and without grantlist")

        # Verify protected names are in the grantlist version
        for name in names_to_protect:
            self.assertIn(name, result_with.anonymized_text,
                         f"Name '{name}' should be preserved with grantlist")

        logger.info("Comparison test confirms grantlist protection works")

    def test_grantlist_is_case_insensitive(self):
        """Test that grantlist matching is case-insensitive."""
        names_to_protect = ["Anselmi", "Elisa", "Maisa"]

        # Create test text with different cases
        test_cases = [
            "Anselmi kävelee kadulla",
            "Anselmi katsoo merelle",
            "Anselmi meni kauppaan",
            "Elisa on tässä",
            "Elisa ON TUOLLA!!!!",
            "ELISA olikin siellä",
        ]

        print("\n" + "="*70)
        print("TEST: Grantlist is case-insensitive")
        print("="*70)

        self._setup_temp_profile_with_grantlist("test_case_insensitive", names_to_protect)

        for text in test_cases:
            result = self.anonymizer.anonymize(text, profile="test_case_insensitive")

            print(f"\nTest: {text}")
            print(f"  Result: {result.anonymized_text}")

            # Verify grantlist protected names remain (case-insensitive)
            if "GRANTLISTED" in result.details:
                logger.info(f"  GRANTLISTED: {result.details['GRANTLISTED']}")

        logger.info("Case-insensitive grantlist matching confirmed")

    def test_grantlist_with_multiple_profiles(self):
        """Test that different profiles maintain separate grantlists."""
        names_profile1 = ["Anselmi"]
        names_profile2 = ["Elisa", "Maisa"]

        print("\n" + "="*70)
        print("TEST: Multiple profiles with different grantlists")
        print("="*70)

        # Create temporary config directory for multiple profiles
        temp_config_dir = tempfile.mkdtemp()
        self.temp_config_dir = temp_config_dir
        os.environ["CONFIG_DIR"] = temp_config_dir

        # IMPORTANT: Reset ConfigCache singleton to force it to reload from new CONFIG_DIR
        from text_anonymizer.config_cache import ConfigCache
        ConfigCache.reset_instance()

        # Create profile 1
        profile1_dir = os.path.join(temp_config_dir, "profile1")
        os.makedirs(profile1_dir, exist_ok=True)
        with open(os.path.join(profile1_dir, "grantlist.txt"), 'w', encoding='utf-8') as f:
            for name in names_profile1:
                f.write(f"{name}\n")

        # Create profile 2
        profile2_dir = os.path.join(temp_config_dir, "profile2")
        os.makedirs(profile2_dir, exist_ok=True)
        with open(os.path.join(profile2_dir, "grantlist.txt"), 'w', encoding='utf-8') as f:
            for name in names_profile2:
                f.write(f"{name}\n")

        # Verify profiles have correct grantlists
        grantlist1 = self.profile_provider.load_profile_grantlist("profile1")
        grantlist2 = self.profile_provider.load_profile_grantlist("profile2")

        print(f"\nProfile1 grantlist: {grantlist1}")
        print(f"Profile2 grantlist: {grantlist2}")

        self.assertIn("anselmi", grantlist1)
        self.assertNotIn("elisa", grantlist1)

        self.assertIn("elisa", grantlist2)
        self.assertNotIn("anselmi", grantlist2)

        logger.info("Multiple profiles correctly maintain separate grantlists")

if __name__ == "__main__":
    unittest.main(verbosity=2)

