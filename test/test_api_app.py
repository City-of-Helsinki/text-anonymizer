#!/usr/bin/env python
"""
Test suite for the Anonymizer FastAPI application.

Verifies that:
1. API is accessible and running
2. Basic anonymization works via /anonymize endpoint
3. Profile-based anonymization works correctly
4. Batch anonymization works via /anonymize_batch endpoint
5. Proper error handling and response validation
"""

import sys
import os
import unittest
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

try:
    import requests
except ImportError:
    logger.error("requests is required for API tests. Install with: pip install requests")
    requests = None

API_URL = "http://127.0.0.1:8000"
API_TIMEOUT = 2.0


class TestAnonymizerAPI(unittest.TestCase):
    """Test suite for the Anonymizer FastAPI app."""

    @classmethod
    def setUpClass(cls):
        """Check if API is available before running tests."""
        if requests is None:
            cls.api_available = False
            logger.warning("requests not installed. Skipping API tests.")
            return

        try:
            response = requests.get(f"{API_URL}/docs", timeout=API_TIMEOUT)
            if response.status_code == 200:
                cls.api_available = True
                logger.info("API is running at %s", API_URL)
            else:
                cls.api_available = False
                logger.warning("API responded with status %d. Skipping API tests.", response.status_code)
        except (requests.ConnectionError, requests.Timeout) as e:
            cls.api_available = False
            logger.warning("Anonymizer API is not running at %s. Skipping API tests. Error: %s", API_URL, e)
        except Exception as e:
            cls.api_available = False
            logger.warning("Unexpected error connecting to API: %s", e)

    def setUp(self):
        """Skip test if API is not available."""
        if not self.api_available:
            test_name = self._testMethodName
            self.skipTest(f"API not running - skipping {test_name}")

    def test_api_docs_accessible(self):
        """Test that API documentation is accessible."""
        response = requests.get(f"{API_URL}/docs", timeout=API_TIMEOUT)
        self.assertEqual(response.status_code, 200)
        logger.info("API documentation is accessible")

    def test_anonymize_simple_text(self):
        """Test basic anonymization endpoint with simple text."""
        payload = {
            "text": "Puhelinnumeroni on 040-1234567.",
            "languages": ["fi"],
            "recognizers": [],
            "profile": None
        }
        response = requests.post(f"{API_URL}/anonymize", json=payload, timeout=API_TIMEOUT)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("anonymized_txt", data)
        self.assertIn("statistics", data)
        self.assertIsNotNone(data["anonymized_txt"])

        # Phone number should be anonymized
        self.assertNotIn("040-1234567", data["anonymized_txt"])
        logger.info("Simple text anonymization successful: %s", data["anonymized_txt"])

    def test_anonymize_finnish_ssn(self):
        """Test anonymization of Finnish social security number."""
        payload = {
            "text": "Minun henkilötunnukseni on 311299-999A.",
            "languages": ["fi"],
            "recognizers": [],
            "profile": None
        }
        response = requests.post(f"{API_URL}/anonymize", json=payload, timeout=API_TIMEOUT)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertNotIn("311299-999A", data["anonymized_txt"])

        # Check statistics
        if data["statistics"]:
            self.assertIsInstance(data["statistics"], dict)
            logger.info("SSN anonymization statistics: %s", data["statistics"])

    def test_anonymize_with_profile_blocklist(self):
        """Test anonymization with a profile (blocklist/grantlist)."""
        payload = {
            "text": "Tunniste blockword123 on lauseessa.",
            "languages": ["fi"],
            "recognizers": [],
            "profile": "example"
        }
        response = requests.post(f"{API_URL}/anonymize", json=payload, timeout=API_TIMEOUT)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("anonymized_txt", data)

        # Blocklisted word should be anonymized
        self.assertNotIn("blockword123", data["anonymized_txt"])

        # Check statistics for MUU_TUNNISTE (custom identifier)
        self.assertIn("statistics", data)
        self.assertIsInstance(data["statistics"], dict)

        if "MUU_TUNNISTE" in data["statistics"]:
            self.assertGreater(data["statistics"]["MUU_TUNNISTE"], 0)
            logger.info("Profile blocklist working: MUU_TUNNISTE count = %d",
                       data["statistics"]["MUU_TUNNISTE"])

    def test_anonymize_with_profile_grantlist(self):
        """Test that grantlisted items are protected when using profile."""
        payload = {
            "text": "Sallittu kohde on example321 lauseessa.",
            "languages": ["fi"],
            "recognizers": [],
            "profile": "example"
        }
        response = requests.post(f"{API_URL}/anonymize", json=payload, timeout=API_TIMEOUT)
        self.assertEqual(response.status_code, 200)

        data = response.json()

        # Grantlisted word should NOT be anonymized
        self.assertIn("example321", data["anonymized_txt"])
        logger.info("Grantlist protection working: %s", data["anonymized_txt"])

    def test_anonymize_with_profile_combined(self):
        """Test profile with both blocklist and grantlist items."""
        payload = {
            "text": "Estetty: blockword123, Sallittu: example321, Puhelin: 040-1234567",
            "languages": ["fi"],
            "recognizers": [],
            "profile": "example"
        }
        response = requests.post(f"{API_URL}/anonymize", json=payload, timeout=API_TIMEOUT)
        self.assertEqual(response.status_code, 200)

        data = response.json()

        # Blocklisted should be anonymized
        self.assertNotIn("blockword123", data["anonymized_txt"])

        # Grantlisted should be protected
        self.assertIn("example321", data["anonymized_txt"])

        # Phone should be anonymized
        self.assertNotIn("040-1234567", data["anonymized_txt"])

        logger.info("Combined profile test successful: %s", data["anonymized_txt"])

    def test_anonymize_without_profile(self):
        """Test that blocklist is not applied when no profile is specified."""
        payload = {
            "text": "Tunniste blockword123 on lauseessa.",
            "languages": ["fi"],
            "recognizers": [],
            "profile": None
        }
        response = requests.post(f"{API_URL}/anonymize", json=payload, timeout=API_TIMEOUT)
        self.assertEqual(response.status_code, 200)

        data = response.json()

        # Without profile, blockword123 should NOT be detected as custom identifier
        # It should remain in text (unless it matches another recognizer)
        statistics = data.get("statistics", {})

        # MUU_TUNNISTE should not be present without profile
        self.assertNotIn("MUU_TUNNISTE", statistics)
        logger.info("Blocklist correctly not applied without profile")

    def test_anonymize_batch(self):
        """Test batch anonymization endpoint."""
        payload = [
            {
                "text": "Henkilötunnukseni on 311299-999A.",
                "languages": ["fi"],
                "recognizers": [],
                "profile": None
            },
            {
                "text": "Tunniste blockword123 on lauseessa.",
                "languages": ["fi"],
                "recognizers": [],
                "profile": "example"
            },
            {
                "text": "Soita minulle numeroon 040-9876543.",
                "languages": ["fi"],
                "recognizers": [],
                "profile": None
            }
        ]
        response = requests.post(f"{API_URL}/anonymize_batch", json=payload, timeout=API_TIMEOUT)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 3)

        # Verify each item has required structure
        for item in data:
            self.assertIn("anonymized_txt", item)
            self.assertIn("statistics", item)

        # Check first item (SSN)
        self.assertNotIn("311299-999A", data[0]["anonymized_txt"])

        # Check second item (blocklist with profile)
        self.assertNotIn("blockword123", data[1]["anonymized_txt"])

        # Check third item (phone)
        self.assertNotIn("040-9876543", data[2]["anonymized_txt"])

        logger.info("Batch anonymization successful: %d items processed", len(data))

    def test_anonymize_batch_mixed_profiles(self):
        """Test batch processing with different profiles."""
        payload = [
            {
                "text": "Item blockword123 without profile context.",
                "languages": ["fi"],
                "recognizers": [],
                "profile": None
            },
            {
                "text": "Item blockword123 with example profile.",
                "languages": ["fi"],
                "recognizers": [],
                "profile": "example"
            }
        ]
        response = requests.post(f"{API_URL}/anonymize_batch", json=payload, timeout=API_TIMEOUT)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(len(data), 2)

        # First should not detect blockword123 as MUU_TUNNISTE
        stats1 = data[0].get("statistics", {})
        self.assertNotIn("MUU_TUNNISTE", stats1)

        # Second should detect blockword123 as MUU_TUNNISTE
        stats2 = data[1].get("statistics", {})
        if "MUU_TUNNISTE" in stats2:
            self.assertGreater(stats2["MUU_TUNNISTE"], 0)

        logger.info("Mixed profile batch test successful")

    def test_anonymize_empty_text(self):
        """Test handling of empty text."""
        payload = {
            "text": "",
            "languages": ["fi"],
            "recognizers": [],
            "profile": None
        }
        response = requests.post(f"{API_URL}/anonymize", json=payload, timeout=API_TIMEOUT)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        # Empty input can return either None or empty string
        self.assertIn(data["anonymized_txt"], [None, ""],
                      "Empty text should return None or empty string")
        logger.info("Empty text handled correctly: %s", data["anonymized_txt"])

    def test_anonymize_multiple_languages(self):
        """Test anonymization with multiple languages."""
        payload = {
            "text": "Hei, olen Matti. Hello, I am John.",
            "languages": ["fi", "en"],
            "recognizers": [],
            "profile": None
        }
        response = requests.post(f"{API_URL}/anonymize", json=payload, timeout=API_TIMEOUT)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("anonymized_txt", data)
        logger.info("Multi-language anonymization: %s", data["anonymized_txt"])

    def test_anonymize_default_languages(self):
        """Test that default language is Finnish if not specified."""
        payload = {
            "text": "Soita numeroon 040-1234567.",
            # languages not specified, should default to ['fi']
            "recognizers": [],
            "profile": None
        }
        response = requests.post(f"{API_URL}/anonymize", json=payload, timeout=API_TIMEOUT)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertNotIn("040-1234567", data["anonymized_txt"])
        logger.info("Default language (Finnish) working correctly")

    def test_anonymize_statistics_structure(self):
        """Test that statistics are returned in correct format with expected entities."""
        payload = {
            "text": "Contact: 040-1234567, SSN: 311299-999A",
            "languages": ["fi"],
            "recognizers": [],
            "profile": None
        }
        response = requests.post(f"{API_URL}/anonymize", json=payload, timeout=API_TIMEOUT)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("statistics", data)
        self.assertIsInstance(data["statistics"], dict)

        # Both phone and SSN should be anonymized
        self.assertNotIn("040-1234567", data["anonymized_txt"])
        self.assertNotIn("311299-999A", data["anonymized_txt"])

        # Statistics should not be empty since we have entities
        self.assertTrue(data["statistics"], "Statistics should not be empty for text with entities")

        # Verify statistics structure
        for entity_type, count in data["statistics"].items():
            self.assertIsInstance(entity_type, str)
            self.assertIsInstance(count, int)
            self.assertGreater(count, 0)

        # Check for expected entity types (phone and SSN)
        has_phone = "PUHELIN" in data["statistics"] or "PHONE_NUMBER" in data["statistics"]
        has_ssn = "FI_SSN" in data["statistics"] or "HETU" in data["statistics"]

        self.assertTrue(has_phone or has_ssn,
                       "Should detect at least phone or SSN in statistics")

        logger.info("Statistics structure valid: %s", data["statistics"])

    def test_anonymize_nonexistent_profile(self):
        """Test that nonexistent profile is handled gracefully without custom recognizers."""
        payload = {
            "text": "Tunniste blockword123 on lauseessa.",
            "languages": ["fi"],
            "recognizers": [],
            "profile": "nonexistent_profile_xyz"
        }
        response = requests.post(f"{API_URL}/anonymize", json=payload, timeout=API_TIMEOUT)

        # Should still return 200 (graceful handling)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("anonymized_txt", data)

        # Nonexistent profile should NOT trigger custom recognizers
        statistics = data.get("statistics", {})
        self.assertNotIn("MUU_TUNNISTE", statistics,
                         "Nonexistent profile should not apply blocklist")

        # blockword123 should remain in text (no custom recognizer applied)
        self.assertIn("blockword123", data["anonymized_txt"],
                      "Word should not be anonymized without valid profile")

        logger.info("Non-existent profile handled gracefully - no custom recognizers applied")

    def test_anonymize_with_profile_regex_patterns(self):
        """Test custom regex patterns from example profile's regex_patterns.json."""
        # Test various patterns defined in config/example/regex_patterns.json
        test_cases = [
            ("TEST_EXAMPLE1", "prefix_variations"),
            ("PROD_EXAMPLE99", "prefix_variations"),
            ("DEV_EXAMPLE123", "prefix_variations"),
            ("EXAMPLEabc", "alphanumeric_suffix"),
            ("EXAMPLE1a2b", "alphanumeric_suffix"),
            ("EXAMPLE123", "word_with_numbers"),
            ("EXAMPLE999999", "word_with_numbers"),
            ("EXAMPLE12", "word_with_digit_range"),
            ("EXAMPLE1234", "word_with_digit_range"),
            ("example", "case_insensitive_variations"),
            ("Example", "case_insensitive_variations"),
            ("EXAMPLE", "exact_match"),
        ]

        for test_word, pattern_name in test_cases:
            payload = {
                "text": f"Tässä lauseessa on {test_word} tunniste.",
                "languages": ["fi"],
                "recognizers": [],
                "profile": "example"
            }
            response = requests.post(f"{API_URL}/anonymize", json=payload, timeout=API_TIMEOUT)
            self.assertEqual(response.status_code, 200)

            data = response.json()

            # The EXAMPLE pattern should be anonymized
            self.assertNotIn(test_word, data["anonymized_txt"],
                           f"{test_word} should be anonymized by pattern '{pattern_name}'")

            # Check statistics for EXAMPLE entity
            statistics = data.get("statistics", {})
            self.assertIn("EXAMPLE", statistics,
                         f"EXAMPLE entity should be detected for '{test_word}' (pattern: {pattern_name})")
            self.assertGreater(statistics["EXAMPLE"], 0,
                              f"EXAMPLE count should be > 0 for '{test_word}'")

        logger.info("Custom regex patterns tested successfully: %d patterns verified", len(test_cases))

    def test_anonymize_with_profile_regex_patterns_no_match(self):
        """Test that words not matching regex patterns are preserved."""
        # These should NOT match any EXAMPLE patterns
        non_matching_words = [
            "EXAMPLES",  # Has 'S' at end, word boundary prevents match
            "MYEXAMPLE",  # Prefix prevents word boundary match
            "EXAMPLE1234567",  # Too many digits for most patterns
            "example123test",  # Has suffix after numbers
            "EX",  # Too short
        ]

        for word in non_matching_words:
            payload = {
                "text": f"Tässä lauseessa on {word} sana.",
                "languages": ["fi"],
                "recognizers": [],
                "profile": "example"
            }
            response = requests.post(f"{API_URL}/anonymize", json=payload, timeout=API_TIMEOUT)
            self.assertEqual(response.status_code, 200)

            data = response.json()

            # Word should remain (not matched by EXAMPLE patterns)
            self.assertIn(word, data["anonymized_txt"],
                         f"{word} should NOT be anonymized by EXAMPLE patterns")

        logger.info("Verified non-matching words preserved: %d words tested", len(non_matching_words))

    def test_anonymize_with_nonexistent_profile_no_regex(self):
        """Test that custom regex patterns are NOT applied with nonexistent profile."""
        # These patterns would match with "example" profile, but not with nonexistent profile
        test_words = [
            "TEST_EXAMPLE1",
            "EXAMPLE123",
        ]

        for word in test_words:
            payload = {
                "text": f"Tässä lauseessa on {word} tunniste.",
                "languages": ["fi"],
                "recognizers": [],
                "profile": "nonexistent_profile_xyz"
            }
            response = requests.post(f"{API_URL}/anonymize", json=payload, timeout=API_TIMEOUT)
            self.assertEqual(response.status_code, 200)

            data = response.json()

            # Word should remain (no custom regex patterns loaded)
            self.assertIn(word, data["anonymized_txt"],
                         f"{word} should NOT be anonymized without valid profile")

            # EXAMPLE entity should NOT be in statistics
            statistics = data.get("statistics", {})
            self.assertNotIn("EXAMPLE", statistics,
                           f"EXAMPLE entity should not be detected without valid profile for '{word}'")

        logger.info("Verified regex patterns not applied with nonexistent profile: %d words tested", len(test_words))



class TestAnonymizerAPIEdgeCases(unittest.TestCase):
    """Test edge cases and error handling for the API."""

    @classmethod
    def setUpClass(cls):
        """Check if API is available before running tests."""
        if requests is None:
            cls.api_available = False
            return

        try:
            response = requests.get(f"{API_URL}/docs", timeout=API_TIMEOUT)
            cls.api_available = response.status_code == 200
        except Exception:
            cls.api_available = False
            logger.warning("API not available for edge case tests")

    def setUp(self):
        """Skip test if API is not available."""
        if not self.api_available:
            test_name = self._testMethodName
            self.skipTest(f"API not running - skipping {test_name}")

    def test_anonymize_very_long_text(self):
        """Test anonymization of longer text with multiple phone numbers."""
        long_text = " ".join([f"This is sentence {i} with phone 040-{i:07d}." for i in range(50)])
        payload = {
            "text": long_text,
            "languages": ["fi"],
            "recognizers": [],
            "profile": None
        }
        response = requests.post(f"{API_URL}/anonymize", json=payload, timeout=10.0)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIsNotNone(data["anonymized_txt"])

        # Verify that at least some phone numbers were anonymized
        # Check a few specific numbers are not in output
        self.assertNotIn("040-0000000", data["anonymized_txt"])
        self.assertNotIn("040-0000010", data["anonymized_txt"])
        self.assertNotIn("040-0000049", data["anonymized_txt"])

        # Statistics should show phone detections
        statistics = data.get("statistics", {})
        phone_count = statistics.get("PUHELIN", statistics.get("PHONE_NUMBER", 0))
        self.assertGreater(phone_count, 0, "Should detect phone numbers in long text")

        logger.info("Long text processed successfully: %d phone numbers detected", phone_count)

    def test_anonymize_special_characters(self):
        """Test handling of special characters while anonymizing phone numbers."""
        payload = {
            "text": "Special chars: @#$%^&*() with phone 040-1234567",
            "languages": ["fi"],
            "recognizers": [],
            "profile": None
        }
        response = requests.post(f"{API_URL}/anonymize", json=payload, timeout=API_TIMEOUT)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("anonymized_txt", data)

        # Phone number should be anonymized
        self.assertNotIn("040-1234567", data["anonymized_txt"])

        # Special characters should be preserved
        self.assertIn("@#$%^&*()", data["anonymized_txt"])

        # Statistics should show phone detection
        statistics = data.get("statistics", {})
        self.assertTrue(
            "PUHELIN" in statistics or "PHONE_NUMBER" in statistics,
            "Phone number should be detected in statistics"
        )

        logger.info("Special characters preserved while phone anonymized: %s", data["anonymized_txt"])

    def test_batch_empty_list(self):
        """Test batch endpoint with empty list."""
        payload = []
        response = requests.post(f"{API_URL}/anonymize_batch", json=payload, timeout=API_TIMEOUT)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data, [])
        logger.info("Empty batch handled correctly")


if __name__ == "__main__":
    unittest.main(verbosity=2)

