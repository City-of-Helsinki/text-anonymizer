"""
Test suite for verifying that the example profile correctly detects EXAMPLE entities.

This test validates that all EXAMPLE regex patterns in the example profile configuration
correctly identify various entity types including specific codes, uppercase codes,
ticket numbers, and reference codes.
"""

import unittest
from text_anonymizer import TextAnonymizer


class TestCustomRegex(unittest.TestCase):
    """Test cases for verifying EXAMPLE entity detection in example profile."""

    def setUp(self):
        """Initialize the text anonymizer and set test parameters."""
        self.label = "EXAMPLE"
        self.profile_name = "example"
        self.anonymizer = TextAnonymizer(debug_mode=False)

    # Test cases for specific code pattern: (?:ABCxyz123|XYZabc321)
    def test_specific_code_abcxyz123_standalone(self):
        """Test detection of ABCxyz123 standalone."""
        text = "ABCxyz123"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        self.assertIn(self.label, result.details)
        self.assertIn(text, result.details[self.label])

    def test_specific_code_xyzabc321_standalone(self):
        """Test detection of XYZabc321 standalone."""
        text = "XYZabc321"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        self.assertIn(self.label, result.details)
        self.assertIn(text, result.details[self.label])

    def test_specific_code_in_sentence(self):
        """Test detection of specific codes in sentences."""
        test_cases = [
            "This is a test sentence with ABCxyz123 included.",
            "ABCxyz123 is part of the department.",
            "The code is XYZabc321 and should be detected.",
        ]
        for text in test_cases:
            with self.subTest(text=text):
                result = self.anonymizer.anonymize(text, profile=self.profile_name)
                self.assertIn(self.label, result.details, f"Entity not detected in: {text}")

    def test_specific_code_multiple_in_text(self):
        """Test detection of multiple specific codes in one text."""
        text = "Code ABCxyz123 and code XYZabc321 both present."
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        self.assertIn(self.label, result.details)
        entities = result.details[self.label]
        self.assertEqual(len(entities), 2, "Should detect both codes")
        self.assertIn("ABCxyz123", entities)
        self.assertIn("XYZabc321", entities)

    def test_specific_code_with_trailing_word(self):
        """Test that trailing words are not included in detection."""
        text = "XYZabc321 unit"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        self.assertIn(self.label, result.details)
        # Should only detect the code, not "unit"
        self.assertEqual(result.details[self.label], ["XYZabc321"])

    # Test cases for uppercase code pattern: [A-Z]{3}[0-9]{3,}
    def test_uppercase_code_three_letters_three_digits(self):
        """Test detection of uppercase code with exactly 3 letters and 3 digits."""
        text = "ABC123"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        self.assertIn(self.label, result.details)
        self.assertIn(text, result.details[self.label])

    def test_uppercase_code_three_letters_many_digits(self):
        """Test detection of uppercase code with 3 letters and many digits."""
        text = "XYZ999999"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        self.assertIn(self.label, result.details)
        self.assertIn(text, result.details[self.label])

    def test_uppercase_code_case_sensitive(self):
        """Test that lowercase codes don't match uppercase pattern."""
        text = "abc123"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        # Lowercase pattern should not be detected by uppercase code pattern
        self.assertNotIn(text, result.details.get(self.label, []))

    def test_uppercase_code_in_context(self):
        """Test uppercase code pattern in sentences."""
        text = "Process DEF456 immediately"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        self.assertIn(self.label, result.details)
        self.assertIn("DEF456", result.details[self.label])

    # Test cases for ticket number pattern: TICKET-\d{4,6}
    def test_ticket_number_four_digits(self):
        """Test detection of ticket number with 4 digits."""
        text = "TICKET-1234"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        self.assertIn(self.label, result.details)
        self.assertIn(text, result.details[self.label])

    def test_ticket_number_six_digits(self):
        """Test detection of ticket number with 6 digits."""
        text = "TICKET-123456"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        self.assertIn(self.label, result.details)
        self.assertIn(text, result.details[self.label])

    def test_ticket_number_five_digits(self):
        """Test detection of ticket number with 5 digits."""
        text = "TICKET-12345"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        self.assertIn(self.label, result.details)
        self.assertIn(text, result.details[self.label])

    def test_ticket_number_too_few_digits(self):
        """Test that ticket number with too few digits is not detected."""
        text = "TICKET-123"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        if self.label in result.details:
            self.assertNotIn(text, result.details[self.label])

    def test_ticket_number_too_many_digits(self):
        """Test that ticket number with too many digits is not detected."""
        text = "TICKET-1234567"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        if self.label in result.details:
            self.assertNotIn(text, result.details[self.label])

    def test_ticket_number_in_sentence(self):
        """Test ticket number detection in context."""
        text = "Please resolve TICKET-5678 as soon as possible."
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        self.assertIn(self.label, result.details)
        self.assertIn("TICKET-5678", result.details[self.label])

    def test_ticket_number_multiple(self):
        """Test detection of multiple ticket numbers."""
        text = "Handle TICKET-1111 and TICKET-2222 today"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        self.assertIn(self.label, result.details)
        entities = result.details[self.label]
        self.assertIn("TICKET-1111", entities)
        self.assertIn("TICKET-2222", entities)

    # Test cases for reference code pattern: REF[A-Z0-9]{8}
    def test_reference_code_with_letters_and_digits(self):
        """Test detection of reference code with letters and digits."""
        text = "REFABC12345"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        self.assertIn(self.label, result.details)
        self.assertIn(text, result.details[self.label])

    def test_reference_code_all_digits(self):
        """Test detection of reference code with all digits."""
        text = "REF12345"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        self.assertIn(self.label, result.details)
        self.assertIn(text, result.details[self.label])

    def test_reference_code_all_uppercase(self):
        """Test detection of reference code with all uppercase letters."""
        text = "REFABCDE"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        self.assertIn(self.label, result.details)
        self.assertIn(text, result.details[self.label])

    def test_reference_code_minimum_length(self):
        """Test that reference code requires at least 5 characters after REF."""
        text = "REFABC1234"  # Exactly 5 characters after REF
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        self.assertIn(self.label, result.details)
        self.assertIn(text, result.details[self.label])

    def test_reference_code_too_short(self):
        """Test that reference code with less than 5 characters is not detected."""
        text = "REFABC1"  # Only 4 characters after REF
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        if self.label in result.details:
            self.assertNotIn(text, result.details[self.label])

    def test_reference_code_in_sentence(self):
        """Test reference code detection in context."""
        text = "Your reference number is REFTEST1234"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        self.assertIn(self.label, result.details)
        self.assertIn("REFTEST1234", result.details[self.label])

    def test_reference_code_lowercase_not_detected(self):
        """Test that lowercase 'ref' prefix is not detected."""
        text = "refabcdefgh"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        # Lowercase 'ref' should not be detected
        self.assertNotIn(text, result.details.get(self.label, []))

    # Test edge cases and combinations
    def test_empty_text(self):
        """Test that empty text produces no detections."""
        text = ""
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        self.assertNotIn(self.label, result.details)

    def test_text_with_no_matches(self):
        """Test text with no matching patterns."""
        text = "This is a normal sentence with no special codes."
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        self.assertNotIn(self.label, result.details)

    def test_mixed_patterns_in_one_text(self):
        """Test detection of different pattern types in one text."""
        text = "Processing tickets ABCxyz123, TICKET-4567, REFTEST1234."
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        self.assertIn(self.label, result.details)
        entities = result.details[self.label]
        self.assertIn("ABCxyz123", entities)
        self.assertIn("TICKET-4567", entities)
        self.assertIn("REFTEST1234", entities)

    def test_pattern_at_text_boundaries(self):
        """Test patterns at the beginning and end of text."""
        text = "ABCxyz123 in middle TICKET-9999"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        self.assertIn(self.label, result.details)
        entities = result.details[self.label]
        self.assertIn("ABCxyz123", entities)
        self.assertIn("TICKET-9999", entities)

    def test_anonymization_output(self):
        """Test that detected patterns are properly anonymized."""
        text = "Code is ABCxyz123"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        # The anonymized text should not contain the original code
        self.assertNotIn("ABCxyz123", result.anonymized_text)
        # Should contain replacement marker
        self.assertIn("<EXAMPLE>", result.anonymized_text)


if __name__ == '__main__':
    unittest.main()

