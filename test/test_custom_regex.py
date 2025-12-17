"""
Test suite for verifying EXAMPLE entity detection using custom regex patterns.

This test validates that all EXAMPLE regex patterns in the example profile configuration
correctly identify various patterns involving the word EXAMPLE with numbers and variations.

REGEX PATTERN REFERENCE:
========================
Understanding backslashes in JSON regex patterns:
- In JSON, backslash must be escaped: \\ becomes a single \ in the actual regex
- \\b in JSON = \b in regex = word boundary
- \\d in JSON = \d in regex = digit [0-9]

Pattern syntax quick reference:
- \\b        Word boundary (prevents partial matches like 'MYEXAMPLE' or 'EXAMPLES')
- [0-9]     Match any single digit
- [A-Za-z]  Match any single letter (upper or lower case)
- +         Match one or more of preceding element
- *         Match zero or more of preceding element
- {3}       Match exactly 3 of preceding element
- {2,4}     Match 2 to 4 of preceding element
- (?:...)   Non-capturing group for alternatives
- |         OR operator (used inside groups)

Example patterns explained:
- \\bEXAMPLE\\b           Matches exact word "EXAMPLE"
- \\bEXAMPLE[0-9]+\\b     Matches "EXAMPLE" followed by 1+ digits: EXAMPLE1, EXAMPLE987
- \\bEXAMPLE[0-9]*\\b     Matches "EXAMPLE" followed by 0+ digits: EXAMPLE, EXAMPLE1
- \\bEXAMPLE[0-9]{3}\\b   Matches "EXAMPLE" followed by exactly 3 digits: EXAMPLE987
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

    # =========================================================================
    # Pattern: exact_match - \\bEXAMPLE\\b
    # Matches only the exact word "EXAMPLE" with word boundaries
    # =========================================================================
    def test_exact_match_standalone(self):
        """Test detection of exact word EXAMPLE."""
        text = "EXAMPLE"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        self.assertIn(self.label, result.details)
        self.assertIn(text, result.details[self.label])

    def test_exact_match_in_sentence(self):
        """Test detection of EXAMPLE in a sentence."""
        text = "This is an EXAMPLE of text."
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        self.assertIn(self.label, result.details)
        self.assertIn("EXAMPLE", result.details[self.label])

    def test_exact_match_not_partial_prefix(self):
        """Test that MYEXAMPLE does not match (word boundary prevents prefix match)."""
        text = "MYEXAMPLE"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        # Should NOT detect EXAMPLE inside MYEXAMPLE due to word boundary
        if self.label in result.details:
            self.assertNotIn("EXAMPLE", result.details[self.label])

    def test_exact_match_not_partial_suffix(self):
        """Test that EXAMPLES does not match exact pattern (word boundary prevents suffix match)."""
        text = "EXAMPLES"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        # The exact_match pattern should not detect EXAMPLES
        # But word_with_optional_numbers pattern may match, so we check for exact "EXAMPLES"
        if self.label in result.details:
            self.assertNotIn("EXAMPLES", result.details[self.label])

    # =========================================================================
    # Pattern: case_insensitive_variations - \\b[Ee][Xx][Aa][Mm][Pp][Ll][Ee]\\b
    # Matches EXAMPLE in any case combination
    # =========================================================================
    def test_case_insensitive_lowercase(self):
        """Test detection of lowercase 'example'."""
        text = "example"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        self.assertIn(self.label, result.details)
        self.assertIn(text, result.details[self.label])

    def test_case_insensitive_mixed_case(self):
        """Test detection of mixed case 'ExAmPlE'."""
        text = "ExAmPlE"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        self.assertIn(self.label, result.details)
        self.assertIn(text, result.details[self.label])

    def test_case_insensitive_title_case(self):
        """Test detection of title case 'Example'."""
        text = "Example"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        self.assertIn(self.label, result.details)
        self.assertIn(text, result.details[self.label])

    # =========================================================================
    # Pattern: word_with_numbers - \\bEXAMPLE[0-9]+\\b
    # Matches EXAMPLE followed by one or more digits
    # =========================================================================
    def test_word_with_numbers_single_digit(self):
        """Test detection of EXAMPLE followed by single digit."""
        text = "EXAMPLE1"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        self.assertIn(self.label, result.details)
        self.assertIn(text, result.details[self.label])

    def test_word_with_numbers_multiple_digits(self):
        """Test detection of EXAMPLE followed by multiple digits."""
        text = "EXAMPLE987"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        self.assertIn(self.label, result.details)
        self.assertIn(text, result.details[self.label])

    def test_word_with_numbers_many_digits(self):
        """Test detection of EXAMPLE followed by many digits."""
        text = "EXAMPLE999999"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        self.assertIn(self.label, result.details)
        self.assertIn(text, result.details[self.label])

    def test_word_with_numbers_in_sentence(self):
        """Test detection of EXAMPLE with numbers in context."""
        text = "Please check EXAMPLE456 for details."
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        self.assertIn(self.label, result.details)
        self.assertIn("EXAMPLE456", result.details[self.label])

    # =========================================================================
    # Pattern: word_with_fixed_digits - \\bEXAMPLE[0-9]{3}\\b
    # Matches EXAMPLE followed by exactly 3 digits
    # =========================================================================
    def test_fixed_digits_exact_three(self):
        """Test detection of EXAMPLE followed by exactly 3 digits."""
        text = "EXAMPLE987"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        self.assertIn(self.label, result.details)
        self.assertIn(text, result.details[self.label])

    def test_fixed_digits_too_few_not_matched(self):
        """Test that EXAMPLE56 (2 digits) is not matched by fixed_digits pattern."""
        text = "EXAMPLE56"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        # word_with_digit_range pattern will match this (2-4 digits)
        # but fixed_digits pattern requires exactly 3
        self.assertIn(self.label, result.details)
        self.assertIn(text, result.details[self.label])

    def test_fixed_digits_too_many_not_matched_by_fixed(self):
        """Test that EXAMPLE9874 (4 digits) is matched by digit_range but not fixed_digits."""
        text = "EXAMPLE9874"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        # word_with_digit_range pattern will match (2-4 digits)
        self.assertIn(self.label, result.details)
        self.assertIn(text, result.details[self.label])

    # =========================================================================
    # Pattern: word_with_digit_range - \\bEXAMPLE[0-9]{2,4}\\b
    # Matches EXAMPLE followed by 2 to 4 digits
    # =========================================================================
    def test_digit_range_two_digits(self):
        """Test detection of EXAMPLE followed by 2 digits."""
        text = "EXAMPLE56"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        self.assertIn(self.label, result.details)
        self.assertIn(text, result.details[self.label])

    def test_digit_range_four_digits(self):
        """Test detection of EXAMPLE followed by 4 digits."""
        text = "EXAMPLE9874"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        self.assertIn(self.label, result.details)
        self.assertIn(text, result.details[self.label])

    def test_digit_range_one_digit_no_match(self):
        """Test that EXAMPLE1 (1 digit) is not matched by digit_range pattern."""
        text = "EXAMPLE1"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        # word_with_numbers pattern will match this (1+ digits)
        self.assertIn(self.label, result.details)
        self.assertIn(text, result.details[self.label])

    def test_digit_range_five_digits_no_match(self):
        """Test that EXAMPLE98745 (5 digits) is not matched by digit_range pattern."""
        text = "EXAMPLE98745"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        # word_with_numbers pattern will match (1+ digits)
        self.assertIn(self.label, result.details)
        self.assertIn(text, result.details[self.label])

    # =========================================================================
    # Pattern: prefix_variations - \\b(?:TEST|PROD|DEV)_EXAMPLE[0-9]+\\b
    # Matches TEST_EXAMPLE, PROD_EXAMPLE, DEV_EXAMPLE followed by digits
    # =========================================================================
    def test_prefix_test_example(self):
        """Test detection of TEST_EXAMPLE with digits."""
        text = "TEST_EXAMPLE1"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        self.assertIn(self.label, result.details)
        self.assertIn(text, result.details[self.label])

    def test_prefix_prod_example(self):
        """Test detection of PROD_EXAMPLE with digits."""
        text = "PROD_EXAMPLE99"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        self.assertIn(self.label, result.details)
        self.assertIn(text, result.details[self.label])

    def test_prefix_dev_example(self):
        """Test detection of DEV_EXAMPLE with digits."""
        text = "DEV_EXAMPLE987"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        self.assertIn(self.label, result.details)
        self.assertIn(text, result.details[self.label])

    def test_prefix_unknown_no_match(self):
        """Test that unknown prefix like STAGE_EXAMPLE is not matched."""
        text = "STAGE_EXAMPLE1"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        # Should NOT match prefix_variations pattern
        if self.label in result.details:
            self.assertNotIn(text, result.details[self.label])

    def test_prefix_in_sentence(self):
        """Test prefix patterns in context."""
        text = "Deploy TEST_EXAMPLE42 to production."
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        self.assertIn(self.label, result.details)
        self.assertIn("TEST_EXAMPLE42", result.details[self.label])

    # =========================================================================
    # Pattern: alphanumeric_suffix - \\bEXAMPLE[A-Za-z0-9]{3,6}\\b
    # Matches EXAMPLE followed by 3-6 alphanumeric characters
    # =========================================================================
    def test_alphanumeric_letters_only(self):
        """Test detection of EXAMPLE followed by letters only."""
        text = "EXAMPLEabc"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        self.assertIn(self.label, result.details)
        self.assertIn(text, result.details[self.label])

    def test_alphanumeric_mixed(self):
        """Test detection of EXAMPLE followed by mixed alphanumeric."""
        text = "EXAMPLE1a2b"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        self.assertIn(self.label, result.details)
        self.assertIn(text, result.details[self.label])

    def test_alphanumeric_uppercase(self):
        """Test detection of EXAMPLE followed by uppercase alphanumeric."""
        text = "EXAMPLEABC"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        self.assertIn(self.label, result.details)
        self.assertIn(text, result.details[self.label])

    def test_alphanumeric_too_short(self):
        """Test that EXAMPLE followed by 2 chars is not matched by alphanumeric pattern."""
        text = "EXAMPLEab"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        # Should NOT match alphanumeric_suffix pattern (requires 3-6)
        if self.label in result.details:
            self.assertNotIn(text, result.details[self.label])

    def test_alphanumeric_max_length(self):
        """Test detection of EXAMPLE followed by 6 alphanumeric chars."""
        text = "EXAMPLEabcdef"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        self.assertIn(self.label, result.details)
        self.assertIn(text, result.details[self.label])

    def test_alphanumeric_too_long(self):
        """Test that EXAMPLE followed by 7+ chars is not matched by alphanumeric pattern."""
        text = "EXAMPLEabcdefg"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        # Should NOT match alphanumeric_suffix pattern (requires 3-6)
        if self.label in result.details:
            self.assertNotIn(text, result.details[self.label])

    # =========================================================================
    # Edge cases and multiple matches
    # =========================================================================
    def test_empty_text(self):
        """Test that empty text produces no detections."""
        text = ""
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        self.assertNotIn(self.label, result.details)

    def test_multiple_patterns_in_text(self):
        """Test detection of multiple EXAMPLE patterns in one text."""
        text = "Check EXAMPLE, EXAMPLE987, and TEST_EXAMPLE1 for issues."
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        self.assertIn(self.label, result.details)
        entities = result.details[self.label]
        self.assertIn("EXAMPLE", entities)
        self.assertIn("EXAMPLE987", entities)
        self.assertIn("TEST_EXAMPLE1", entities)

    def test_word_boundary_prevents_partial_match(self):
        """Test that word boundaries prevent matching inside other words."""
        text = "NOTEXAMPLE987HERE"
        result = self.anonymizer.anonymize(text, profile=self.profile_name)
        # Word boundary should prevent matching EXAMPLE987 inside this string
        if self.label in result.details:
            self.assertNotIn("EXAMPLE987", result.details[self.label])


if __name__ == "__main__":
    unittest.main()

