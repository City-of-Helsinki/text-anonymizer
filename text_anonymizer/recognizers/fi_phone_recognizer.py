from typing import List, Optional

from presidio_analyzer import Pattern, PatternRecognizer


class FiPhoneRecognizer(PatternRecognizer):
    """Recognize phone numbers using regex.
    Patterns: +35891234567, +358 9 123 4567, 09 123 4567
    :param patterns: List of patterns to be used by this recognizer
    :param context: List of context words to increase confidence in detection
    :param supported_language: Language this recognizer supports
    :param supported_entity: The entity this recognizer can detect
    """

    PATTERNS = [
        Pattern("Phonenumber international", r"\b(\+?[0-9]{11,12})\b", 1.0),
        Pattern("Phonenumber international with spaces", r"\b(\+?[0-9]{2,3}\s?[0-9]{2,3}\s?[0-9]{1,3}\s?[0-9]{3}\s?[0-9]{4})\b", 0.7),
        Pattern("Phonenumber local with spaces 2-1-3-3-4", r"\b([0-9]{2,3}\s?[0-9]{1,3}\s?[0-9]{3,4}\s?[0-9]{3,4})\b", 0.7),
        Pattern("Phonenumber local with spaces 3-4-3", r"\b([0-9]{2,3}\s?[0-9]{3,4}\s?[0-9]{3,4})\b", 0.7),
        Pattern("Phonenumber local with spaces 2-5-3", r"\b([0-9]{2,3}\s?[0-9]{3,5}\s?[0-9]{3,5})\b", 0.6),
        Pattern("Organization number", r"\b(\(?[0-9]{2,3}\)?\s?[0-9]{5,6}\)?)\b", 0.6),
    ]

    CONTEXT = [
        "puhelin",
        "numero",
        "puhelinnumero"
    ]

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "fi",
        supported_entity: str = "PHONE_NUMBER",
    ):
        patterns = patterns if patterns else self.PATTERNS
        context = context if context else self.CONTEXT
        super().__init__(
            supported_entity=supported_entity,
            patterns=patterns,
            context=context,
            supported_language=supported_language,
        )

    def invalidate_result(self, pattern_text: str) -> bool:
        """
        Check if the pattern text cannot be validated as a Phone entity.

        :param pattern_text: Text detected as pattern by regex
        :return: True if invalidated
        """
        return False