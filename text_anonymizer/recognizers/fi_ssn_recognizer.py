from typing import List, Optional

from presidio_analyzer import Pattern, PatternRecognizer


class FiSsnRecognizer(PatternRecognizer):
    """Recognize Finnish Social Security Number (SSN) using regex.

    :param patterns: List of patterns to be used by this recognizer
    :param context: List of context words to increase confidence in detection
    :param supported_language: Language this recognizer supports
    :param supported_entity: The entity this recognizer can detect
    """

    PATTERNS = [
        Pattern("SSN Finnish", r"\b([0-3]{1}[0-9]{1}[0-1]{1}[0-9]{1}[0-9]{2})([-+A]{1})([0-9]{3})([a-zA-Z0-9]{1})\b", 1.0),
        Pattern("SSN Finnish partial", r"\b([0-3]{1}[0-9]{1}[0-1]{1}[0-9]{1}[0-9]{2})([-+A]{1})", 0.6),
        Pattern("SSN Finnish partially censored", r"\b([0-3]{1}[0-9]{1}[0-1]{1}[0-9]{1}[0-9]{2})([-+A]{1})?([a-zA-Z]{3,4})?\b", 0.55),
        Pattern("SSN Finnish incomplete", r"\b([0-3]{1}[0-9]{1}[0-1]{1}[0-9]{1}[0-9]{2})([-+A]{1})?([0-9]{3})?([a-zA-Z0-9]{1})?\b", 0.5),
    ]

    CONTEXT = [
        "henkilötunnus",
        "hetu",
        "sotu"
        "sosiaaliturvatunnus",
        "social",
        "security",
        "ssn",
        "ssns",
        "ssn#",
        "ss#",
        "ssid",
    ]

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "fi",
        supported_entity: str = "FI_SSN",
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
        Check if the pattern text cannot be validated as a FI_SSN entity.

        :param pattern_text: Text detected as pattern by regex
        :return: True if invalidated
        """

        text_upper = pattern_text.upper()
        if text_upper.count("A") > 2 or text_upper.count("-") > 1 or text_upper.count("+") > 1:
            return True
        return False
