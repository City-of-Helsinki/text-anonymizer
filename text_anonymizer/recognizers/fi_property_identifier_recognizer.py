from typing import List, Optional

from presidio_analyzer import Pattern, PatternRecognizer


class FiRealPropertyIdentifierRecognizer(PatternRecognizer):
    """Recognize property identifier (=KITU, kiinteistötunnus) using regex.

    :param patterns: List of patterns to be used by this recognizer
    :param context: List of context words to increase confidence in detection
    :param supported_language: Language this recognizer supports
    :param supported_entity: The entity this recognizer can detect
    """

    PATTERNS = [
        Pattern("Real property identifier Finnish XXX-XXX-XXXX-XXXX",
                r"\b([0-9]{1,3})[-]([0-9]{1,3})[-]([0-9]{1,4})[-]([0-9]{1,4})\b", 0.7),
        Pattern("Real property identifier Finnish XXX-XXX-XXXX-XXXX-YYYY",
                r"\b([0-9]{1,3})[-]([0-9]{1,3})[-]([0-9]{1,4})[-]([0-9]{1,4})[-]([0-9A-Za-z]{1,4})\b", 0.7),
        Pattern("Real property identifier Finnish ZZZXXXYYYYQQQQ", r"\b([0-9]{14,19})\b", 0.3)
    ]

    CONTEXT = [
        "kiinteistö",
        "kiinteistötunnus",
        "talo",
        "real property",
    ]

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "fi",
        supported_entity: str = "FI_RPI",
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
        return False
