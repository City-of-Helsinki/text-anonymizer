from typing import List, Optional

from presidio_analyzer import Pattern, PatternRecognizer


class FilenameRecognizer(PatternRecognizer):
    """Recognize common file names using regex.

    :param patterns: List of patterns to be used by this recognizer
    :param context: List of context words to increase confidence in detection
    :param supported_language: Language this recognizer supports
    :param supported_entity: The entity this recognizer can detect
    """

    PATTERNS = [
        Pattern("File name", r"\b+(http)?(\w)+[.](txt|doc|xls|xlsx|docx|pdf|jpg|png|ppt|pptx)\b+", 0.70),  # Filename
        Pattern("File URL", r'[A-Za-z0-9]+://[A-Za-z0-9%-_]+(/[A-Za-z0-9%-_])*[.](txt|doc|xls|xlsx|docx|pdf|jpg|png|ppt|pptx)\b+(#|\\?)[A-Za-z0-9%-_&=]*', 0.75)
    ]

    CONTEXT = [
        "liite",
        "liitetiedosto"
    ]

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "fi",
        supported_entity: str = "FILENAME",
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
        Check if the pattern text cannot be validated as a car register number entity.

        :param pattern_text: Text detected as pattern by regex
        :return: True if invalidated
        """

        return False
