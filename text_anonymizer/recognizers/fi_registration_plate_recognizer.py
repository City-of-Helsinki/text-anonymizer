from typing import List, Optional

from presidio_analyzer import Pattern, PatternRecognizer


class FiRegistrationPlateRecognizer(PatternRecognizer):
    """Recognize Finnish motor vehicle registration plate or license number using regex.

    :param patterns: List of patterns to be used by this recognizer
    :param context: List of context words to increase confidence in detection
    :param supported_language: Language this recognizer supports
    :param supported_entity: The entity this recognizer can detect
    """

    PATTERNS = [
        Pattern("Registration plate car XXX-000", r"\b([A-Za-z]{3})[-]([0-9]{3})\b", 0.75),  # Auton rekisterikilpi
        Pattern("Registration plate motorcycle XX-000", r"\b([A-Za-z]{2})[-]([0-9]{3})\b", 0.75),  # Moottoripyörän rekisterikilpi
        Pattern("Registration plate diplomat XX-0000", r"\b([A-Za-z]{2})[-]([0-9]{4})\b", 0.5),  # Diplomaattikilpi
        Pattern("Registration plate car XXX-000", r"\b([A-Za-z]{3})[-\s]?([0-9]{3})\b", 0.5),  # Auton rekisterikilpi
    ]

    CONTEXT = [
        "rekisteri",
        "rekisterinumero",
        "rekkari",
        "car",
        "auto",
        "ajoneuvo",
        "kilpi"
    ]

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "fi",
        supported_entity: str = "FI_REGISTRATION_PLATE",
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
