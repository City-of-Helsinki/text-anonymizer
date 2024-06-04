from presidio_analyzer.nlp_engine import NlpArtifacts
import regex as re
from text_anonymizer import geo_location_provider
from typing import List, Optional
from presidio_analyzer import Pattern, PatternRecognizer, RecognizerResult


class FiAddressRecognizer(PatternRecognizer):
    """Recognize Finnish municipality using deny list.

    :param patterns: List of patterns to be used by this recognizer
    :param context: List of context words to increase confidence in detection
    :param supported_language: Language this recognizer supports
    :param supported_entity: The entity this recognizer can detect
    """

    # TODO: read list from csv
    DENY_LIST = geo_location_provider.get_location_names()

    PATTERNS = [
        Pattern("STREET", r"([A-Za-z]+)(katu|tie|kuja|polku|gatan|vägen|väylä)(\s)([0-9]+)?(\s)?([A-Za-z])?(\s)?([0-9]+)?", 1.0),  # noqa E501
        Pattern("ZIP", r"\b([0-9]{5})\b", 0.75),  # noqa E501
    ]
    CONTEXT = [
        "osoite",
        "area",
        "address"
    ]

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "fi",
        supported_entity: str = "ADDRESS",
    ):
        patterns = patterns if patterns else self.PATTERNS
        context = context if context else self.CONTEXT
        deny_list = self.DENY_LIST
        super().__init__(
            supported_entity=supported_entity,
            patterns=patterns,
            context=context,
            deny_list=deny_list,
            supported_language=supported_language,
        )

    def analyze(
            self,
            text: str,
            entities: List[str],
            nlp_artifacts: NlpArtifacts = None,
            regex_flags: int = None,
    ) -> List[RecognizerResult]:
        return super().analyze(text, entities, nlp_artifacts, regex_flags=re.X)

    STOPWORDS = ['hteystieto']

    def invalidate_result(self, pattern_text: str) -> bool:
        """
        Check if the pattern text cannot be validated as a location entity.

        :param pattern_text: Text detected as pattern by regex
        :return: True if invalidated
        """
        pattern_text_lower = pattern_text.lower()
        for sw in self.STOPWORDS:
            if sw in pattern_text_lower:
                return True

        return False

