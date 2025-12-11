import re
from typing import List, Optional

from presidio_analyzer import Pattern, PatternRecognizer, RecognizerResult


class RegexRecognizer(PatternRecognizer):
    """Recognize patterns using user-defined regular expressions.

    This recognizer allows flexible pattern matching using regex patterns
    with customizable entity types and confidence scores.

    :param patterns: List of Pattern objects with regex patterns
    :param supported_language: Language this recognizer supports
    :param supported_entity: The entity this recognizer can detect
    """

    DEFAULT_EXPLANATION = "Identified as custom regex pattern."

    def __init__(
        self,
        patterns: List[Pattern],
        supported_language: str = "fi",
        supported_entity: str = "CUSTOM_REGEX",
    ):
        super().__init__(
            supported_entity=supported_entity,
            patterns=patterns,
            supported_language=supported_language,
        )
        # Store entity type for use in analyze method
        self._entity_type = supported_entity

    def analyze(self, text, entities, nlp_artifacts=None):
        """Analyze text using regex patterns.

        :param text: Text to analyze
        :param entities: Entities to look for
        :param nlp_artifacts: NLP artifacts (not used in this recognizer)
        :return: List of RecognizerResult objects
        """
        results = []

        if not self.patterns:
            return results

        for pattern in self.patterns:
            try:
                matches = re.finditer(pattern.regex, text)
                for match in matches:
                    result = RecognizerResult(
                        entity_type=self._entity_type,
                        start=match.start(),
                        end=match.end(),
                        score=pattern.score,
                        recognition_metadata={
                            "pattern_name": pattern.name,
                            "pattern": pattern.regex,
                        },
                    )
                    results.append(result)
            except re.error as e:
                print(f"Error in regex pattern '{pattern.name}': {str(e)}")
                continue

        return results

