from typing import List, Optional

from fuzzywuzzy import fuzz
from presidio_analyzer import Pattern, PatternRecognizer, RecognizerResult, AnalysisExplanation


class GenericWordListRecognizer(PatternRecognizer):
    """Recognize user defined words using deny list in TXT format

    :param patterns: List of patterns to be used by this recognizer
    :param context: List of context words to increase confidence in detection
    :param supported_language: Language this recognizer supports
    :param supported_entity: The entity this recognizer can detect
    """

    DEFAULT_EXPLANATION = "Identified as custom dictionary word by {ratio}% similarity."

    CONTEXT = "Custom dictionary"

    minimum_word_length = 10

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "fi",
        supported_entity: str = "CUSTOM",
        match_ratio=91,
        deny_list: [List[str]] = [],
    ):

        context = context if context else self.CONTEXT

        for w in deny_list:
            if len(w) < self.minimum_word_length:
                self.minimum_word_length = len(w)

        self.match_ratio = match_ratio
        super().__init__(
            supported_entity=supported_entity,
            patterns=None,
            context=context,
            deny_list=deny_list,
            supported_language=supported_language,
        )

    def analyze(self, text, entities, nlp_artifacts=None):
        results = []
        if not nlp_artifacts:
            # No nlp artifacts provided, cannot process
            return results
        # Iterate trough tokens
        ratio = 0
        for token in nlp_artifacts.tokens:
            t = str(token.text).lower()  # k is already lowercase
            if len(t) < self.minimum_word_length:
                continue
            for k in self.deny_list:
                match = False
                if t == k or t.startswith(k):
                    # exact match
                    match = True
                    ratio = self.match_ratio
                    break
                else:
                    # Try fuzzy match
                    ratio = fuzz.ratio(k, t)
                    if ratio > self.match_ratio:
                        # Fuzzy match
                        match = True
                        break
            if match:
                # Build result and explanation
                label = self.supported_entities[0]
                textual_explanation = self.DEFAULT_EXPLANATION.format(ratio=ratio)

                # Build explanation
                explanation = AnalysisExplanation(
                    recognizer=self.__class__.__name__,
                    original_score=ratio,
                    textual_explanation=textual_explanation,
                )
                start_char_index = nlp_artifacts.tokens_indices[token.i]
                end_char_index = start_char_index + len(t)
                result = RecognizerResult(label, start_char_index, end_char_index, ratio, explanation)
                results.append(result)
        return results

    def invalidate_result(self, pattern_text: str) -> bool:
        return False