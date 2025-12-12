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
        deny_list: Optional[List[str]] = None,
    ):

        context = context if context else self.CONTEXT

        if deny_list is None:
            deny_list = []

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
        """Analyze text for deny list items, supporting multi-word phrases.

        Matches both single-word and multi-word deny list items by checking
        sequences of tokens against the deny list.

        :param text: Text to analyze
        :param entities: Entities to consider
        :param nlp_artifacts: NLP artifacts with tokens
        :return: List of RecognizerResult objects
        """
        results = []
        if not nlp_artifacts:
            # No nlp artifacts provided, cannot process
            return results

        # Create a list of deny list items sorted by word count (longest first)
        # This ensures multi-word phrases are matched before single words
        sorted_deny_list = sorted(self.deny_list, key=lambda x: len(x.split()), reverse=True)

        # Iterate through tokens and try to match deny list items
        i = 0
        while i < len(nlp_artifacts.tokens):
            matched = False

            for deny_item in sorted_deny_list:
                deny_tokens = deny_item.split()
                deny_tokens_count = len(deny_tokens)

                # Check if we have enough remaining tokens
                if i + deny_tokens_count > len(nlp_artifacts.tokens):
                    continue

                # Get the sequence of tokens to compare
                current_sequence = nlp_artifacts.tokens[i : i + deny_tokens_count]
                current_text = " ".join([str(t.text).lower() for t in current_sequence])

                # Try exact match first (case-insensitive)
                ratio = 0
                match = False

                if current_text == deny_item.lower():
                    ratio = self.match_ratio
                    match = True
                elif current_text.startswith(deny_item.lower()):
                    ratio = self.match_ratio
                    match = True
                else:
                    # Try fuzzy match on the full phrase
                    ratio = fuzz.ratio(deny_item.lower(), current_text)
                    match = ratio > self.match_ratio

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

                    # Calculate character positions
                    start_char_index = nlp_artifacts.tokens_indices[current_sequence[0].i]
                    end_char_index = nlp_artifacts.tokens_indices[current_sequence[-1].i] + len(current_sequence[-1].text)

                    result = RecognizerResult(label, start_char_index, end_char_index, ratio, explanation)
                    results.append(result)

                    # Move past the matched tokens
                    i += deny_tokens_count
                    matched = True
                    break

            if not matched:
                # No match found for current token, move to next
                i += 1

        return results

    def invalidate_result(self, pattern_text: str) -> bool:
        return False