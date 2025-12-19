import json
import re

from spacy.matcher import Matcher

from typing import List, Optional

from presidio_analyzer import Pattern, RecognizerResult, AnalysisExplanation, LocalRecognizer


class SpacyAddressRecognizer(LocalRecognizer):
    """Recognize user defined words using deny list in TXT format

    :param patterns: List of patterns to be used by this recognizer
    :param context: List of context words to increase confidence in detection
    :param supported_language: Language this recognizer supports
    :param supported_entity: The entity this recognizer can detect
    """
    # See https://demos.explosion.ai/matcher for explanation of patterns
    DEFAULT_PATTERNS = [
        # LOC/GPE-anchored address patterns with optional alpha prefix
        # e.g. "Agnes Sjöbergin katu 1", "Antti Mäen kuja 99 D 2"
        [
            {'IS_ALPHA': True, 'OP': '*'},
            {'ENT_TYPE': {'IN': ['LOC', 'GPE']}},
            {'IS_DIGIT': True},
        ],
        [
            {'IS_ALPHA': True, 'OP': '*'},
            {'ENT_TYPE': {'IN': ['LOC', 'GPE']}},
            {'IS_DIGIT': True},
            {'IS_ALPHA': True, 'LENGTH': 1},
        ],
        [
            {'IS_ALPHA': True, 'OP': '*'},
            {'ENT_TYPE': {'IN': ['LOC', 'GPE']}},
            {'IS_DIGIT': True},
            {'IS_ALPHA': True, 'LENGTH': 1},
            {'IS_DIGIT': True},
        ],
        # Fallback patterns that do not rely on ENT_TYPE, only on shape
        # e.g. "Bulevardi 18", "Hyttitie 27", "Hakaniemen halli 84"
        [
            {'IS_ALPHA': True, 'OP': '+'},
            {'IS_DIGIT': True},
        ],
        [
            {'IS_ALPHA': True, 'OP': '+'},
            {'IS_DIGIT': True},
            {'IS_ALPHA': True, 'LENGTH': 1},
            {'IS_DIGIT': True, 'OP': '?'},
        ],
    ]

    RESULT_TYPE = "Address_pattern_"

    CONTEXT = "ADDRESS"

    minimum_word_length = 10

    def __init__(
            self,
            patterns: Optional[List[Pattern]] = None,
            context: Optional[List[str]] = None,
            supported_language: str = "fi",
            supported_entity: str = "ADDRESS",
            score: float = 0.8,
            anonymize_full_string: bool = True
    ):

        context = context if context else self.CONTEXT
        self.patterns = patterns if patterns else self.DEFAULT_PATTERNS
        self.score = score
        self.anonymize_full_string = anonymize_full_string
        super().__init__(
            supported_entities=[supported_entity],
            supported_language=supported_language,
            context=context,
        )

    def analyze(self, text, entities, nlp_artifacts=None):
        results = []
        if nlp_artifacts and nlp_artifacts.nlp_engine and nlp_artifacts.nlp_engine.nlp:
            nlp = nlp_artifacts.nlp_engine.nlp['fi']
            matcher = Matcher(nlp.vocab)

            c = 0
            for pattern in self.patterns:
                matcher.add(f"{self.RESULT_TYPE}{str(c)}", [pattern])
                c += 1

            doc = nlp_artifacts.tokens.doc
            matches = matcher(doc)

            best_result = None
            max_span_length = 0

            for match_id, start, end in matches:
                span = doc[start:end]
                span_length = end - start

                # Only process if this is the longest match so far
                if span_length > max_span_length:
                    # Some texts may have no entities at all; handle that gracefully
                    first_entity = doc.ents[0].label_ if doc.ents else "NONE"
                    label = self.supported_entities[0]
                    match_pattern = matcher.get(match_id)
                    textual_explanation = (
                        f"First entity: {first_entity}, Pattern: {json.dumps(match_pattern)}"
                    )

                    explanation = AnalysisExplanation(
                        recognizer=self.__class__.__name__,
                        original_score=self.score,
                        textual_explanation=textual_explanation,
                    )

                    start_index = span.start_char
                    end_index = span.end_char

                    if not self.anonymize_full_string and span and span.text:
                        m = re.search(r"\d", span.text)
                        if m:
                            start_index = span.start_char + m.start()

                    if span:
                        best_result = RecognizerResult(
                            label,
                            start_index,
                            end_index,
                            self.score,
                            analysis_explanation=explanation,
                        )
                        max_span_length = span_length

            if best_result:
                results.append(best_result)

        return results
