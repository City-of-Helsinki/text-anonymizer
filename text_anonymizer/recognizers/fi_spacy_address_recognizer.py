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
        [
            {'ENT_TYPE': 'PERSON', 'OP': '+'},
            {"TEXT": {"REGEX": "([0-9]{1,4}})(\s)?([A-Za-z]{0,1})?(\s)?([0-9]{0,4})?"}, 'OP': '+'}
        ],
        [
            {'ENT_TYPE': 'LOC', 'OP': '+'},
            {"TEXT": {"REGEX": "([0-9]{1,4}})(\s)?([A-Za-z]{0,1})?(\s)?([0-9]{0,4})?"}, 'OP': '+'}
        ],
        [
            {'ENT_TYPE': 'GPE', 'OP': '+'},
            {"TEXT": {"REGEX": "([0-9]{1,4}})(\s)?([A-Za-z]{0,1})?(\s)?([0-9]{0,4})?"}, 'OP': '+'}
        ],
        [
            {'ENT_TYPE': 'PERSON', 'OP': '+'},
            {'ENT_TYPE': 'CARDINAL', 'OP': '+', 'IS_DIGIT': True},
            {'IS_ALPHA': True, 'OP': '+', 'LENGTH': 1},
            {'ENT_TYPE': 'CARDINAL', 'OP': '?'},
        ],
        [
            {'ENT_TYPE': 'LOC', 'OP': '+'},
            {'ENT_TYPE': 'CARDINAL', 'OP': '+', 'IS_DIGIT': True},
            {'IS_ALPHA': True, 'OP': '?', 'LENGTH': 1},
            {'ENT_TYPE': 'CARDINAL', 'OP': '?'},
        ],
        [
            {'ENT_TYPE': 'GPE', 'OP': '+'},
            {'ENT_TYPE': 'CARDINAL', 'OP': '+'},
            {'IS_ALPHA': True, 'OP': '+', 'LENGTH': 1, 'IS_DIGIT': True},
            {'ENT_TYPE': 'CARDINAL', 'OP': '?'},
        ]
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
            # Ensure nlp artifacts
            nlp = nlp_artifacts.nlp_engine.nlp['fi']
            matcher = Matcher(nlp.vocab)
            # Add match ID "Address" with no callback and one pattern
            c = 0
            for pattern in self.patterns:
                matcher.add(f"{self.RESULT_TYPE}{str(c)}", [pattern])
                c += 1

            # Iterate trough tokens
            doc = nlp_artifacts.tokens.doc
            matches = matcher(doc)
            start_index = 100000000
            end_index = 0
            best_label = ''
            span = None

            # DEBUG entity recognition
            # for e in doc.ents:
            #     print(e.text, e.label_)

            for match_id, start, end in matches:
                span = doc[start:end]  # The matched span

                first_entity = doc.ents[0].label_

                # Build result and explanation
                label = self.supported_entities[0]
                match_pattern = matcher.get(match_id)
                textual_explanation = f"First entity: {first_entity}, Pattern: {json.dumps(match_pattern)}"

                # Build explanation
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
                        # Match is for address end part inside span. Append it to start position of span in original text.
                        start_index = span.start_char + m.start()
                if span:
                    t = text[start_index:end_index]
                    result = RecognizerResult(label, start_index, end_index, self.score, analysis_explanation=explanation)
                    results.append(result)
        else:
            # No nlp artifacts provided, cannot process
            return results

        return results
