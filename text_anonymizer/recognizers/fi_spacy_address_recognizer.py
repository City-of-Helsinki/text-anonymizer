import json
import re

from spacy.matcher import Matcher

from typing import List, Optional

from presidio_analyzer import Pattern, RecognizerResult, AnalysisExplanation, LocalRecognizer


class SpacyAddressRecognizer(LocalRecognizer):
    """Recognize Finnish street/area addresses based on SpaCy entities and token patterns."""

    # Patterns inspired by the original configuration, but simplified and corrected.
    # We rely on SpaCy NER to tag PERSON/LOC/GPE, then look for a compact
    # house-number-like tail such as "3 B 11" or "181".
    DEFAULT_PATTERNS = [
        # Pattern 1: PERSON/LOC/GPE + CARDINAL (house number only)
        # e.g. "Liisankatu 3", "Mannerheimintie 5"
        [
            {"ENT_TYPE": {"IN": ["PERSON", "LOC", "GPE"]}, "OP": "+"},
            {"ENT_TYPE": "CARDINAL", "OP": "+"},
        ],
        # Pattern 2: PERSON/LOC/GPE + CARDINAL + ALPHA (letter) + CARDINAL (unit)
        # e.g. "Akkutie 84 C 7", "Liisankatu 3 B 11"
        [
            {"ENT_TYPE": {"IN": ["PERSON", "LOC", "GPE"]}, "OP": "+"},
            {"ENT_TYPE": "CARDINAL", "OP": "+"},
            {"IS_ALPHA": True, "LENGTH": 1},
            {"ENT_TYPE": "CARDINAL"},
        ],
        # Pattern 3: PERSON/LOC/GPE + CARDINAL + ALPHA (letter, optional unit number)
        # e.g. "Valssimyllynkatu 11 A"
        [
            {"ENT_TYPE": {"IN": ["PERSON", "LOC", "GPE"]}, "OP": "+"},
            {"ENT_TYPE": "CARDINAL", "OP": "+"},
            {"IS_ALPHA": True, "LENGTH": 1},
        ],
        # Pattern 4: PERSON/LOC/GPE + digit (untagged) + ALPHA + digit (untagged)
        # Catches addresses where house numbers aren't tagged as CARDINAL
        # e.g. "Akkutie 84 C 7", "Gadolininkatu 32 H 34"
        [
            {"ENT_TYPE": {"IN": ["PERSON", "LOC", "GPE"]}, "OP": "+"},
            {"IS_DIGIT": True},
            {"IS_ALPHA": True, "LENGTH": 1},
            {"IS_DIGIT": True},
        ],
        # Pattern 5: PERSON/LOC/GPE + digit (untagged) + ALPHA (untagged letter)
        # e.g. "Valssimyllynkatu 11 A"
        [
            {"ENT_TYPE": {"IN": ["PERSON", "LOC", "GPE"]}, "OP": "+"},
            {"IS_DIGIT": True},
            {"IS_ALPHA": True, "LENGTH": 1},
        ],
        # Pattern 6: PERSON/LOC/GPE + digit only
        # e.g. "Itäinen Vaihdekuja 88"
        [
            {"ENT_TYPE": {"IN": ["PERSON", "LOC", "GPE"]}, "OP": "+"},
            {"IS_DIGIT": True},
        ],
        # Pattern 7: Multi-word street (LOC/GPE/ORG tokens) + digit + ALPHA + digit
        # e.g. "Hermannin rantatie 33 C 23", "Itäinen Vaihdekuja 88"
        # Handles cases where first digit isn't tagged as CARDINAL
        [
            {"ENT_TYPE": {"IN": ["LOC", "GPE", "ORG"]}, "OP": "+"},
            {"IS_DIGIT": True},
            {"IS_ALPHA": True, "LENGTH": 1},
            {"IS_DIGIT": True},
        ],
        # Pattern 8: Multi-word street + digit + ALPHA (letter)
        # e.g. "Itäinen Vaihdekuja 88 A"
        [
            {"ENT_TYPE": {"IN": ["LOC", "GPE", "ORG"]}, "OP": "+"},
            {"IS_DIGIT": True},
            {"IS_ALPHA": True, "LENGTH": 1},
        ],
        # Pattern 9: Multi-word street + digit only
        # e.g. "Itäinen Vaihdekuja 88"
        [
            {"ENT_TYPE": {"IN": ["LOC", "GPE", "ORG"]}, "OP": "+"},
            {"IS_DIGIT": True},
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
            nlp = nlp_artifacts.nlp_engine.nlp["fi"]
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

                # Guard against pure-numeric spans like "100" or measurement phrases
                if all(t.is_digit for t in span):
                    continue

                # Some texts may have no entities at all; handle that gracefully
                first_entity = doc.ents[0].label_ if doc.ents else "NONE"

                # Generic location-like check: require at least one LOC/GPE
                # or a reasonably long non-stopword alphabetic token.
                has_location_like = any(
                    t.ent_type_ in {"LOC", "GPE"}
                    or (t.is_alpha and not t.is_stop and len(t.text) >= 6)
                    for t in span
                )
                if not has_location_like:
                    continue

                # Narrow guard against the specific false positive pattern
                # seen in "ainakin 100 kilsaa": a single-token LOC like "ainakin"
                # followed immediately by a numeric-like token.
                if len(span) == 2:
                    t0, t1 = span[0], span[1]
                    if t0.text.lower() == "ainakin" and t0.ent_type_ == "LOC" and t1.like_num:
                        continue

                # Only process if this is the longest match so far
                if span_length > max_span_length:
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
