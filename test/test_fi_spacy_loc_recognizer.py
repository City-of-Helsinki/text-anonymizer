import unittest

from presidio_analyzer.predefined_recognizers import SpacyRecognizer

import test_data
from base_recoginizer_test import BaseRecognizerTest


class TestFiSpacyRecognizer(unittest.TestCase):

    def test_self(self):
        finnish_spacy_recognizer = SpacyRecognizer(ner_strength=0.8,
                                                   supported_language='fi', supported_entities=['LOCATION'])
        test_cases = test_data.test_street
        test_base = BaseRecognizerTest(finnish_spacy_recognizer, test_cases)
        self.assertTrue(test_base.test_recognizer(), 'Recognizer self test failed.')


if __name__ == '__main__':
    unittest.main()