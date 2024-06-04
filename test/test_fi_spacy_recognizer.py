import unittest

from presidio_analyzer.predefined_recognizers import SpacyRecognizer

import test_data
from base_recoginizer_test import BaseRecognizerTest


class TestFiSpacyRecognizer(unittest.TestCase):

    def test_self(self):
        finnish_spacy_recognizer = SpacyRecognizer(ner_strength=0.50,
                                                   supported_language='fi')
        test_cases_fi = test_data.test_names_fi
        test_base = BaseRecognizerTest(finnish_spacy_recognizer, test_cases_fi)
        self.assertTrue(test_base.test_recognizer(), 'Recognizer self test failed.')

if __name__ == '__main__':
    unittest.main()