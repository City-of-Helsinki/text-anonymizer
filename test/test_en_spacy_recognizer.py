import unittest

from presidio_analyzer.predefined_recognizers import SpacyRecognizer

import test_data
from base_recoginizer_test import BaseRecognizerTest


class TestEnSpacyRecognizer(unittest.TestCase):

    def test_self(self):
        en_spacy_recognizer = SpacyRecognizer(ner_strength=0.50,
                                                   supported_language='en')
        test_names_en = test_data.test_names_en
        test_base = BaseRecognizerTest(en_spacy_recognizer, test_names_en)
        self.assertTrue(test_base.test_recognizer(lang="en"), 'Recognizer self test failed.')

if __name__ == '__main__':
    unittest.main()