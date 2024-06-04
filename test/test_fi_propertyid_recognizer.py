import unittest

import test_data
from base_recoginizer_test import BaseRecognizerTest
from text_anonymizer.recognizers.fi_property_identifier_recognizer import FiRealPropertyIdentifierRecognizer


class TestPropertyIDRecognizer(unittest.TestCase):

    def test_self(self):
        recognizer_fi = FiRealPropertyIdentifierRecognizer()
        test_cases = test_data.test_property_identifier
        bad_cases = test_data.bad_property_identifier
        test_base = BaseRecognizerTest(recognizer_fi, test_cases, bad_cases)
        self.assertTrue(test_base.test_recognizer(), 'Recognizer self test failed.')


if __name__ == '__main__':
    unittest.main()