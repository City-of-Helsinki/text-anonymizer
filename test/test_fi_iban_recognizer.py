import unittest

from presidio_analyzer.predefined_recognizers import IbanRecognizer

import test_data
from base_recoginizer_test import BaseRecognizerTest


class TestIBANRecognizer(unittest.TestCase):

    def test_self(self):
        recognizer_fi = IbanRecognizer(supported_language='fi')
        test_cases = test_data.test_iban
        bad_cases = test_data.bad_iban
        test_base = BaseRecognizerTest(recognizer_fi, test_cases, bad_cases)
        self.assertTrue(test_base.test_recognizer(), 'Recognizer self test failed.')


if __name__ == '__main__':
    unittest.main()