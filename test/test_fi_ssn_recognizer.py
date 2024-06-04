import unittest

import test_data
from base_recoginizer_test import BaseRecognizerTest
from text_anonymizer.recognizers.fi_ssn_recognizer import FiSsnRecognizer


class TestSSNRecognizer(unittest.TestCase):

    def test_self(self):
        ssn_recognizer_fi = FiSsnRecognizer()
        test_cases = test_data.test_ssn
        bad_cases = test_data.bad_ssn
        test_base = BaseRecognizerTest(ssn_recognizer_fi, test_cases, bad_cases)
        self.assertTrue(test_base.test_recognizer(), 'Recognizer self test failed.')


if __name__ == '__main__':
    unittest.main()