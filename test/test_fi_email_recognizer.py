import unittest

from presidio_analyzer.predefined_recognizers import EmailRecognizer

import test_data
from base_recoginizer_test import BaseRecognizerTest


class TestEmailRecognizer(unittest.TestCase):

    def test_self(self):
        recognizer_fi = EmailRecognizer(supported_language='fi')
        test_cases = test_data.test_email
        bad_cases = test_data.bad_email
        test_base = BaseRecognizerTest(recognizer_fi, test_cases, bad_cases)
        self.assertTrue(test_base.test_recognizer(), 'Recognizer self test failed.')


if __name__ == '__main__':
    unittest.main()