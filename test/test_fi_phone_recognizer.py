import unittest

from presidio_analyzer.predefined_recognizers import PhoneRecognizer

import test_data
from base_recoginizer_test import BaseRecognizerTest
from text_anonymizer.recognizers.fi_phone_recognizer import FiPhoneRecognizer


class TestPhoneRecognizer(unittest.TestCase):

    def test_phone(self):
        recognizer_fi = PhoneRecognizer(context=PhoneRecognizer.CONTEXT,
                                          supported_language='fi',
                                          supported_regions=("FI", "UK", "DE", "SE"))
        test_cases = test_data.test_phonenumbers
        bad_cases = test_data.bad_phonenumbers
        test_base = BaseRecognizerTest(recognizer_fi, test_cases, bad_cases)
        self.assertTrue(test_base.test_recognizer(), 'Recognizer self test failed.')

    def test_fi_phone(self):
        recognizer_fi = FiPhoneRecognizer(context=PhoneRecognizer.CONTEXT,
                                        supported_language='fi',)
        test_cases = test_data.test_phonenumbers_fi
        bad_cases = test_data.bad_phonenumbers
        test_base = BaseRecognizerTest(recognizer_fi, test_cases, bad_cases)
        self.assertTrue(test_base.test_recognizer(), 'Recognizer self test failed.')


if __name__ == '__main__':
    unittest.main()