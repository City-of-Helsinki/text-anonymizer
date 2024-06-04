import unittest

import test_data
from base_recoginizer_test import BaseRecognizerTest
from text_anonymizer.recognizers.fi_registration_plate_recognizer import FiRegistrationPlateRecognizer

class TestCarRecognizer(unittest.TestCase):

    def test_self(self):
        recognizer_fi = FiRegistrationPlateRecognizer()
        test_cases = test_data.test_register_number
        bad_cases = test_data.bad_register_number
        test_base = BaseRecognizerTest(recognizer_fi, test_cases, bad_cases)
        self.assertTrue(test_base.test_recognizer(), 'Recognizer self test failed.')


if __name__ == '__main__':
    unittest.main()