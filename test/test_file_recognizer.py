import unittest

import test_data
from base_recoginizer_test import BaseRecognizerTest
from text_anonymizer.recognizers.filename_recognizer import FilenameRecognizer


class TestFileRecognizer(unittest.TestCase):

    def test_self(self):
        recognizer_fi = FilenameRecognizer()
        test_cases = test_data.test_filenames
        bad_cases = test_data.bad_filenames
        test_base = BaseRecognizerTest(recognizer_fi, test_cases, bad_cases)
        self.assertTrue(test_base.test_recognizer(), 'Recognizer self test failed.')


if __name__ == '__main__':
    unittest.main()