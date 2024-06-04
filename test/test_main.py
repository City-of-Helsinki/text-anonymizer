import unittest
import test_util_text_anonymizer


class MainTest(unittest.TestCase):

    def test_anonymizer(self):
        anonymizer_test_result = test_util_text_anonymizer.test_naturaltext_anonymizer()
        self.assertTrue(anonymizer_test_result, "Composite anonymizer test failed")

    if __name__ == '__main__':
        test_anonymizer()
