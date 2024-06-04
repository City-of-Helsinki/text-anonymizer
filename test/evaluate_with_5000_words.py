import unittest
import random

import test_util_text_anonymizer


class EvaluateWith5000Words(unittest.TestCase):
    '''

    Test for false positives. These common words should not be anonymized.
    Success is if words are as is after anonymization and not anonymized.
    '''

    def test_anonymizer(self):
        # Use fixed seed so training will always be the same
        random.seed(1234)
        threshold = 0.95
        word_test_result, anonymized_words = test_util_text_anonymizer.evaluate_anonymizer_with_plain_words(iterations=5000)
        self.assertTrue(word_test_result > threshold, "Name anonymizer test failed")
        print("Word not-anonymizer test passed")
        print(f"(Incorrectly) anonymized words: \n{anonymized_words}")

    if __name__ == '__main__':
        test_anonymizer()

'''
Log:

Finished. Iterations: 104000,  error rate: 1.95%, anonymized: 98.05%, partially anonymized: 0.0%

Finished. Iterations: 5000,  error rate: 1.98%, anonymized: 98.02%, partially anonymized: 0.0%

'''
