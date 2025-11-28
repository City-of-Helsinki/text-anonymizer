import unittest
import random

import test_util_text_anonymizer


class EvaluateWith1000Streets(unittest.TestCase):

    def test_anonymizer(self):
        # Use fixed seed so training will always be the same
        random.seed(1234)
        threshold = 0.95
        name_test_result, unrecognized_streets = test_util_text_anonymizer.evaluate_anonymizer_with_streets(1000)
        self.assertTrue(name_test_result > threshold, "Street anonymizer test failed")
        print("Street anonymizer test passed")
        print(f"Unrecognized streets: \n{unrecognized_streets}")

    if __name__ == '__main__':
        test_anonymizer()

"""
Test report for 1000 Finnish street addresses recognition :

Spacy model fi_core_news_lg: 
Finished. Iterations: 5000,  error rate: 2.02%, anonymized: 92.72%, partially anonymized: 5.26%

Model custom_spacy_model built with main_train.py:

- With entity ruler: Finished. Iterations: 1000,  error rate: 26.0%, anonymized: 74.0%, partially anonymized: 0.0%
- Without entity ruler: Finished. Iterations: 1000,  error rate: 68.7%, anonymized: 31.3%, partially anonymized: 0.0%

Model custom_spacy_model built with train_custom_spacy_model.py:
Finished. Iterations: 1000,  error rate: 1.3%, anonymized: 98.7%, partially anonymized: 0.0%
"""