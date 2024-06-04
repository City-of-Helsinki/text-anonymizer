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
Core: 
Finished. Iterations: 5000,  error rate: 2.02%, anonymized: 92.72%, partially anonymized: 5.26%

Custom:
Finished. Iterations: 5000,  error rate: 0.48%, anonymized: 98.24%, partially anonymized: 1.28%
"""