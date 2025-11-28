import unittest
import random

import test_util_text_anonymizer


class EvaluateWith5000Names(unittest.TestCase):

    def test_anonymizer(self):
        # Use fixed seed so training will always be the same
        random.seed(1234)
        threshold = 0.95

        name_test_result, unrecognized_names = test_util_text_anonymizer.evaluate_anonymizer_with_generated_names(5000)

        self.assertTrue(name_test_result > threshold, "Name anonymizer test failed")
        print("Name anonymizer test passed")
        print(f"Unrecognized names: \n{unrecognized_names}")

    if __name__ == '__main__':
        test_anonymizer()

"""
Core: 
Finished. Iterations: 5000,  error rate: 2.02%, anonymized: 92.72%, partially anonymized: 5.26%
Finished. Iterations: 5000,  error rate: 0.1%, anonymized: 96.74%, partially anonymized: 3.16%


Custom:
Finished. Iterations: 5000,  error rate: 0.48%, anonymized: 98.24%, partially anonymized: 1.28%
Finished. Iterations: 5000,  error rate: 0.32%, anonymized: 97.62%, partially anonymized: 2.06%

"""