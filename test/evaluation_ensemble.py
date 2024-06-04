import unittest
import random
from datetime import datetime

import tabulate

import test_util_text_anonymizer


class EvaluationEnsemble(unittest.TestCase):

    def evaluate_words(self, iterations=5000):
        # Use fixed seed so training will always be the same
        random.seed(1234)
        word_test_result, anonymized_words = test_util_text_anonymizer.evaluate_anonymizer_with_plain_words(iterations=iterations)
        return word_test_result, anonymized_words

    def evaluate_names(self, iterations=5000):
        # Use fixed seed so training will always be the same
        random.seed(1234)
        name_test_result, unrecognized_names = test_util_text_anonymizer.evaluate_anonymizer_with_generated_names(iterations=iterations)
        return name_test_result, unrecognized_names

    def evaluate_streets(self, iterations=1000):
        # Use fixed seed so training will always be the same
        random.seed(1234)
        streets_test_result, unrecognized_streets = test_util_text_anonymizer.evaluate_anonymizer_with_streets(iterations=iterations)
        return streets_test_result, unrecognized_streets

    def test_evaluation(self):
        '''
        Evaluate the anonymizer with 5000 words, 5000 names and 1000 streets
        Build a table of evaluation results.
        '''
        word_samples = 5000
        name_samples = 5000
        street_samples = 1000

        word_test_result, anonymized_words = self.evaluate_words(word_samples)
        name_test_result, unrecognized_names = self.evaluate_names(name_samples)
        streets_test_result, unrecognized_streets = self.evaluate_streets(street_samples)

        results = [
            {"Test": "words", "Accuracy": word_test_result, "Missed": len(anonymized_words), "Samples": word_samples},
            {"Test": "names", "Accuracy": name_test_result, "Missed": len(unrecognized_names), "Samples": name_samples},
            {"Test": "streets", "Accuracy": streets_test_result, "Missed": len(unrecognized_streets), "Samples": street_samples}
        ]

        print("Evaluation results\n")
        print(f"\nDate: { datetime.now().strftime('%d.%m.%Y')}\n")
        tab_results = tabulate.tabulate(results, headers="keys", tablefmt="pipe")
        print(tab_results)

