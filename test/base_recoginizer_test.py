from typing import List

from presidio_analyzer import RecognizerRegistry, AnalyzerEngine, EntityRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider

'''
Base test class for testing single recognizer
Pass recognizer object and test strings as constructor parameters
'''


class BaseRecognizerTest():

    def __init__(self, recognizer: EntityRecognizer, test_cases: List[str], bad_test_cases: List[str] = []):
        if recognizer is None or not test_cases:
            self.assertTrue(False, 'Required parameters missing. Include recognizer instance and test strings. '
                                   'Example: setup(FiSsnRecognizer(), [1234567-1234])')
        self.test_cases = test_cases
        self.bad_test_cases = bad_test_cases
        # Init analyzer engine
        registry = RecognizerRegistry()
        registry.add_recognizer(recognizer)
        config_file = "../text_anonymizer/config/languages-config.yml"
        provider = NlpEngineProvider(conf_file=config_file)
        nlp_engine = provider.create_engine()

        self.analyzer = AnalyzerEngine(
            registry=registry,
            supported_languages=["fi"],
            nlp_engine=nlp_engine)


    @staticmethod
    def get_min_start(res):
        min = 9999999
        if len(res) > 0:
            for r in res:
                if min > r.start:
                    min = r.start
        return min

    @staticmethod
    def get_max_start(res):
        max = 0
        if len(res) > 0:
            for r in res:
                if max < r.end:
                    max = r.end
        return max

    def test_recognizer(self, lang="fi"):
        test_pass = True
        for text in self.test_cases:
            current_test_pass = True
            res = self.analyzer.analyze(text=text, language=lang)
            # Check that recognizer returns valid analysis
            if not res:
                current_test_pass = False
            elif self.get_min_start(res) > 2:   # allow 2 characters from start
                current_test_pass = False
            elif self.get_max_start(res) < len(text) - 2:    # allow 2 characters from end
                current_test_pass = False
            print('Expecting result. Testing:', text, 'Analysis result:', res, 'Pass: ', current_test_pass)
            if not current_test_pass:
                test_pass = False
        for text in self.bad_test_cases:
            current_test_pass = True
            res = self.analyzer.analyze(text=text, language="fi")
            # Check that recognizer returns valid analysis
            if len(res) > 0:
                current_test_pass = False

            print('Expecting no result. Testing:', text, 'Analysis result:', res, 'Pass: ', current_test_pass)
            if not current_test_pass:
                test_pass = False
        return test_pass

