import unittest

from presidio_analyzer import RecognizerRegistry, AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider

from text_anonymizer.recognizers.fi_spacy_address_recognizer import SpacyAddressRecognizer


def build_analyzer(anonymize_full_address: bool):
    recognizer = SpacyAddressRecognizer(supported_language='fi', anonymize_full_string=anonymize_full_address)

    # Init analyzer engine
    registry = RecognizerRegistry()
    registry.add_recognizer(recognizer)
    config_file = "../text_anonymizer/config/languages-config.yml"
    provider = NlpEngineProvider(conf_file=config_file)
    nlp_engine = provider.create_engine()

    analyzer = AnalyzerEngine(
        registry=registry,
        supported_languages=["fi"],
        nlp_engine=nlp_engine)
    return analyzer


class TestFiSpacyAddressRecognizer(unittest.TestCase):

    def test_partial_address_anonymization(self):
        # Test data
        test_cases_partial = [('Liisankatu 3 B 11', 11), ('Agronominkatu 181, Rastila', 14),
                              ('Pohjoisesplanadi 11 00099 HELSINGIN KAUPUNKI', 17),
                              ('Mannerheimintie 6 B 00100 Helsinki', 16), ('mannerheimintie 5 A 00100 Helsinki', 16),
                              ('Valssimyllynkatu 11', 17), ('Wavulinintie 5 A 100', 13),
                              ]
        test_cases_none = ['Wavulinintien tien risteyksessä istuu iso koira',
                           'Meidän kotikadulla mannerheimintiellä ei ole aurattu',
                           'Voisitteko hoitaa lumityöt insinöörinkadulla',
                           'Tampereelta on Helsinkiin pitkä matka, ainakin 100km',
                           'Minä olen hiihtänyt 10 vuotta']

        # Test partial address anonymization
        anonymize_full_address = False
        analyzer = build_analyzer(anonymize_full_address)

        for text, index in test_cases_partial:
            print(f"Checking {text}.\t\t\t Expecting to be anonymized from: {text[index:]}")
            res = analyzer.analyze(text=text.lower(), language="fi")
            # Check that recognizer returns valid analysis
            self.assertGreater(len(res), 0, f"No recognizer result for: {text}")
            min = self.get_min_start(res)
            self.assertEqual(index, min, f"Correct: {text[index:]}, Incorrect: {text[min:]}, full text {text}")
            print(f"PASS")

        for text in test_cases_none:
            res = analyzer.analyze(text=text.lower(), language="fi")
            # Check that recognizer returns no analysis
            print(f"Checking {text}")
            if len(res) > 0:
                print(f"Test fails: {text} -> {res} = {text[res[0].start:res[0].end]}")
            self.assertEqual(len(res), 0, f"Got recognizer result for no reason {text}")
            print(f"PASS")


    def test_full_address_anonymization(self):
        # Test data
        test_cases_partial = [('Liisankatu 181, 00000 Helsinki', 0),
                              ('Mannerheimintie 5 A 00100 Helsinki', 0),
                              ('Osoitteessa mannerheimintie 5 A on ovi.', 12),
                              ('OSOITTEENI ON VALSSIMYLLYNKATU 11', 14),
                              ('Terveisin insinööri Nieminen insinöörinkatu 3 B', 29),
                              ('Wavulinintie 5 talon', 0)
                              ]
        test_cases_none = ['Wavulinintien tien risteyksessä istuu iso koira',
                           'Meidän kotikadulla mannerheimintiellä ei ole aurattu',
                           'Voisitteko hoitaa lumityöt insinöörinkadulla']

        # Test full address anonymization
        anonymize_full_address = True
        analyzer = build_analyzer(anonymize_full_address)

        for text, index in test_cases_partial:
            print(f"Checking {text}.\t\t\t Expecting to be anonymized: {text[index:]}")
            res = analyzer.analyze(text=text, language="fi")
            # Check that recognizer returns valid analysis
            self.assertGreater(len(res), 0, f"No recognizer result for: {text}")
            min = self.get_min_start(res)
            # self.assertEqual(min, index, f"Correct: {text[index:]} Incorrect: {text[min:]}")
            print(f"PASS")

        for text in test_cases_none:
            res = analyzer.analyze(text=text, language="fi")
            # Check that recognizer returns no analysis
            print(f"Checking {text}")
            self.assertEqual(len(res), 0, f"There should be recognizer result for: {text}")
            print(f"PASS")

    @staticmethod
    def get_min_start(res):
        vmin = 9999999
        if len(res) > 0:
            for r in res:
                if vmin > r.start:
                    vmin = r.start
        return vmin

    @staticmethod
    def get_max_start(res):
        vmax = 0
        if len(res) > 0:
            for r in res:
                if vmax < r.end:
                    vmax = r.end
        return vmax


if __name__ == '__main__':
    unittest.main()
