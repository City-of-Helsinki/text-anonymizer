import phonenumbers
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer.predefined_recognizers import EmailRecognizer, PhoneRecognizer, IpRecognizer, IbanRecognizer

from text_anonymizer.recognizers.fi_address_recognizer import FiAddressRecognizer
from text_anonymizer.recognizers.fi_property_identifier_recognizer import FiRealPropertyIdentifierRecognizer

'''
Example of using the Presidio analyzer with Finnish recognizers without the anoymizer class.
'''

TEST_TEXT = "Hello, my name is David. "

# from text_anonymizer.fi_ssn_recognizer import FiSsnRecognizer
from text_anonymizer.recognizers.fi_ssn_recognizer import FiSsnRecognizer

config_file = "../text_anonymizer/config/languages-config.yml"

# Create NLP engine based on configuration file
provider = NlpEngineProvider(conf_file=config_file)
nlp_engine = provider.create_engine()

# Add recognizers to registry
registry = RecognizerRegistry()

# Setting up an English Email recognizer:
email_recognizer_en = EmailRecognizer(supported_language="fi", context=["email", "mail"])
registry.add_recognizer(email_recognizer_en)
TEST_TEXT += "Please contact me to my email address: david.test@sonera.fi. "

# Setting up a Spanish Email recognizer
ip_recognizer = IpRecognizer(supported_language="fi")
registry.add_recognizer(ip_recognizer)
TEST_TEXT += "I'm using internet from address 22.191.177.111. "

# Setting up a phone number recognizer
phone_recognizer_fi = PhoneRecognizer(supported_language="fi", context=PhoneRecognizer.CONTEXT,
                                      supported_regions=phonenumbers.SUPPORTED_REGIONS)
registry.add_recognizer(phone_recognizer_fi)
TEST_TEXT += "My phone number is +358998877654. "

# Finnish hetu recognizer
ssn_recognizer_fi = FiSsnRecognizer()
registry.add_recognizer(ssn_recognizer_fi)
TEST_TEXT += "BTW, my ssn is 010188A100K, not 010188A100KK. "

# IBAN recognizer
iban_recognizer_fi = IbanRecognizer(supported_language="fi")
registry.add_recognizer(iban_recognizer_fi)
TEST_TEXT += "Tilinumeroni on FI49 5000 9420 0287 30 tai FI4950009420028731."

# Municipality recognizer
city_recognizer = FiAddressRecognizer(supported_entity="ADDRESS",
                                      supported_language='fi')

registry.add_recognizer(city_recognizer)
TEST_TEXT += " Osoitteeni on kauppakatu 1 40100 Föglö."

# Property identifier recognizer
poperty_id_recognizer = FiRealPropertyIdentifierRecognizer(supported_entity="PROPERTY_ID", supported_language="fi")
registry.add_recognizer(poperty_id_recognizer)
TEST_TEXT += " Kiinteistötunnukseni on 999-888-12-1, määräala: 999-999-12-44-M601"

# Run analyzer and print results
analyzer = AnalyzerEngine(
    registry=registry,
    supported_languages=["fi"],
    nlp_engine=nlp_engine)

print(TEST_TEXT)
res = analyzer.analyze(text=TEST_TEXT, language="fi")
for r in res:
    s = TEST_TEXT[r.start : r.end]
    print(">> ", s, r)
