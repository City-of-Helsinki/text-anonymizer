import os
from typing import List

from presidio_analyzer import RecognizerRegistry, AnalyzerEngine, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer.predefined_recognizers import EmailRecognizer, PhoneRecognizer, SpacyRecognizer, IpRecognizer, \
    IbanRecognizer
from presidio_anonymizer import AnonymizerEngine, ConflictResolutionStrategy

from text_anonymizer import default_settings
from text_anonymizer.constants import *
from text_anonymizer.custom_list_provider import get_block_list, get_grant_list
from text_anonymizer.models.anonymizer_result import AnonymizerResult
from text_anonymizer.models.anonymizer_settings import AnonymizerSettings
from text_anonymizer.recognizers.fi_address_recognizer import FiAddressRecognizer
from text_anonymizer.recognizers.fi_generic_word_list_recognizer import GenericWordListRecognizer
from text_anonymizer.recognizers.fi_phone_recognizer import FiPhoneRecognizer
from text_anonymizer.recognizers.fi_property_identifier_recognizer import FiRealPropertyIdentifierRecognizer
from text_anonymizer.recognizers.fi_registration_plate_recognizer import FiRegistrationPlateRecognizer
from text_anonymizer.recognizers.fi_spacy_address_recognizer import SpacyAddressRecognizer
from text_anonymizer.recognizers.fi_ssn_recognizer import FiSsnRecognizer
from text_anonymizer.recognizers.filename_recognizer import FilenameRecognizer


class TextAnonymizer:
    SUPPORTED_LANGUAGES = ['fi', 'en']

    this_dir, this_filename = os.path.split(__file__)
    CONFIG_FILE = os.path.join(this_dir, "config", "languages-config.yml")

    languages = None
    registry = None
    nlp_engine = None
    log_decision_process = False
    mask_mappings = None
    recognizer_configuration = None
    debug_mode = False
    score_threshold = 0.5
    operator_config = {}

    def __init__(self,
                 languages=['fi', 'en'],
                 settings: AnonymizerSettings = None,
                 recognizer_configuration: List[str] = None,
                 debug_mode=False, entity_mapping=None):
        """
        Anonymizer initialization.
        :param operator_config: Option to control how entities are labeled in anonymization
        :param languages: List of languages used in processing. See TextAnonymizer.SUPPORTED_LANGUAGES
        :param settings: Anonymizer settings. See default_settings.py
        :param recognizer_configuration: List of recognizers that used in analysis and anonymization. See default_settings.py
        :param debug_mode: True/False, debug mode gives more information about analysis results.
        """

        # debug mode
        self.debug_mode = debug_mode

        # Settings
        anonymizer_settings: AnonymizerSettings = default_settings.DEFAULT_SETTINGS

        # Operator config
        self.operator_config = anonymizer_settings.operator_config

        # Allow manipulation only to these setting parameters
        if settings:
            # Setup operator config
            if settings.operator_config:
                self.operator_config = settings.operator_config

            if settings.score_threshold:
                anonymizer_settings.score_threshold = settings.score_threshold
            if settings.recognizer_configuration:
                anonymizer_settings.recognizer_configuration = settings.recognizer_configuration

        self.score_threshold = anonymizer_settings.score_threshold

        # recognizer configuration
        self.recognizer_configuration = anonymizer_settings.recognizer_configuration
        if recognizer_configuration:
            self.recognizer_configuration = recognizer_configuration

        # Masking configuration
        if self.debug_mode == True:
            self.mask_mappings = anonymizer_settings.mask_mappings_debug
        else:
            self.mask_mappings = anonymizer_settings.mask_mapppings

        # Ensure that only supported languages are given
        languages_cleaned = []
        for lang in languages:
            if lang in self.SUPPORTED_LANGUAGES:
                languages_cleaned.append(lang)
        self.languages = languages_cleaned

        # Setup selected recognizers
        self.registry = RecognizerRegistry(supported_languages=self.languages)

        if RECOGNIZER_EMAIL in self.recognizer_configuration:
            email_recognizer_en = EmailRecognizer(supported_language='fi')
            self.registry.add_recognizer(email_recognizer_en)

        if RECOGNIZER_PHONE in self.recognizer_configuration:
            # Setting up a phone number recognizer
            phone_recognizer_fi = PhoneRecognizer(context=PhoneRecognizer.CONTEXT,
                                                  supported_language='fi',
                                                  supported_regions=("FI", "UK", "DE", "SE"))
            phone_recognizer_custom = FiPhoneRecognizer()
            self.registry.add_recognizer(phone_recognizer_fi)
            self.registry.add_recognizer(phone_recognizer_custom)

        if RECOGNIZER_SSN in self.recognizer_configuration:
            # Finnish hetu recognizer
            ssn_recognizer_fi = FiSsnRecognizer()
            self.registry.add_recognizer(ssn_recognizer_fi)

        if RECOGNIZER_FILE_NAME in self.recognizer_configuration:
            # File attachment recognizer
            file_recognizer_fi = FilenameRecognizer(supported_language='fi')
            self.registry.add_recognizer(file_recognizer_fi)

        if RECOGNIZER_IP in self.recognizer_configuration:
            # IP recognizer
            ip_recognizer = IpRecognizer(supported_language='fi')
            self.registry.add_recognizer(ip_recognizer)

        if RECOGNIZER_IBAN in self.recognizer_configuration:
            # IBAN recognizer
            iban_recognizer_fi = IbanRecognizer(supported_language='fi')
            self.registry.add_recognizer(iban_recognizer_fi)

        if RECOGNIZER_REGISTRATION_PLATE in self.recognizer_configuration:
            # Car register plates
            registration_plate_recognizer = FiRegistrationPlateRecognizer(supported_language='fi')
            self.registry.add_recognizer(registration_plate_recognizer)

        if RECOGNIZER_ADDRESS in self.recognizer_configuration:
            # Location recognizer
            location_recognizer = FiAddressRecognizer(supported_entity="ADDRESS",
                                                      supported_language='fi')
            self.registry.add_recognizer(location_recognizer)

        if RECOGNIZER_BLOCKLIST in self.recognizer_configuration:
            # Customized recognizer for user defined strings that should be anonymized
            deny_list = get_block_list()
            if len(deny_list) > 0:
                block_list_recognizer = GenericWordListRecognizer(supported_entity="OTHER", supported_language='fi',
                                                                      deny_list=deny_list)
                self.registry.add_recognizer(block_list_recognizer)

        if RECOGNIZER_GRANTLIST in self.recognizer_configuration:
            # Customized recognizer for user defined strings that should NOT be anonymized
            deny_list = get_grant_list()
            if len(deny_list) > 0:
                grant_list_recognizer = GenericWordListRecognizer(supported_entity="GRANTLISTED", supported_language='fi',
                                                                  deny_list=deny_list)
                self.registry.add_recognizer(grant_list_recognizer)

        if RECOGNIZER_PROPERTY in self.recognizer_configuration:
            # Real property id
            real_poperty_id_recognizer = FiRealPropertyIdentifierRecognizer(supported_entity="REAL_PROPERTY_ID",
                                                                            supported_language="fi")
            self.registry.add_recognizer(real_poperty_id_recognizer)

        if RECOGNIZER_SPACY_FI in self.recognizer_configuration:
            # Finnish spacy recognizer
            finnish_spacy_recognizer = SpacyRecognizer(ner_strength=0.90,
                                                       supported_entities=['PERSON', 'DATE'],
                                                       supported_language='fi')
            self.registry.add_recognizer(finnish_spacy_recognizer)

        if RECOGNIZER_SPACY_EN in self.recognizer_configuration:
            # English spacy recognizer
            english_spacy_recognizer = SpacyRecognizer(ner_strength=0.90,
                                                       supported_entities=["PERSON", "PHONE_NUMBER"],
                                                       supported_language='en')
            self.registry.add_recognizer(english_spacy_recognizer)

        if RECOGNIZER_SPACY_ADDRESS in self.recognizer_configuration:
            # Finnish spacy recognizer
            address_spacy_recognizer = SpacyAddressRecognizer(anonymize_full_string=False, supported_entity='ADDRESS')
            self.registry.add_recognizer(address_spacy_recognizer)

        # Init engines
        provider = NlpEngineProvider(conf_file=self.CONFIG_FILE)
        self.nlp_engine = provider.create_engine()

        self.anonymizer_engine = AnonymizerEngine()
        self.analyzer_engine = AnalyzerEngine(
            log_decision_process=self.log_decision_process,
            registry=self.registry,
            default_score_threshold=self.score_threshold,
            supported_languages=self.languages,
            nlp_engine=self.nlp_engine)

    def anonymize_text(self, text) -> str:
        """
        Anonymizes given text.
        :param text:
        :return:
        """
        result: AnonymizerResult = self.anonymize(text)
        # Suppress stats and details
        return result.anonymized_text if result else None

    def anonymize(self, text, user_languages: List[str] = None, user_recognizers: List[str] = None,
                  use_labels: bool = True) -> AnonymizerResult:
        """
        Anonymizes given text and returns also statistics about process.
        :param use_labels: Toggle custom labels on/off: True/False
        :param user_recognizers: List of recognizers to be used in request
        :param user_languages:  List of languages to be used in request
        :param text: Text to be anonymized
        :return: AnonymizerResult object
        """

        if not text:
            result: AnonymizerResult = AnonymizerResult()
            result.anonymized_text = None
            result.details = {}
            result.statistics = {}
            return result

        # Determine whether to use default languages or user defined
        languages = self.determine_languages(user_languages)

        analyzer_results: List[RecognizerResult] = []
        for lang in languages:
            # Get analyzer results
            a_results = self.analyzer_engine.analyze(text=text, language=lang)
            analyzer_results.extend(a_results)

        # Remove duplicates
        analyzer_results = self.anonymizer_engine._remove_conflicts_and_get_text_manipulation_data(analyzer_results, ConflictResolutionStrategy.MERGE_SIMILAR_OR_CONTAINED)
        # Remove unwanted recognizer results
        analyzer_results = self.filter_analyzer_results(analyzer_results, user_recognizers)

        # Add custom labels
        if use_labels:
            self.labelize(analyzer_results)

        # Use analyzer results in anonymization
        anonymizer_result = self.anonymizer_engine.anonymize(
            text=text,
            analyzer_results=analyzer_results,
            operators=self.operator_config
        )

        # Construct statistics
        statistics, details = self.build_statistics(analyzer_results, text)
        # Build result object
        result: AnonymizerResult = AnonymizerResult()
        result.anonymized_text = anonymizer_result.text if anonymizer_result.text else None
        result.details = details
        result.statistics = statistics

        return result

    def anonymize_with_statistics(self, text) -> (str, dict):
        text, stats, details = self.anonymize(text)
        return text, stats

    def labelize(self, analyzer_results):
        for ar in analyzer_results:
            e_type = ar.entity_type
            mapped_type = e_type
            if e_type in self.mask_mappings:
                mapped_type = self.mask_mappings[e_type]
            if self.debug_mode:
                ar.entity_type = "{t}_{s}".format(t=mapped_type, s=ar.score)
            else:
                ar.entity_type = mapped_type

    def determine_languages(self, user_defined_languages=[]):
        supported_languages = self.languages
        languages = []
        if user_defined_languages is not None and len(user_defined_languages) > 0:
            for l in user_defined_languages:
                if l in supported_languages:
                    languages.append(l)
        else:
            languages = supported_languages
        return languages

    @staticmethod
    def build_statistics(analyzer_results, text) -> (object, object):
        stats = {}
        details = {}
        for r in analyzer_results:
            entity = text[r.start:r.end]
            if r.entity_type in stats.keys():
                stats[r.entity_type] += 1
                details[r.entity_type].append(entity)
            else:
                stats[r.entity_type] = 1
                details[r.entity_type] = [entity]
        return stats, details

    @staticmethod
    def combine_statistics(statistics: []):
        combined_stats = {}
        for s in statistics:
            for k in s.keys():
                if k in combined_stats.keys():
                    combined_stats[k] += s[k]
                else:
                    combined_stats[k] = s[k]
        return combined_stats

    @staticmethod
    def combine_details(details: []):
        combined_details = {}
        for s in details:
            for k in s.keys():
                if k in combined_details.keys():
                    combined_details[k].append(s[k])
                else:
                    combined_details[k] = [s[k]]
        return combined_details

    def filter_analyzer_results(self, analyzer_results, user_recognizers):
        if user_recognizers is not None and len(user_recognizers) > 0:
            # Use only results matching recognizers-list
            analyzer_results_filtered = []
            for r in analyzer_results:
                if r.entity_type in user_recognizers:
                    analyzer_results_filtered.append(r)
            analyzer_results = analyzer_results_filtered
        return analyzer_results