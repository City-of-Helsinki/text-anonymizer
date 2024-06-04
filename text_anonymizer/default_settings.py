from presidio_anonymizer.entities import OperatorConfig

from text_anonymizer.constants import *
from text_anonymizer.models.anonymizer_settings import AnonymizerSettings

DEFAULT_SETTINGS = AnonymizerSettings()
DEFAULT_SETTINGS.score_threshold = 0.5
DEFAULT_SETTINGS.mask_mapppings = {
    'ADDRESS': 'OSOITE',
    'EMAIL_ADDRESS': 'SÄHKÖPOSTI',
    'FI_REGISTRATION_PLATE': 'REKISTERINUMERO',
    'PHONE_NUMBER': 'PUHELIN',
    'FI_SSN': 'HENKILÖTUNNUS',
    'IP_ADDRESS': 'IP-OSOITE',
    'IBAN_CODE': 'TILINUMERO',
    'OTHER': 'KIELTOLISTA_TUNNISTE',
    'REAL_PROPERTY_ID': 'KIINTEISTÖTUNNUS',
    'PERSON': 'NIMI',
    'GRANTLISTED': 'GRANTLISTED',
    'FILENAME': 'TIEDOSTONIMI'
}
DEFAULT_SETTINGS.mask_mappings_debug = {
    'ETUNIMI': 'VOIKKO_ETU_NIMI',
    'SUKUNIMI': 'VOIKKO_SUKU_NIMI',
    'FIRST_NAME': 'EN_ETUNIMI',
    'LAST_NAME': 'EN_SUKUNIMI',
    'NAME': 'EN_NIMI',
    'STREET_NAME': 'KADUNNIMI',
    'ADDRESS': 'OSOITE',
    'EMAIL_ADDRESS': 'SÄHKÖPOSTI',
    'FI_REGISTRATION_PLATE': 'REKISTERINUMERO',
    'PHONE_NUMBER': 'PUHELIN',
    'PHONENUMBER': 'PUHELIN',
    'FI_SSN': 'HENKILÖTUNNUS',
    'IP_ADDRESS': 'IP-OSOITE',
    'IBAN_CODE': 'TILINUMERO',
    'OTHER': 'TUNNISTE',
    'REAL_PROPERTY_ID': 'KIINTEISTÖTUNNUS',
    'PERSON': 'SPACY_NIMI',
}

DEFAULT_SETTINGS.operator_config = {
    "GRANTLISTED": OperatorConfig("mask", {'chars_to_mask': 0, 'masking_char': '*', 'from_end': False}),
}

DEFAULT_SETTINGS.recognizer_configuration = [RECOGNIZER_EMAIL,
                                             RECOGNIZER_PHONE,
                                             RECOGNIZER_SSN,
                                             RECOGNIZER_FILE_NAME,
                                             RECOGNIZER_IP,
                                             RECOGNIZER_IBAN,
                                             RECOGNIZER_REGISTRATION_PLATE,
                                             RECOGNIZER_BLOCKLIST,
                                             RECOGNIZER_GRANTLIST,
                                             RECOGNIZER_PROPERTY,
                                             RECOGNIZER_SPACY_FI,
                                             RECOGNIZER_SPACY_EN,
                                             RECOGNIZER_SPACY_ADDRESS
                                             ]

RECOGNIZER_CONFIGURATION_ALL = [RECOGNIZER_EMAIL,
                                RECOGNIZER_PHONE,
                                RECOGNIZER_SSN,
                                RECOGNIZER_FILE_NAME,
                                RECOGNIZER_IP,
                                RECOGNIZER_IBAN,
                                RECOGNIZER_REGISTRATION_PLATE,
                                RECOGNIZER_ADDRESS,
                                RECOGNIZER_BLOCKLIST,
                                RECOGNIZER_GRANTLIST,
                                RECOGNIZER_PROPERTY,
                                RECOGNIZER_SPACY_FI,
                                RECOGNIZER_SPACY_EN,
                                RECOGNIZER_SPACY_ADDRESS
                                ]

RECOGNIZER_CONFIGURATION_WEBAPP = [RECOGNIZER_EMAIL,
                                    RECOGNIZER_PHONE,
                                    RECOGNIZER_SSN,
                                    RECOGNIZER_FILE_NAME,
                                    RECOGNIZER_IP,
                                    RECOGNIZER_IBAN,
                                    RECOGNIZER_REGISTRATION_PLATE,
                                    RECOGNIZER_ADDRESS,
                                    RECOGNIZER_PROPERTY,
                                    RECOGNIZER_SPACY_FI,
                                    RECOGNIZER_SPACY_EN,
                                    RECOGNIZER_SPACY_ADDRESS
                                    ]