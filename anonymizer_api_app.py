from typing import List
import os
import logging

import uvicorn
from fastapi import FastAPI

from text_anonymizer import TextAnonymizer
from text_anonymizer.models.api_models import AnonymizerApiRequest, AnonymizerApiResponse
from text_anonymizer.config_watcher import ConfigWatcher
from text_anonymizer.config_cache import ConfigCache

logger = logging.getLogger(__name__)

anonymizer_api = FastAPI()
languages = ['fi']
debug = False
text_anonymizer = TextAnonymizer(languages=languages, debug_mode=debug)

# Enable/disable watcher via env var CONFIG_WATCHER_ENABLED (default: true)
WATCHER_ENABLED = os.getenv("CONFIG_WATCHER_ENABLED", "true").lower() == "true"
CONFIG_DIR = ConfigCache.instance().config_dir


def on_config_change():
    """Callback triggered when config files change."""
    global text_anonymizer
    logger.info("Config change detected, recreating anonymizer")
    text_anonymizer = TextAnonymizer(languages=languages, debug_mode=debug)


config_watcher = ConfigWatcher(CONFIG_DIR, enabled=WATCHER_ENABLED, on_change_callback=on_config_change)
config_watcher.start()
logger.info("Config watcher started (enabled=%s)", WATCHER_ENABLED)


@anonymizer_api.post("/anonymize")
def anonymize(request_data: AnonymizerApiRequest) -> AnonymizerApiResponse:
    anonymizer_result = text_anonymizer.anonymize(
        request_data.text,
        user_languages=request_data.languages,
        user_recognizers=request_data.recognizers,
        profile=request_data.profile,
    )

    response: AnonymizerApiResponse = AnonymizerApiResponse()
    response.anonymized_txt = anonymizer_result.anonymized_text
    response.statistics = anonymizer_result.statistics
    return response


@anonymizer_api.post("/anonymize_batch")
def anonymize_batch(request_data: List[AnonymizerApiRequest]) -> List[AnonymizerApiResponse]:
    responses = []
    for request in request_data:
        anonymizer_result = text_anonymizer.anonymize(
            request.text,
            user_languages=request.languages,
            user_recognizers=request.recognizers,
            profile=request.profile,
        )

        response: AnonymizerApiResponse = AnonymizerApiResponse()
        response.anonymized_txt = anonymizer_result.anonymized_text
        response.statistics = anonymizer_result.statistics
        responses.append(response)

    return responses


if __name__ == "__main__":
    uvicorn.run(anonymizer_api, host="0.0.0.0", port=8000)