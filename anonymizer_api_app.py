from typing import List

import uvicorn
from fastapi import FastAPI

from text_anonymizer import TextAnonymizer
from text_anonymizer.models.api_models import AnonymizerApiRequest, AnonymizerApiResponse

anonymizer_api = FastAPI()
languages = ['fi']
debug = False
text_anonymizer = TextAnonymizer(languages=languages, debug_mode=debug)


@anonymizer_api.post("/anonymize")
def anonymize(request_data: AnonymizerApiRequest) -> AnonymizerApiResponse:
    anonymizer_result = text_anonymizer.anonymize(
        request_data.text,
        user_languages=request_data.languages,
        user_recognizers=request_data.recognizers,
        profile=request_data.profile,
        custom_blocklist=request_data.custom_blocklist,
        custom_grantlist=request_data.custom_grantlist
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
            custom_blocklist=request.custom_blocklist,
            custom_grantlist=request.custom_grantlist
        )

        response: AnonymizerApiResponse = AnonymizerApiResponse()
        response.anonymized_txt = anonymizer_result.anonymized_text
        response.statistics = anonymizer_result.statistics
        responses.append(response)

    return responses


if __name__ == "__main__":
    uvicorn.run(anonymizer_api, host="0.0.0.0", port=8000)