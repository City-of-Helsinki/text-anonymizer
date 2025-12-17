from typing import List, Optional
from pydantic import BaseModel, Field

from text_anonymizer.default_settings import DEFAULT_SETTINGS


class AnonymizerApiResponse(BaseModel):
    anonymized_txt: str = None
    statistics: dict = None


class AnonymizerApiRequest(BaseModel):
    text: str = None
    languages: List[str] = Field(default=['fi'], example=['fi', 'en'])
    recognizers: List[str] = Field(default=[], example=[k for k in DEFAULT_SETTINGS.mask_mapppings.keys()])
    profile: Optional[str] = Field(default=None, description="Profile name for configuration set (e.g., 'palautteet', 'asiakaspalvelu')")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Hei, olen Erkki Esimerkki, example123 ryhmästä, erkki.esimerkki@example123.com. "
                        "Hi! My name is John Doe, group: example123, email:john.doe@example123.com",
                "languages": ["fi"],
                "recognizers": [k for k in DEFAULT_SETTINGS.mask_mapppings.keys()],
                "profile": "example"
            }
        }
