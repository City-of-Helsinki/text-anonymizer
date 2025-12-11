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
                "text": "Hei, olen Erkki Esimerkki, erkki.esimerkki@example.com. "
                        "Hi! My name is John Doe, john.doe@example.com",
                "languages": ["fi", "en"],
                "recognizers": [k for k in DEFAULT_SETTINGS.mask_mapppings.keys()],
                "profile": "palautteet"
            }
        }
