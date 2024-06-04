from pydantic import BaseModel


class AnonymizerResult(BaseModel):
    anonymized_text: str = None
    statistics: dict = None
    details: dict = None