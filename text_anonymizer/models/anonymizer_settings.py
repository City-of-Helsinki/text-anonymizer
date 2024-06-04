from typing import List


class AnonymizerSettings():

    score_threshold: float = 0.5

    mask_mapppings: dict = {}

    mask_mappings_debug: dict = {}

    operator_config: dict = {}

    recognizer_configuration: List[str] = []


