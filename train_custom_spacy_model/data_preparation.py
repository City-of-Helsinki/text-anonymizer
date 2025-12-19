"""
ETL and Data Preparation Module for SpaCy NER Training

This module handles:
- Loading raw data from CSV/text files
- Generating training sentences
- Creating training examples with entity annotations
- Augmenting data with variations
- Generating negative examples
"""

import csv
import os
import random
from typing import List, Tuple, Optional

from spacy.training import Example
from spacy.training.iob_utils import offsets_to_biluo_tags

# Use fixed seed for reproducibility
random.seed(78608831)

# Entity type constants
STREET_ENTITY = 'LOC'
AREA_ENTITY = 'GPE'
NAME_ENTITY = 'PERSON'

# Toggle CARDINAL labels in street-related training data
# When False, street examples will not add CARDINAL entities for house numbers.
ENABLE_STREET_CARDINAL = False

# Configuration for data generation
DATA_CONFIG = {
    'areas_size': 150,
    'streets_size': 875,
    'streets_size_with_grammatical_variations': 395,
    'names_size': 1575,
    'negative_examples_size': 500,
    'mixed_person_street': 100,
    'mixed_person_area': 40,
}

def validate_entity_alignment(nlp, text: str, entities: List[List]) -> bool:
    """
    Validate that entities align with SpaCy's tokenization.
    Returns True if alignment is valid, False otherwise.

    Args:
        nlp: SpaCy language model
        text: The text containing entities
        entities: List of [start, end, label] entity annotations

    Returns:
        bool: True if all entities align properly with tokens
    """
    try:
        doc = nlp.make_doc(text)
        tags = offsets_to_biluo_tags(doc, entities)
        # If any tag is '-', it means misalignment
        return '-' not in tags
    except Exception:
        return False


def create_validated_example(nlp, text: str, entities: List[List]) -> Optional[Example]:
    """
    Create a SpaCy Example only if entities are properly aligned.
    Validates BEFORE creating doc to prevent warnings.

    Args:
        nlp: SpaCy language model
        text: The text containing entities
        entities: List of [start, end, label] entity annotations

    Returns:
        Example if valid, None if misaligned
    """
    # Check alignment first using make_doc (no pipeline, no warnings)
    try:
        doc = nlp.make_doc(text)
        tags = offsets_to_biluo_tags(doc, entities)
        if '-' in tags:
            return None
    except Exception:
        return None

    # Only process through full pipeline if alignment is valid
    try:
        doc = nlp(text)
        example = Example.from_dict(doc, {"entities": entities})
        return example
    except Exception:
        return None


def extract_suffix_from_pattern(pattern: str, placeholder: str) -> str:
    """
    Extract any suffix attached to a placeholder in a pattern string.

    Example: "{area}ssa" -> "ssa"
             "{street} {number}" -> ""

    Args:
        pattern: The pattern string with placeholders
        placeholder: The placeholder name to search for (e.g., 'area', 'street')

    Returns:
        The suffix string, or empty string if no suffix
    """
    import re
    regex_pattern = r'\{' + placeholder + r'\}(\w+)'
    match = re.search(regex_pattern, pattern)
    return match.group(1) if match else ""


def normalize_finnish_suffixes(text: str, base_word: str, suffixed_word: str) -> str:
    """
    Normalize common Finnish suffix mistakes in the text.
    Corrects grammatical errors where the suffix doesn't match the word stem.

    Examples:
        katulla -> kadulla
        tiella -> tiellä
    Args:
        text: The full text containing the suffixed word
        base_word: The base word without suffix (e.g., "katu", "tie")
        suffixed_word: The word with suffix as it appears in text (e.g., "katulla", "tiella")

    Returns:
        Text with corrected suffixes
    """
    corrections = {
        # katu
        'katulla': 'kadulla',
        'katussa': 'kadussa',
        'katun':  'kadun',
        'katulle': 'kadulle',

        # tie
        'tiella': 'tiellä',
        'tiessa': 'tiessä',

        # kuja
        'kujassa': 'kujalla',

        # polku
        'polkulla': 'polulla',
        'polkussa': 'polulla',
        'polkun': 'polun',
    }

    # Convert to lowercase for comparison and correction
    result = text
    for incorrect, correct in corrections.items():
        # Case-insensitive replacement, preserving original case
        import re
        pattern = re.compile(re.escape(incorrect), re.IGNORECASE)

        def replace_preserve_case(match):
            original = match.group()
            if original.isupper():
                return correct.upper()
            elif original[0].isupper():
                return correct[0].upper() + correct[1:]
            else:
                return correct

        result = pattern.sub(replace_preserve_case, result)

    return result


class DataLoader:
    """Loads raw data from files"""

    def __init__(self, base_dir: str = None):
        if base_dir is None:
            this_dir, _ = os.path.split(__file__)
            base_dir = this_dir

        self.base_dir = base_dir
        self.first_names = []
        self.last_names = []
        self.streets = []
        self.areas = []
        self.products = []
        self.organizations = []

    def load_all(self):
        """Load all data files"""
        self._load_names()
        self._load_streets()
        self._load_areas()
        self._load_products()
        self._load_organizations()

    def _load_names(self):
        """Load first and last names from CSV files"""
        first_names_file = os.path.join(self.base_dir, "../test/data/etunimet.csv")
        last_names_file = os.path.join(self.base_dir, "../test/data/sukunimet.csv")

        with open(first_names_file, 'r') as f:
            for line in csv.reader(f, delimiter=';'):
                name = line[0] if random.randint(1, 3) > 1 else line[0].lower()
                self.first_names.append(name)
                if len(self.first_names) >= 2000:
                    break

        with open(last_names_file, 'r') as f:
            for line in csv.reader(f, delimiter=';'):
                self.last_names.append(line[0])
                if len(self.last_names) >= 2000:
                    break

    def _load_streets(self):
        """Load street names from text file"""
        streets_file = os.path.join(self.base_dir, "../test/data/helsinki_kadunnimet.txt")

        with open(streets_file, 'r') as f:
            for line in csv.reader(f, delimiter=';'):
                street = line[0] if random.randint(1, 3) == 1 else line[0].lower()
                self.streets.append(street)

    def _load_areas(self):
        """Load area names from text file"""
        areas_file = os.path.join(self.base_dir, "../test/data/helsinki_alueet.txt")

        with open(areas_file, 'r') as f:
            for line in csv.reader(f, delimiter=';'):
                area = line[0] if random.randint(1, 3) == 1 else line[0].lower()
                self.areas.append(area)

    def _load_products(self):
        """Load product names from text file"""
        products_file = os.path.join(self.base_dir, "../test/data/tuotenimet.txt")

        with open(products_file, 'r') as f:
            for line in csv.reader(f, delimiter=';'):
                product = line[0] if random.randint(1, 3) == 1 else line[0].lower()
                self.products.append(product)

    def _load_organizations(self):
        """Load organization names from text file"""
        orgs_file = os.path.join(self.base_dir, "../test/data/organisaatiot.txt")

        with open(orgs_file, 'r') as f:
            for line in csv.reader(f, delimiter=';'):
                org = line[0] if random.randint(1, 3) == 1 else line[0].lower()
                self.organizations.append(org)


class SentenceGenerator:
    """Generates training sentences with entities"""

    # Import sentence templates from training script

    # from training_data import (
    #     SENTENCES_NAME, SENTENCES_STREETS, SENTENCES_AREAS,
    #     ADJECTIVES, ADVERBS
    # )

    @staticmethod
    def generate_sentence(entity_text: str, sentence_list: List[str], include_number: bool = False) -> Tuple[str, int, int, Optional[List[Tuple[int, int, str]]]]:
        """
        Generate a sentence with the entity at a specific position.
        Handles suffixes attached to the {s} placeholder (e.g., {s}ssa, {s}lla).
        Normalizes common Finnish suffix mistakes.

        Args:
            entity_text: The entity text to insert (e.g., street name)
            sentence_list: List of sentence templates to choose from
            include_number: If True, also find and track numbers in the sentence

        Returns:
            Tuple of (sentence, start_pos, end_pos, number_infos)
            where number_infos is a list of (number_start, number_end, number_str) or None
        """
        import re

        try:
            # Select a random sentence template
            sent = random.choice(sentence_list)

            # Extract suffix from template if {s} has one attached (e.g., {s}ssa -> "ssa")
            suffix = extract_suffix_from_pattern(sent, "s")

            # Replace placeholder with entity, eg. {s} -> kankaantie
            if '{s}' in sent:
                sent = sent.replace('{s}', entity_text)

            # Replace {number} placeholder if present with a random number
            if '{number}' in sent:
                number = random.randint(1, 150)
                sent = sent.replace('{number}', str(number))

            # Replace other placeholders
            if '{adj}' in sent:
                from training_data import ADJECTIVES
                sent = sent.replace('{adj}', random.choice(ADJECTIVES))
            if '{adv}' in sent:
                from training_data import ADVERBS
                sent = sent.replace('{adv}', random.choice(ADVERBS))

            # Normalize Finnish suffix mistakes BEFORE finding entity position
            # This ensures positions are calculated on the normalized text
            if suffix:
                sent = normalize_finnish_suffixes(sent, entity_text, entity_text + suffix)

            # Now find the entity in the normalized text
            # The entity might have been corrected by normalization (e.g., katulla -> kadulla)
            # So we search for the base entity and the suffix together
            search_text = entity_text + suffix
            entity_start = sent.find(search_text)

            if entity_start == -1:
                # Fallback: just search for the base entity
                entity_start = sent.find(entity_text)
                if entity_start == -1:
                    # Text was normalized and entity not found, retry
                    return SentenceGenerator.generate_sentence(entity_text, sentence_list, include_number)
                entity_end = entity_start + len(entity_text) + len(suffix)
            else:
                entity_end = entity_start + len(search_text)

            # Find all numbers in the sentence if requested
            number_infos = None
            if include_number:
                number_infos = []
                # Find standalone numbers with optional Finnish suffixes like :n, :ssa, :lle, :sta
                # These suffixes are attached with colon and SpaCy tokenizes them together
                # Pattern matches: 10, 10:n, 10:ssa, 10:lle, 10:sta, 10:ssä, 10:ltä, etc.
                for match in re.finditer(r'\b(\d{1,3})(:[a-zäöå]+)?\b', sent):
                    num_start = match.start()
                    num_end = match.end()  # Includes the suffix if present
                    num_str = match.group()

                    # Skip if it overlaps with entity
                    if (num_start >= entity_start and num_start < entity_end) or \
                       (num_end > entity_start and num_end <= entity_end):
                        continue

                    # Skip postal codes (5 digit numbers starting with 0)
                    # Check context - if followed by more digits, skip
                    if num_end < len(sent) and sent[num_end:num_end+1].isdigit():
                        continue
                    if num_start > 0 and sent[num_start-1:num_start].isdigit():
                        continue

                    number_infos.append((num_start, num_end, num_str))

                # Only keep first number to avoid over-annotation
                if number_infos:
                    number_infos = number_infos[:1]
                else:
                    number_infos = None

            return sent, entity_start, entity_end, number_infos
        except ValueError:
            # Recursively try again if entity not found in generated sentence
            return SentenceGenerator.generate_sentence(entity_text, sentence_list, include_number)

    @staticmethod
    def generate_evaluation_sentence(value: str, sentence: str) -> Tuple[str, int, int]:
        """Generate an evaluation sentence with the entity at a specific position"""
        # Handle both {s} and {} format placeholders
        if '{s}' in sentence:
            sent = sentence.format(s=value)
        else:
            sent = sentence.format(value)

        s = sent.index(value)
        e = s + len(value)
        return sent, s, e

    @staticmethod
    def generate_full_names(first_names: List[str], last_names: List[str], amount: int = 1) -> List[str]:
        """Generate full names with variations using a 1-100 probability scale.

        Approximate target distribution (over many samples):
          - ~50% full name (first + last)
          - ~15% full name (last + first)
          - ~20% only first name
          - ~15% only last name
        Plus orthogonal variations:
          - ~10% have double or hyphenated first names
          - ~10% have hyphenated last names
        """
        full_names: List[str] = []

        for _ in range(amount):
            # Base first and last names
            first = random.choice(first_names)
            last = random.choice(last_names)

            # --- First name variation (up to ~10% combined) ---
            r_first = random.randint(1, 100)
            if r_first <= 5:
                # 1-5: double first name (e.g., "Matti Juhani")
                first = f"{first} {random.choice(first_names)}"
            elif r_first <= 10:
                # 6-10: hyphenated first name (e.g., "Anna-Kaisa")
                first = f"{first}-{random.choice(first_names)}"

            # --- Last name variation (up to ~10%) ---
            r_last = random.randint(1, 100)
            if r_last <= 10:
                # 1-10: hyphenated last name (e.g., "Virtanen-Koskinen")
                last = f"{last}-{random.choice(last_names)}"

            # --- Overall name format and order ---
            # Single draw in [1, 100] with simple brackets.
            r_format = random.randint(1, 100)
            if r_format <= 20:
                # 1-20: only first name (~20%)
                name = first
            elif r_format <= 35:
                # 21-35: only last name (~15%)
                name = last
            elif r_format <= 85:
                # 36-85: full name in "first last" order (~50%)
                name = f"{first} {last}"
            else:
                # 86-100: full name in "last first" order (~15%)
                name = f"{last} {first}"

            full_names.append(name)

        return full_names

    @staticmethod
    def augment_street_variations(street: str, max_variations: int = 2) -> List[str]:
        """Generate variations with Finnish case suffixes"""
        street_suffixes = ['llä', 'lle', 'lta', 'ltä', 'lla', 'n', 'ksi', 'ssa', 'ssä', 'sta', 'stä']
        available_suffixes = [s for s in street_suffixes if not street.endswith(s)]
        num_variations = min(max_variations, len(available_suffixes))
        if num_variations == 0:
            return []
        suffixes = random.sample(available_suffixes, num_variations)
        return [f"{street}{suffix}" for suffix in suffixes]


class TrainingDataGenerator:
    """Generates SpaCy training examples from raw data"""

    def __init__(self, nlp, data_loader: DataLoader):
        self.nlp = nlp
        self.data_loader = data_loader
        self.train_data = []

    def generate_all(self):
        """Generate all training data"""
        print("="*80)
        print("GENERATING TRAINING DATA")
        print("="*80)

        # Generate samples
        name_list = SentenceGenerator.generate_full_names(
            self.data_loader.first_names,
            self.data_loader.last_names,
            DATA_CONFIG['names_size']
        )
        street_list = random.sample(self.data_loader.streets, DATA_CONFIG['streets_size'])
        area_list = random.sample(self.data_loader.areas, DATA_CONFIG['areas_size'])

        print(f"Generating {DATA_CONFIG['names_size']} sentences with names")
        print(f"Generating {DATA_CONFIG['streets_size']} sentences with streets")
        print(f"Generating {DATA_CONFIG['streets_size_with_grammatical_variations']} sentences with streets using grammatical variations")
        print(f"Generating {DATA_CONFIG['areas_size']} sentences with areas")

        # Generate single-entity examples
        self._generate_name_examples(name_list)
        self._generate_street_examples(street_list)
        self._generate_street_examples_with_grammatical_variations(DATA_CONFIG['streets_size_with_grammatical_variations'])
        self._generate_area_examples(area_list)

        # Generate mixed-context examples
        self._generate_mixed_examples(name_list, street_list, area_list)

        # Generate negative examples
        self._generate_negative_examples()

        # Generate additional entity type examples (TIME, FAC, EVENT, etc.)
        if ENABLE_ADDITIONAL_ENTITY_TRAINING:
            self._generate_additional_entity_examples()

        print(f"\nTotal training examples generated: {len(self.train_data)}")
        print("="*80 + "\n")

        return self.train_data

    def _generate_name_examples(self, name_list: List[str]):
        """Generate training examples for person names"""
        from training_data import SENTENCES_NAME

        skipped = 0
        for name in name_list:
            sentence, start, end, _ = SentenceGenerator.generate_sentence(name, SENTENCES_NAME)
            entities = [[start, end, NAME_ENTITY]]

            example = create_validated_example(self.nlp, sentence, entities)
            if example:
                self.train_data.append(example)
            else:
                skipped += 1

        if skipped > 0:
            print(f"  Skipped {skipped} misaligned name examples")

    def _generate_street_examples(self, street_list: List[str]):
        """Generate training examples for street names with optional CARDINAL for numbers"""
        from training_data import SENTENCES_STREETS

        skipped = 0
        for street in street_list:
            street_lower = street.lower()

            # Choose sentence templates based on street name type
            # Use include_number=True to generate and track numbers in the sentence
            if ' ' not in street_lower and any(x in street_lower for x in ['katu', 'tie', 'polku']):
                sentence, start, end, number_infos = SentenceGenerator.generate_sentence(
                    street_lower, SENTENCES_STREETS, include_number=True
                )
            else:
                sentence, start, end, number_infos = SentenceGenerator.generate_sentence(
                    street_lower, SENTENCES_STREETS[:11], include_number=True
                )

            # Handle multi-word streets
            parts = street_lower.split(' ')
            entities = []
            if len(parts) > 1:
                i = start
                for p in parts:
                    j = i + len(p)
                    entities.append([i, j, STREET_ENTITY])
                    i = j + 1
            else:
                entities.append([start, end, STREET_ENTITY])

            # Add CARDINAL entities for numbers if present (configurable)
            if ENABLE_STREET_CARDINAL and number_infos:
                for num_start, num_end, _ in number_infos:
                    # Check it doesn't overlap with street entity
                    overlaps = False
                    for ent in entities:
                        ent_start, ent_end, _ = ent
                        if (num_start >= ent_start and num_start < ent_end) or \
                           (num_end > ent_start and num_end <= ent_end):
                            overlaps = True
                            break
                    if not overlaps:
                        entities.append([num_start, num_end, 'CARDINAL'])
                # Sort by position
                entities.sort(key=lambda x: x[0])

            example = create_validated_example(self.nlp, sentence, entities)
            if example:
                self.train_data.append(example)
            else:
                skipped += 1

            # Add LIMITED variations with Finnish case suffixes
            if ' ' not in street_lower and random.random() < 0.3:
                variations = SentenceGenerator.augment_street_variations(street_lower, max_variations=1)
                for street_var in variations:
                    sentence_var, start_var, end_var, number_infos_var = SentenceGenerator.generate_sentence(
                        street_var, SENTENCES_STREETS[:5], include_number=True
                    )
                    entities_var = [[start_var, end_var, STREET_ENTITY]]

                    # Add CARDINAL for variation too (configurable)
                    if ENABLE_STREET_CARDINAL and number_infos_var:
                        for num_start_var, num_end_var, _ in number_infos_var:
                            if not (num_start_var >= start_var and num_start_var < end_var) and \
                               not (num_end_var > start_var and num_end_var <= end_var):
                                entities_var.append([num_start_var, num_end_var, 'CARDINAL'])
                        entities_var.sort(key=lambda x: x[0])

                    example_var = create_validated_example(self.nlp, sentence_var, entities_var)
                    if example_var:
                        self.train_data.append(example_var)
                    else:
                        skipped += 1

        if skipped > 0:
            print(f"  Skipped {skipped} misaligned street examples")


    def _generate_street_examples_with_grammatical_variations(self, limit: int):
        """Generate training examples for street names using pre-inflected data."""
        from training_data import STREET_DATA_MAP

        skipped = 0
        generated_count = 0

        # Iterate through each grammatical case (nominative, genitive, etc.)
        for case_data in STREET_DATA_MAP:
            streets = case_data["streets"]
            templates = case_data["templates"]

            if not streets or not templates:
                continue

            # Create examples by combining each street with a random template from its case
            for street in streets:
                # The street is already inflected (e.g., "Kankaankatu", "Kankaanakadulla")
                sentence, start, end, number_infos = SentenceGenerator.generate_sentence(
                    street, templates, include_number=True
                )
                entities = [[start, end, STREET_ENTITY]]

                # Add CARDINAL entities for numbers if present (configurable)
                if ENABLE_STREET_CARDINAL and number_infos:
                    for num_start, num_end, _ in number_infos:
                        if not (num_start >= start and num_start < end) and \
                           not (num_end > start and num_end <= end):
                            entities.append([num_start, num_end, 'CARDINAL'])
                    entities.sort(key=lambda x: x[0])

                example = create_validated_example(self.nlp, sentence, entities)
                if example:
                    self.train_data.append(example)
                    generated_count += 1
                else:
                    skipped += 1
                if generated_count >= limit:
                    break
            print(f"- built {len(streets)} street examples for using {len(templates)} templates.")

        print(f"Generated {generated_count} street examples from pre-inflected data.")
        if skipped > 0:
            print(f"  Skipped {skipped} misaligned street examples")

    def _generate_area_examples(self, area_list: List[str]):
        """Generate training examples for area names"""
        from training_data import SENTENCES_AREAS

        skipped = 0
        for area in area_list:
            sentence, start, end, _ = SentenceGenerator.generate_sentence(area.lower(), SENTENCES_AREAS)
            entities = [[start, end, AREA_ENTITY]]

            example = create_validated_example(self.nlp, sentence, entities)
            if example:
                self.train_data.append(example)
            else:
                skipped += 1

        if skipped > 0:
            print(f"  Skipped {skipped} misaligned area examples")

    def _try_generate_person_street_example(self, name_list: List[str], street_list: List[str],
                                             all_patterns: List[str], max_retries: int = 3) -> Optional[Example]:
        """
        Try to generate a valid PERSON + STREET example, retrying with new random values if misalignment occurs.

        Args:
            name_list: List of person names to choose from
            street_list: List of streets to choose from
            all_patterns: List of sentence patterns
            max_retries: Maximum number of retries with new random values

        Returns:
            Example if successful, None if all retries failed
        """
        for attempt in range(max_retries):
            name = random.choice(name_list)
            street = random.choice(street_list)
            number = random.randint(1, 150)
            pattern = random.choice(all_patterns)

            # Extract any suffix attached to street in the pattern (e.g., {street}lla -> "lla")
            street_suffix = extract_suffix_from_pattern(pattern, "street")

            text = pattern.format(name=name, street=street.lower(), number=number)

            # Normalize Finnish suffix mistakes (e.g., katulla -> kadulla, tiella -> tiellä)
            # This must be done BEFORE calculating entity positions
            if street_suffix:
                text = normalize_finnish_suffixes(text, street.lower(), street.lower() + street_suffix)

            # Find entities in the NORMALIZED text
            name_start = text.find(name)
            if name_start == -1:
                continue
            name_end = name_start + len(name)

            street_lower = street.lower()
            # Try to find street with suffix first
            search_street_with_suffix = street_lower + street_suffix
            street_start = text.find(search_street_with_suffix)

            if street_start != -1:
                # Found with suffix - use the full span including suffix
                street_end = street_start + len(search_street_with_suffix)
            else:
                # Try without suffix - but include suffix in span if street was found
                street_start = text.find(street_lower)
                if street_start == -1:
                    continue
                # Always include the suffix in the span, even if not found as separate text
                street_end = street_start + len(street_lower) + len(street_suffix)

            # Verify exact matches in normalized text
            if text[name_start:name_end] != name:
                continue

            # Verify the street portion matches what we expect
            if street_end > len(text):
                continue

            # Find the number in text (if pattern includes {number})
            number_str = str(number)
            number_start = text.find(number_str)
            number_end = number_start + len(number_str) if number_start != -1 else -1

            # Create entities list (order by position)
            entities = []
            entity_positions = []

            # Add name entity
            entity_positions.append((name_start, [name_start, name_end, NAME_ENTITY]))

            # Add street entity
            entity_positions.append((street_start, [street_start, street_end, STREET_ENTITY]))

            # Add cardinal entity for number (if enabled and found and doesn't overlap with other entities)
            if ENABLE_STREET_CARDINAL and number_start != -1:
                # Check for overlap with street or name
                overlaps = False
                if (number_start >= street_start and number_start < street_end) or \
                   (number_end > street_start and number_end <= street_end):
                    overlaps = True
                if (number_start >= name_start and number_start < name_end) or \
                   (number_end > name_start and number_end <= name_end):
                    overlaps = True

                if not overlaps:
                    entity_positions.append((number_start, [number_start, number_end, 'CARDINAL']))

            # Sort entities by position and extract just the entity data
            entity_positions.sort(key=lambda x: x[0])
            entities = [ep[1] for ep in entity_positions]

            # Validate using spaCy before adding to training data
            example = create_validated_example(self.nlp, text, entities)
            if example:
                return example
            # If validation failed, loop will retry with new random values

        return None

    def _try_generate_person_area_example(self, name_list: List[str], area_list: List[str],
                                          all_patterns: List[str], max_retries: int = 3) -> Optional[Example]:
        """
        Try to generate a valid PERSON + AREA example, retrying with new random values if misalignment occurs.

        Args:
            name_list: List of person names to choose from
            area_list: List of areas to choose from
            all_patterns: List of sentence patterns
            max_retries: Maximum number of retries with new random values

        Returns:
            Example if successful, None if all retries failed
        """
        for attempt in range(max_retries):
            name = random.choice(name_list)
            area = random.choice(area_list)
            pattern = random.choice(all_patterns)

            # Extract any suffix attached to area in the pattern (e.g., {area}ssa -> "ssa")
            area_suffix = extract_suffix_from_pattern(pattern, "area")

            text = pattern.format(name=name, area=area.lower())

            # Normalize Finnish suffix mistakes (e.g., kujassa -> kujalla)
            # This must be done BEFORE calculating entity positions
            if area_suffix:
                text = normalize_finnish_suffixes(text, area.lower(), area.lower() + area_suffix)

            # Find entities in the NORMALIZED text
            name_start = text.find(name)
            if name_start == -1:
                continue
            name_end = name_start + len(name)

            area_lower = area.lower()
            # Try to find area with suffix first
            search_area_with_suffix = area_lower + area_suffix
            area_start = text.find(search_area_with_suffix)

            if area_start != -1:
                # Found with suffix - use the full span including suffix
                area_end = area_start + len(search_area_with_suffix)
            else:
                # Try without suffix - but include suffix in span if area was found
                area_start = text.find(area_lower)
                if area_start == -1:
                    continue
                # Always include the suffix in the span, even if not found as separate text
                area_end = area_start + len(area_lower) + len(area_suffix)

            # Verify exact matches in normalized text
            if text[name_start:name_end] != name:
                continue

            # Verify the area portion is within bounds
            if area_end > len(text):
                continue

            # Create entities list
            if name_start < area_start:
                entities = [
                    [name_start, name_end, NAME_ENTITY],
                    [area_start, area_end, AREA_ENTITY]
                ]
            else:
                entities = [
                    [area_start, area_end, AREA_ENTITY],
                    [name_start, name_end, NAME_ENTITY]
                ]

            # Validate using spaCy before adding to training data
            # This filters out examples with tokenization misalignment
            example = create_validated_example(self.nlp, text, entities)
            if example:
                return example
            # If validation failed, loop will retry with new random values

        return None

    def _generate_mixed_examples(self, name_list: List[str], street_list: List[str], area_list: List[str]):
        """Generate mixed context examples with multiple entity types"""
        from training_data import (
            MIXED_PATTERNS_PERSON_STREET,
            MIXED_PATTERNS_PERSON_STREET_EXTENDED,
            MIXED_PATTERNS_PERSON_AREA
        )

        print("\n" + "="*80)
        print("Generating MIXED CONTEXT examples (PERSON + STREET/AREA)")
        print("="*80)

        # PERSON + STREET examples
        mixed_person_street_count = 0
        mixed_person_street_skipped = 0

        all_patterns = MIXED_PATTERNS_PERSON_STREET + MIXED_PATTERNS_PERSON_STREET_EXTENDED

        for _ in range(500):
            if mixed_person_street_count >= DATA_CONFIG['mixed_person_street']:
                break

            # Try to generate a valid example, retrying with new random values on misalignment
            example = self._try_generate_person_street_example(name_list, street_list, all_patterns, max_retries=3)
            if example:
                self.train_data.append(example)
                mixed_person_street_count += 1
            else:
                mixed_person_street_skipped += 1

        print(f"  Added {mixed_person_street_count} PERSON + STREET examples (skipped {mixed_person_street_skipped})")

        # PERSON + AREA examples
        mixed_person_area_count = 0
        mixed_person_area_skipped = 0

        for _ in range(150):
            if mixed_person_area_count >= DATA_CONFIG['mixed_person_area']:
                break

            # Try to generate a valid example, retrying with new random values on misalignment
            example = self._try_generate_person_area_example(name_list, area_list, MIXED_PATTERNS_PERSON_AREA, max_retries=3)
            if example:
                self.train_data.append(example)
                mixed_person_area_count += 1
            else:
                mixed_person_area_skipped += 1

        print(f"  Added {mixed_person_area_count} PERSON + AREA examples (skipped {mixed_person_area_skipped})")

        print(f"  Total mixed context examples: {mixed_person_street_count + mixed_person_area_count}")
        print("="*80 + "\n")

    def _generate_negative_examples(self):
        """Generate negative examples (sentences with no entities)"""
        from training_data import FALSE_POSITIVES, NEGATIVE_TEMPLATES, ADJECTIVES, ADVERBS

        # Generate additional negative examples from templates
        negative_examples = list(FALSE_POSITIVES)
        for _ in range(100):
            template = random.choice(NEGATIVE_TEMPLATES)
            sentence = template.format(
                adj=random.choice(ADJECTIVES),
                adv=random.choice(ADVERBS)
            )
            negative_examples.append(sentence)

        print(f"Generated {len(negative_examples)} negative examples (sentences with no entities)")

        for sentence in negative_examples:
            doc = self.nlp(sentence)
            # Negative examples have NO entities - empty list
            entities = []

            example = Example.from_dict(doc, {"entities": entities})
            self.train_data.append(example)

    def _generate_additional_entity_examples(self):
        """Generate training examples for additional entity types.

        These sentences are DIFFERENT from evaluation.py to avoid data leakage.
        They anchor the labels TIME, FAC, EVENT, PRODUCT, NORP, ORDINAL,
        WORK_OF_ART, QUANTITY, MONEY, PERCENT so we can monitor if
        PERSON/LOC fine-tuning interferes with them.
        """
        # Training sentences - intentionally simple to align with spaCy tokenization
        samples = [
            # TIME – single time phrases
            ("Kokous alkaa kello yhdeksältä.", [[13, 29, "TIME"]]),   # "kello yhdeksältä"
            ("Bussi lähtee kello kolmelta.", [[13, 27, "TIME"]]),     # "kello kolmelta"

            # FAC – simple building/place names without hyphens
            ("Tapaaminen on kaupungintalolla.", [[14, 30, "FAC"]]),   # "kaupungintalolla"
            ("Kokous järjestetään Messukeskuksessa.", [[20, 36, "FAC"]]),  # "Messukeskuksessa"

            # EVENT – clear event names
            ("Talvifestivaali houkutteli tuhansia ihmisiä.", [[0, 15, "EVENT"]]),  # "Talvifestivaali"

            # PRODUCT – product/brand names
            ("Ostin uuden Lumme pesuaineen.", [[12, 17, "PRODUCT"]]),   # "Lumme"
            ("Leivoin kakun Blueberry Mix -jauheesta.", [[14, 27, "PRODUCT"]]),  # "Blueberry Mix"

            # ORDINAL – Finnish ordinals as single tokens
            ("Hän asuu talon kolmannessa kerroksessa.", [[15, 26, "ORDINAL"]]),  # "kolmannessa"
            ("Joukkue tuli neljänneksi kilpailussa.", [[13, 24, "ORDINAL"]]),    # "neljänneksi"

            # WORK_OF_ART – clear titles
            ("Luin kirjan Sateen jälkeen eilen.", [[12, 26, "WORK_OF_ART"]]),    # "Sateen jälkeen"
            ("Elokuva Hiljainen katu oli vaikuttava.", [[8, 22, "WORK_OF_ART"]]),  # "Hiljainen katu"

            # QUANTITY – simple measurement phrases
            ("Tarvitsemme kaksi litraa maitoa.", [[12, 31, "QUANTITY"]]),   # "kaksi litraa maitoa"
            ("Tilasimme viisi kiloa hiekkaa.", [[10, 29, "QUANTITY"]]),     # "viisi kiloa hiekkaa"

            # PERCENT – percentages as phrases
            ("Työttömyys laski viisi prosenttia.", [[17, 33, "PERCENT"]]),    # "viisi prosenttia"
        ]

        added = 0
        skipped = 0
        for text, entities in samples:
            example = create_validated_example(self.nlp, text, entities)
            if example:
                self.train_data.append(example)
                added += 1
            else:
                skipped += 1

        print(f"Generated {added} additional entity examples (TIME, FAC, EVENT, etc.)")
        if skipped > 0:
            print(f"  Skipped {skipped} misaligned additional entity examples")


# Toggle to include training examples for additional entity types (TIME, FAC, etc.)
# These use DIFFERENT sentences than evaluation.py to avoid data leakage
ENABLE_ADDITIONAL_ENTITY_TRAINING = True


def prepare_training_data(base_model: str = "fi_core_news_lg", seed: int = None) -> Tuple[list, object]:
    """
    Prepare training data for SpaCy NER fine-tuning.

    This is the main entry point for data preparation, orchestrating:
    1. Loading raw data (names, streets, areas, etc.)
    2. Generating training examples using TrainingDataGenerator
    3. Returning the training data and NLP model

    Args:
        base_model: Base SpaCy model name to load
        seed: Optional random seed for reproducibility

    Returns:
        Tuple of (training_examples, nlp_model)
            - training_examples: List of SpaCy Example objects
            - nlp_model: Loaded SpaCy language model
    """
    import spacy

    # Set random seed if provided
    if seed is not None:
        random.seed(seed)

    # Load base model
    print(f"Loading base model: {base_model}...")
    nlp = spacy.load(base_model)

    # Load raw data
    print("Loading raw data (names, streets, areas, etc.)...")
    data_loader = DataLoader()
    data_loader.load_all()

    # Generate training data
    print("Generating training examples...")
    generator = TrainingDataGenerator(nlp, data_loader)
    training_examples = generator.generate_all()

    return training_examples, nlp
