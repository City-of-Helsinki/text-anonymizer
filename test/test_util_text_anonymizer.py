import csv
import os
import random
import time

import test_data
from text_anonymizer import TextAnonymizer
from text_anonymizer.constants import RECOGNIZER_SPACY_ADDRESS
from text_anonymizer.default_settings import RECOGNIZER_SPACY_FI, RECOGNIZER_GLINER_FI

ACCEPTED_ERROR_RATE = 0.05   # 5% error rate

this_dir, this_filename = os.path.split(__file__)

_FIRST_NAMES_FILE_PATH = os.path.join("data", "etunimet.csv")
_FIRST_NAMES_DATA_FILE = os.path.join(this_dir, _FIRST_NAMES_FILE_PATH)
_FIRST_NAMES = []

_LAST_NAMES_FILE_PATH = os.path.join("data", "sukunimet.csv")
_LAST_NAMES_DATA_FILE = os.path.join(this_dir, _LAST_NAMES_FILE_PATH)
_LAST_NAMES = []


_STREETS_FILE_PATH = os.path.join("data", "helsinki_kadunnimet.txt")
_STREETS_DATA_FILE = os.path.join(this_dir, _STREETS_FILE_PATH)
_STREETS = []

_WORDS_FILE_PATH = os.path.join("data", "nykysuomensanalista2022.csv")
_WORDS_DATA_FILE = os.path.join(this_dir, _WORDS_FILE_PATH)
_WORDS = []

TOP_FIRST_NAMES = 1000
TOP_LAST_NAMES = 1000
TOP_STREETS = 1000
TOP_WORDS = 104000


with open(_LAST_NAMES_DATA_FILE, 'r') as data:
    # This skips the first row of the CSV file.
    reader = csv.reader(data, delimiter=';')
    next(reader)
    for line in reader:
        _LAST_NAMES.append(line[0])
        # take top 10000 last names
        if len(_LAST_NAMES) >= TOP_LAST_NAMES:
            break

with open(_FIRST_NAMES_DATA_FILE, 'r') as data:
    # This skips the first row of the CSV file.
    reader = csv.reader(data, delimiter=';')
    next(reader)
    for line in reader:
        _FIRST_NAMES.append(line[0])
        # take top 1000 first names
        if len(_FIRST_NAMES) >= TOP_FIRST_NAMES:
            break

with open(_STREETS_DATA_FILE, 'r') as data:
    # This skips the first row of the CSV file.
    reader = csv.reader(data, delimiter=';')
    next(reader)
    for line in reader:
        _STREETS.append(line[0])
        # take top 1000 first names
        if len(_STREETS) >= TOP_STREETS:
            break

with open(_WORDS_DATA_FILE, 'r') as data:
    # This skips the first row of the CSV file.
    reader = csv.reader(data, delimiter='\t')
    next(reader)
    for line in reader:
        _WORDS.append(line[0])
        # take top 1000 first names
        if len(_WORDS) >= TOP_WORDS:
            break

def generate_full_names(amount=1):
    full_names = []
    for a in range(amount):
        random_first_name = random.choice(_FIRST_NAMES)
        random_last_name = random.choice(_LAST_NAMES)
        random_name = random_first_name + ' ' + random_last_name
        full_names.append(random_name)
    return full_names

def generate_streets(amount=1):
    streets = []
    for a in range(amount):
        random_street = random.choice(_STREETS)
        # add some random numbers to street
        random_street += ' ' + str(random.randint(1, 100))
        # add some random letter to street
        if random.randint(0, 1) == 1:
            random_street += " "
            random_street += random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])
            random_street += " "
            random_street += str(random.randint(1, 50))
        streets.append(random_street)
    return streets

def generate_words(amount=1):
    words = []
    for a in range(amount):
        words.append(random.choice(_WORDS))
    return words

def evaluate_anonymizer_with_generated_names(iterations=10000) -> (float, list[str]):
    print("\nRunning name anonymization test with {i} iterations using randomly generated names from top-{tl} "
          "last names and top-{tf} first names.".format(i=iterations, tl=TOP_LAST_NAMES, tf=TOP_FIRST_NAMES))
    recognizers = [RECOGNIZER_GLINER_FI]
    text_anonymizer = TextAnonymizer(languages=['fi'], recognizer_configuration=recognizers)
    # Measure time
    start_time = time.time()
    print("Using recognizer configuration: ", recognizers)
    print(text_anonymizer.analyzer_engine.registry.recognizers)

    success_count = 0
    partial_success_count = 0
    random_names = generate_full_names(iterations)
    failed_list = []
    for random_name in random_names:
        anonymized = text_anonymizer.anonymize_text(random_name)
        success_start = anonymized.startswith('<')
        success_end = anonymized.endswith('>')
        success = success_start and success_end
        partial_success = not success and (success_start or success_end)
        if success:
            success_count += 1
        if partial_success:
            partial_success_count += 1
        if not success and not partial_success:
            print("Not recognized name found: {name} -> {anonymized}, partial={partial_success}"
              .format(name=random_name, anonymized=anonymized, success=success, partial_success=partial_success))
            failed_list.append(random_name)

    success_percentage = round((success_count/iterations)*100, 2)
    partial_success_percentage = round((partial_success_count / iterations) * 100, 2)
    error_rate =  round((iterations - success_count - partial_success_count) / iterations * 100, 2)
    print(f"\nFinished. Iterations: {iterations},  error rate: {error_rate}%, anonymized: {success_percentage}%, "
          f"partially anonymized: {partial_success_percentage}%".format(
            error_rate=error_rate,
            success_percentage=success_percentage,
            partial_success_percentage=partial_success_percentage))
    print("\n")
    # Print time taken
    end_time = time.time()
    print("Time taken: {:.2f} seconds".format(end_time - start_time))
    return success_percentage, failed_list

def evaluate_anonymizer_with_plain_words(iterations=10000) -> (float, list[str]):
    print("\nRunning word anonymization test with {i} iterations using randomly selected plain words from dataset of {tl} words"
          .format(i=iterations, tl=TOP_WORDS))
    text_anonymizer = TextAnonymizer(languages=['fi'], recognizer_configuration=[RECOGNIZER_SPACY_FI])
    success_count = 0
    partial_success_count = 0
    random_words = generate_words(iterations)
    failed_list = []
    for random_word in random_words:
        anonymized = text_anonymizer.anonymize_text(random_word)
        success = "<" not in anonymized

        if success:
            success_count += 1
        if not success:
            failed_list.append(random_word)

    success_percentage = round((success_count/iterations)*100, 2)
    partial_success_percentage = round((partial_success_count / iterations) * 100, 2)
    error_rate =  round((iterations - success_count - partial_success_count) / iterations * 100, 2)
    print(f"\nFinished. Iterations: {iterations},  icorrectly anonymized: {error_rate}%, correctly not anonymized: {success_percentage}%, "
          .format(
            error_rate=error_rate,
            success_percentage=success_percentage,
            partial_success_percentage=partial_success_percentage))
    return success_percentage, failed_list

def evaluate_anonymizer_with_streets(iterations=1000) -> (float, list[str]):
    print("\nRunning street anonymization test with {i} iterations using list of {tl} street names."
          .format(i=iterations, tl=TOP_STREETS))
    text_anonymizer = TextAnonymizer(languages=['fi'], recognizer_configuration=[RECOGNIZER_SPACY_ADDRESS])
    success_count = 0
    partial_success_count = 0
    random_streets = generate_streets(iterations)

    failed_list = []
    for random_street in random_streets:
        anonymized = text_anonymizer.anonymize_text(random_street)
        # street name intact
        streetname = random_street.split(' ')[0]
        # rest of the address anonymized
        name_success = streetname in anonymized
        no_numbers_success = not any(char.isdigit() for char in anonymized)
        has_label = '<' in anonymized and '>' in anonymized

        success = name_success and no_numbers_success and has_label
        partial_success = no_numbers_success or has_label

        if success:
            success_count += 1
            # print(f"Success: {random_street} -> {anonymized}")
        elif partial_success:
            partial_success_count += 1
            print(f"Partial success: {random_street} -> {anonymized}")
        else:
            print("Not recognized street found: {street} -> {anonymized}, partial={partial_success}"
              .format(street=random_street, anonymized=anonymized, success=success, partial_success=partial_success))
            failed_list.append(random_street)

    success_percentage = round((success_count/iterations)*100, 2)
    partial_success_percentage = round((partial_success_count / iterations) * 100, 2)
    error_rate =  round((iterations - success_count - partial_success_count) / iterations * 100, 2)
    print(f"\nFinished. Iterations: {iterations},  error rate: {error_rate}%, anonymized: {success_percentage}%, "
          f"partially anonymized: {partial_success_percentage}%".format(
            error_rate=error_rate,
            success_percentage=success_percentage,
            partial_success_percentage=partial_success_percentage))
    return success_percentage, failed_list


def test_naturaltext_anonymizer(test_values=test_data.default_test_cases, verbose=False):
    text_anonymizer = TextAnonymizer(languages=['fi'])
    test_count = 0
    error_count = 0
    for s in test_values:
        test_count += 1
        # Check that test string is valid
        if '<' in s or '>' in s:
            print("Warning: test string contains < or >: ", s)

        test_text, i1, i2 = generate_test_text(s)
        anonymized_text = text_anonymizer.anonymize_text(test_text)
        j1, j2 = find_masked_str_pos(anonymized_text)
        sample_text = retrieve_sample(test_text, anonymized_text, s)
        mask_is_valid = True
        mask = None
        msg = ''

        # Try to locate label from anonymized text
        try:
            mask = anonymized_text[j1:j2]
        except ValueError:
            mask_is_valid = False

        # Check if there is masked label at all
        if mask is None or mask == '':
            msg = "Masking full word failed: {s}, mask: {m}".format(s=s, m=mask)
            mask_is_valid = False
        # If there is masked label, check anyway if test string is fully visible:
        elif s in anonymized_text:
            msg = "Test string still present in text: {s}, mask: {m}".format(s=s, m=mask)
            mask_is_valid = False
        # If not, scan through the anonymized text and verify is the censored word fully masked
        else:
            # Iterate test string letter by letter
            for a in range(1, len(s)):
                s1 = "{sub}<".format(sub=s[:a])
                s2 = ">{sub}".format(sub=s[a:])
                if s1 in anonymized_text:
                    msg = "Masking full word failed: {s}, mask: {m}, found: {s1}".format(s=s, m=mask, s1=s1)
                    mask_is_valid = False
                    break
                if s2 in anonymized_text:
                    msg = "Masking full word failed: {s}, mask: {m}, found: {s2}".format(s=s, m=mask, s2=s2)
                    mask_is_valid = False
                    break

        # Actual test: check if mask is valid
        if not mask_is_valid:
            error_count += 1
            print('\nError: ' + msg)
            print('Sample: '+sample_text)
            print("--")

        if verbose:
            print('Original: ' + test_text + '\n')
            print('Anonymized: ' + anonymized_text + '\n')

    print("{c}/{t} tests passed".format(c=(test_count-error_count), t=test_count))
    print("Result: {r}".format(r="FAILED" if error_count > 0 else "SUCCESS"))
    return error_count == 0


def generate_test_text(test_str: str):
    base_text = 'Hei, yhteystietoni on: '
    ret_text = base_text + test_str
    i1 = len(base_text)
    i2 = i1 + len(test_str)
    ret_text += '.'
    return ret_text, i1, i2


def retrieve_sample(test_text, anonymized_text, s):
    i1, i2 = find_original_str_pos(original_text=test_text, s=s)
    # add some padding
    i1 = (i1 - 10) if i1 >= 10 else 0
    i2 = (i2 + 10) if i2 < len(anonymized_text) - 10 else len(anonymized_text)
    sample = anonymized_text[i1: i2] + '...'
    return anonymized_text


def find_original_str_pos(original_text: str, s: str):
    try:
        i1 = original_text.index(s)
    except ValueError:
        i1 = -1
    i2 = i1 + len(s)
    return i1, i2


def find_masked_str_pos(anonymized_text: str):
    try:
        i1 = anonymized_text.index('<')
    except ValueError:
        i1 = -1
    try:
        i2 = anonymized_text.index('>') + 1
    except ValueError:
        i2 = -1
    return i1, i2


if __name__ == '__main__':
    test_naturaltext_anonymizer()

