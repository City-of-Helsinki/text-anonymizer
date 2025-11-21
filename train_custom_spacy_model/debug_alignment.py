"""
Debug script to identify misalignment issues in mixed-entity sentence generation
"""

import spacy
from spacy.training.iob_utils import offsets_to_biluo_tags
import random

# Load model
nlp = spacy.load('fi_core_news_lg')

# Test the exact case from the warning
name = "Lars Sjölund"
area = "vartiosaaressa"  # This is "vartiosaar" + "essa" suffix
street = "kontiaisenkujan"

# Pattern from train_custom_spacy_model.py
pattern = "{name} kertoi että {area} {street} puisto on hieno."

# Generate text
text = pattern.format(name=name, area=area, street=street)

print("="*80)
print("DEBUGGING MIXED ENTITY GENERATION")
print("="*80)

print(f"\n1. Pattern: {pattern}")
print(f"\n2. Values:")
print(f"   name   = '{name}'")
print(f"   area   = '{area}'")
print(f"   street = '{street}'")

print(f"\n3. Generated text:")
print(f"   '{text}'")

print(f"\n4. Finding entity positions with text.find():")

# Find entities the way the code does it
name_start = text.find(name)
name_end = name_start + len(name)
print(f"   name at [{name_start}, {name_end}]: '{text[name_start:name_end]}'")

area_start = text.find(area)
area_end = area_start + len(area)
print(f"   area at [{area_start}, {area_end}]: '{text[area_start:area_end]}'")

street_start = text.find(street)
street_end = street_start + len(street)
print(f"   street at [{street_start}, {street_end}]: '{text[street_start:street_end]}'")

# Build entities list
entities = [
    [name_start, name_end, 'PERSON'],
    [area_start, area_end, 'GPE'],
    [street_start, street_end, 'LOC']
]

print(f"\n5. Entities list: {entities}")

print(f"\n6. SpaCy tokenization:")
doc = nlp.make_doc(text)
for i, token in enumerate(doc):
    print(f"   Token {i}: '{token.text}' at [{token.idx}, {token.idx + len(token.text)}]")

print(f"\n7. BILUO tags check:")
try:
    tags = offsets_to_biluo_tags(doc, entities)
    print(f"   Tags: {tags}")

    if '-' in tags:
        print(f"\n   MISALIGNMENT DETECTED!")
        print(f"   Tokens with '-' tag are misaligned:")
        for i, (token, tag) in enumerate(zip(doc, tags)):
            if tag == '-':
                print(f"      Token {i}: '{token.text}' at [{token.idx}, {token.idx + len(token.text)}] -> tag: {tag}")
    else:
        print(f"   All entities aligned correctly!")

except Exception as e:
    print(f"   ERROR: {e}")

# Now test with the actual problem case from the warning
print("\n" + "="*80)
print("TESTING ACTUAL WARNING CASE")
print("="*80)

text_warning = "Gustav Sjölund kertoi että vartiosaaressa kontiaisenkujan puisto on mahtava leikkipaikka."
entities_warning = [[0, 12, 'PERSON'], [25, 36, 'GPE'], [40, 54, 'LOC']]

print(f"\nText: '{text_warning}'")
print(f"Entities: {entities_warning}")

print(f"\nSpaCy tokenization:")
doc_warning = nlp.make_doc(text_warning)
for i, token in enumerate(doc_warning):
    print(f"   Token {i}: '{token.text}' at [{token.idx}, {token.idx + len(token.text)}]")

print(f"\nBILUO tags:")
tags_warning = offsets_to_biluo_tags(doc_warning, entities_warning)
print(f"   Tags: {tags_warning}")

if '-' in tags_warning:
    print(f"\n   MISALIGNMENT:")
    for i, (token, tag) in enumerate(zip(doc_warning, tags_warning)):
        print(f"      Token {i}: '{token.text}' [{token.idx}:{token.idx + len(token.text)}] -> {tag}")
        if tag == '-':
            print(f"         ^^^ MISALIGNED!")

# Check character encoding
print("\n" + "="*80)
print("CHARACTER ENCODING CHECK")
print("="*80)

test_str = "Lars Sjölund kertoi että lännessä on itä."
print(f"\nString: '{test_str}'")
print(f"Length: {len(test_str)}")
print(f"Character breakdown:")
for i, char in enumerate(test_str):
    print(f"   [{i}] '{char}' (Unicode: {ord(char)}, bytes: {char.encode('utf-8')})")

print(f"\nSearching for 'että' in text:")
pos = test_str.find('että')
print(f"   Position: {pos}")
print(f"   Substring: '{test_str[pos:pos+4]}'")

