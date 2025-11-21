import unittest
import random
from datetime import datetime

import tabulate

import test_util_text_anonymizer
from text_anonymizer import TextAnonymizer
from text_anonymizer.constants import RECOGNIZER_SPACY_FI, RECOGNIZER_SPACY_ADDRESS, RECOGNIZER_SPACY_EN


class TestStreetPersonConfusion(unittest.TestCase):
    """
    Test to detect when streets are misclassified as persons and vice versa.

    This test addresses user feedback that anonymization sometimes mixes streets and persons.
    Some streets are named after persons, so some confusion is expected, but we want to
    identify which specific streets and names cause misclassification.
    """

    def test_street_misclassified_as_person(self):
        """
        Test if streets are being incorrectly labeled as PERSON.

        We generate street addresses and check if they get labeled as PERSON
        instead of ADDRESS when both recognizers are active.
        """
        print("\n" + "="*80)
        print("Testing: Streets misclassified as PERSON")
        print("="*80)

        random.seed(1234)
        iterations = 1000

        # Initialize anonymizer with both recognizers
        text_anonymizer = TextAnonymizer(
            languages=['fi'],
            recognizer_configuration=[RECOGNIZER_SPACY_FI, RECOGNIZER_SPACY_ADDRESS],
            debug_mode=True  # Enable debug mode to see scores
        )

        streets = test_util_text_anonymizer.generate_streets(iterations)

        misclassified_as_person = []
        correctly_classified = []
        not_recognized = []

        for street in streets:
            result = text_anonymizer.anonymize(street)
            anonymized = result.anonymized_text
            details = result.details

            # Check what entity types were detected
            has_person = any('NIMI' in key or 'PERSON' in key for key in details.keys())
            has_address = any('OSOITE' in key or 'ADDRESS' in key for key in details.keys())

            if has_person and not has_address:
                misclassified_as_person.append({
                    'original': street,
                    'anonymized': anonymized,
                    'details': details
                })
            elif has_address:
                correctly_classified.append(street)
            else:
                not_recognized.append(street)

        # Calculate statistics
        total = len(streets)
        misclassified_count = len(misclassified_as_person)
        correct_count = len(correctly_classified)
        not_recognized_count = len(not_recognized)

        misclassification_rate = (misclassified_count / total) * 100
        correct_rate = (correct_count / total) * 100
        not_recognized_rate = (not_recognized_count / total) * 100

        print(f"\nResults for {iterations} street samples:")
        print(f"  - Correctly classified as ADDRESS: {correct_count} ({correct_rate:.2f}%)")
        print(f"  - Misclassified as PERSON: {misclassified_count} ({misclassification_rate:.2f}%)")
        print(f"  - Not recognized: {not_recognized_count} ({not_recognized_rate:.2f}%)")

        if misclassified_as_person:
            print(f"\nStreets misclassified as PERSON (showing first 20):")
            for item in misclassified_as_person[:20]:
                print(f"  {item['original']:40} -> {item['anonymized']:40} | {item['details']}")

        # Store results for reporting
        self.street_misclassification_results = {
            'total': total,
            'misclassified': misclassified_as_person,
            'correct': correct_count,
            'not_recognized': not_recognized_count,
            'misclassification_rate': misclassification_rate
        }

        return misclassification_rate, misclassified_as_person

    def test_person_misclassified_as_street(self):
        """
        Test if person names are being incorrectly labeled as ADDRESS.

        We generate person names and check if they get labeled as ADDRESS
        instead of PERSON when both recognizers are active.
        """
        print("\n" + "="*80)
        print("Testing: Person names misclassified as ADDRESS")
        print("="*80)

        random.seed(1234)
        iterations = 1000

        # Initialize anonymizer with both recognizers
        text_anonymizer = TextAnonymizer(
            languages=['fi'],
            recognizer_configuration=[RECOGNIZER_SPACY_FI, RECOGNIZER_SPACY_EN, RECOGNIZER_SPACY_ADDRESS],
            debug_mode=True
        )

        names = test_util_text_anonymizer.generate_full_names(iterations)

        misclassified_as_address = []
        correctly_classified = []
        not_recognized = []

        for name in names:
            result = text_anonymizer.anonymize(name)
            anonymized = result.anonymized_text
            details = result.details

            # Check what entity types were detected
            has_person = any('NIMI' in key or 'PERSON' in key for key in details.keys())
            has_address = any('OSOITE' in key or 'ADDRESS' in key for key in details.keys())

            if has_address and not has_person:
                misclassified_as_address.append({
                    'original': name,
                    'anonymized': anonymized,
                    'details': details
                })
            elif has_person:
                correctly_classified.append(name)
            else:
                not_recognized.append(name)

        # Calculate statistics
        total = len(names)
        misclassified_count = len(misclassified_as_address)
        correct_count = len(correctly_classified)
        not_recognized_count = len(not_recognized)

        misclassification_rate = (misclassified_count / total) * 100
        correct_rate = (correct_count / total) * 100
        not_recognized_rate = (not_recognized_count / total) * 100

        print(f"\nResults for {iterations} name samples:")
        print(f"  - Correctly classified as PERSON: {correct_count} ({correct_rate:.2f}%)")
        print(f"  - Misclassified as ADDRESS: {misclassified_count} ({misclassification_rate:.2f}%)")
        print(f"  - Not recognized: {not_recognized_count} ({not_recognized_rate:.2f}%)")

        if misclassified_as_address:
            print(f"\nNames misclassified as ADDRESS (showing first 20):")
            for item in misclassified_as_address[:20]:
                print(f"  {item['original']:40} -> {item['anonymized']:40} | {item['details']}")

        # Store results for reporting
        self.person_misclassification_results = {
            'total': total,
            'misclassified': misclassified_as_address,
            'correct': correct_count,
            'not_recognized': not_recognized_count,
            'misclassification_rate': misclassification_rate
        }

        return misclassification_rate, misclassified_as_address

    def test_mixed_context(self):
        """
        Test both streets and names in mixed contexts to see how they're classified.

        This simulates real-world scenarios where both types of entities appear together.
        """
        print("\n" + "="*80)
        print("Testing: Mixed context with both streets and names")
        print("="*80)

        random.seed(1234)
        iterations = 500

        # Initialize anonymizer with both recognizers
        text_anonymizer = TextAnonymizer(
            languages=['fi'],
            recognizer_configuration=[RECOGNIZER_SPACY_FI, RECOGNIZER_SPACY_EN, RECOGNIZER_SPACY_ADDRESS],
            debug_mode=True
        )

        names = test_util_text_anonymizer.generate_full_names(iterations)
        streets = test_util_text_anonymizer.generate_streets(iterations)

        mixed_texts = []
        for i in range(iterations):
            # Create texts with both name and street
            text = f"{names[i]} asui 1890-luvulla osoitteessa {streets[i]}."
            mixed_texts.append({
                'text': text,
                'name': names[i],
                'street': streets[i]
            })

        issues = []

        for item in mixed_texts:
            result = text_anonymizer.anonymize(item['text'])
            anonymized = result.anonymized_text
            details = result.details

            # Check if both entities were properly detected
            has_person = any('NIMI' in key or 'PERSON' in key for key in details.keys())
            has_address = any('OSOITE' in key or 'ADDRESS' in key for key in details.keys())

            if not has_person or not has_address:
                issues.append({
                    'original': item['text'],
                    'anonymized': anonymized,
                    'details': details,
                    'has_person': has_person,
                    'has_address': has_address,
                    'name': item['name'],
                    'street': item['street']
                })
            else:
                # Print success in debug mode
                print(f"Success: Both entities detected in: '{anonymized}'")

        print(f"\nResults for {iterations} mixed context samples:")
        print(f"  - Both entities detected: {iterations - len(issues)} ({((iterations - len(issues))/iterations)*100:.2f}%)")
        print(f"  - Issues found: {len(issues)} ({(len(issues)/iterations)*100:.2f}%)")

        if issues:
            print(f"\nMixed context issues (showing first 20):")
            for item in issues[:20]:
                print(f"\n  Original: {item['original']}")
                print(f"  Anonymized: {item['anonymized']}")
                print(f"  Has PERSON: {item['has_person']}, Has ADDRESS: {item['has_address']}")
                print(f"  Details: {item['details']}")

        return issues

    def test_evaluation_summary(self):
        """
        Run all confusion tests and generate a summary report.
        """
        print("\n" + "="*80)
        print("STREET ADDRESS vs PERSON CONFUSION EVALUATION")
        print(f"Date: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
        print("="*80)

        # Run all tests
        street_misc_rate, street_misc_list = self.test_street_misclassified_as_person()
        person_misc_rate, person_misc_list = self.test_person_misclassified_as_street()
        mixed_issues = self.test_mixed_context()

        # Generate summary table
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)

        results = [
            {
                "Test Type": "Streets → PERSON",
                "Samples": self.street_misclassification_results['total'],
                "Misclassified": len(street_misc_list),
                "Rate": f"{street_misc_rate:.2f}%"
            },
            {
                "Test Type": "Names → ADDRESS",
                "Samples": self.person_misclassification_results['total'],
                "Misclassified": len(person_misc_list),
                "Rate": f"{person_misc_rate:.2f}%"
            },
            {
                "Test Type": "Mixed Context Issues",
                "Samples": 500,
                "Misclassified": len(mixed_issues),
                "Rate": f"{(len(mixed_issues)/500)*100:.2f}%"
            }
        ]

        tab_results = tabulate.tabulate(results, headers="keys", tablefmt="pipe")
        print("\n" + tab_results)

        # Analyze problematic streets (those named after persons)
        print("\n" + "="*80)
        print("ANALYSIS: Problematic Streets (likely named after persons)")
        print("="*80)

        problematic_streets = []
        for item in street_misc_list[:50]:
            street_name = item['original'].split()[0]  # Get just the street name
            problematic_streets.append(street_name)

        if problematic_streets:
            print("\nTop streets misclassified as persons:")
            for street in problematic_streets[:30]:
                print(f"  - {street}")

if __name__ == '__main__':
    unittest.main()

