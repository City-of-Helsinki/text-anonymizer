"""
Evaluation Pipeline Module for SpaCy NER Model

This module handles:
- Comprehensive model evaluation with test data
- Per-entity-type performance metrics
- Custom test scenarios (names, streets, areas)
- External test set evaluation
"""

import random
from typing import Dict, List, Tuple

import spacy
from spacy.training import Example

from model_version import FINETUNED_MODEL_VERSION


class ModelEvaluator:
    """Comprehensive model evaluation"""

    def __init__(self, nlp: spacy.Language = None, model_path: str = None):
        """
        Initialize evaluator with either a loaded model or path to model

        Args:
            nlp: Loaded SpaCy model (optional)
            model_path: Path to model to load (optional)
        """
        if nlp is None and model_path is None:
            model_path = f"../custom_spacy_model/{FINETUNED_MODEL_VERSION}"

        if nlp is None:
            print(f"Loading model from {model_path}...")
            self.nlp = spacy.load(model_path)
        else:
            self.nlp = nlp

        # Entity types
        self.LOC = 'LOC'
        self.AREA_ENTITY = 'GPE'
        self.NAME_ENTITY = 'PERSON'

    def evaluate_with_test_data(self, eval_data: List[Example], verbose: bool = True) -> Dict:
        """
        Evaluate model with prepared test data

        Args:
            eval_data: List of evaluation examples
            verbose: Whether to print detailed results

        Returns:
            Dictionary with evaluation scores
        """
        scores = self.nlp.evaluate(eval_data)

        if verbose:
            print("\n" + "="*80)
            print(" MODEL EVALUATION RESULTS")
            print("="*80)

            if "ents_p" in scores:
                precision = scores["ents_p"]
                recall = scores["ents_r"]
                f1_score = scores["ents_f"]

                print(f"\n Overall Performance:")
                print(f"  • Precision: {precision*100:6.2f}%  (correctness of detected entities)")
                print(f"  • Recall:    {recall*100:6.2f}%  (coverage of actual entities)")
                print(f"  • F1 Score:  {f1_score*100:6.2f}%  (harmonic mean)")

            if "ents_per_type" in scores:
                per_type = scores["ents_per_type"]
                print(f"\n Performance by Entity Type:")
                print(f"  {'Type':<12} | {'Precision':>10} | {'Recall':>10} | {'F1 Score':>10}")
                print("  " + "-"*52)
                for entity_type, metrics in per_type.items():
                    print(f"  {entity_type:<12} | {metrics['p']*100:9.2f}% | {metrics['r']*100:9.2f}% | {metrics['f']*100:9.2f}%")

            print("="*80 + "\n")

        return scores

    def test_name_recognition(self, names: List[str], test_sentences: List[str], amount: int = 50) -> float:
        """
        Test model's ability to recognize person names in context

        Args:
            names: List of names to test
            test_sentences: List of sentence templates
            amount: Number of tests to run

        Returns:
            Percentage of tests passed
        """
        from data_preparation import SentenceGenerator

        results = []
        for _ in range(amount):
            # Generate full names
            name_list = random.sample(names, min(2, len(names)))

            # Create test text with names
            test_text = "Tämä on keksitty lause jossa testataan nimien tunnistusta. "
            for idx, name in enumerate(name_list):
                sentence, _, _ = SentenceGenerator.generate_sentence(name, test_sentences)
                test_text += sentence + " "

            doc = self.nlp(test_text)

            # Count correct detections
            correct = 0
            for ent in doc.ents:
                if ent.label_ == self.NAME_ENTITY:
                    entity_str = str(ent).replace('\\.', '')
                    if entity_str in name_list:
                        correct += 1
                    elif any(part in name_list for part in entity_str.split()):
                        correct += 0.5

            # Consider test passed if we got at least as many as we put in
            results.append(correct >= len(name_list))

        pass_rate = results.count(True) / len(results) * 100
        return pass_rate

    def test_area_recognition(self, areas: List[str], test_sentences: List[str], amount: int = 50) -> float:
        """
        Test model's ability to recognize area names

        Args:
            areas: List of area names to test
            test_sentences: List of sentence templates
            amount: Number of tests to run

        Returns:
            Percentage of tests passed
        """
        from data_preparation import SentenceGenerator

        results = []
        for _ in range(amount):
            area_list = random.sample(areas, min(2, len(areas)))

            test_text = "Tämä on testi alueiden tunnistukselle. "
            for area in area_list:
                sentence, _, _ = SentenceGenerator.generate_sentence(area, test_sentences)
                test_text += sentence + " "

            doc = self.nlp(test_text)

            correct = 0
            for ent in doc.ents:
                if ent.label_ in [self.AREA_ENTITY, self.LOC]:
                    if str(ent) in area_list:
                        correct += 1

            results.append(correct >= len(area_list))

        pass_rate = results.count(True) / len(results) * 100
        return pass_rate

    def test_street_recognition(self, streets: List[str], test_sentences: List[str], amount: int = 50) -> float:
        """
        Test model's ability to recognize street names

        Args:
            streets: List of street names to test
            test_sentences: List of sentence templates
            amount: Number of tests to run

        Returns:
            Percentage of tests passed
        """
        from data_preparation import SentenceGenerator

        results = []
        for _ in range(amount):
            street_list = random.sample(streets, min(2, len(streets)))

            test_text = "Tämä on testi katujen tunnistukselle. "
            for street in street_list:
                sentence, _, _ = SentenceGenerator.generate_sentence(street, test_sentences)
                test_text += sentence + " "

            doc = self.nlp(test_text)

            correct = 0
            for ent in doc.ents:
                if ent.label_ in [self.LOC, self.AREA_ENTITY, 'STREET']:
                    if str(ent) in street_list:
                        correct += 1

            results.append(correct >= len(street_list))

        pass_rate = results.count(True) / len(results) * 100
        return pass_rate

    def run_comprehensive_tests(
        self,
        names: List[str] = None,
        streets: List[str] = None,
        areas: List[str] = None,
        amount: int = 50
    ) -> Dict[str, float]:
        """
        Run comprehensive tests on all entity types

        Args:
            names: List of names for testing
            streets: List of streets for testing
            areas: List of areas for testing
            amount: Number of tests per entity type

        Returns:
            Dictionary with test results
        """
        print("\n" + "="*80)
        print("RUNNING COMPREHENSIVE TESTS")
        print("="*80)

        results = {}

        if names:
            from training_data import SENTENCES_NAME
            name_score = self.test_name_recognition(names, SENTENCES_NAME, amount)
            results['names'] = name_score
            print(f"  Names Recognition:   {name_score:6.1f}% pass rate")

        if streets:
            from training_data import SENTENCES_STREETS
            street_score = self.test_street_recognition(streets, SENTENCES_STREETS, amount)
            results['streets'] = street_score
            print(f"  Streets Recognition: {street_score:6.1f}% pass rate")

        if areas:
            from training_data import SENTENCES_AREAS
            area_score = self.test_area_recognition(areas, SENTENCES_AREAS, amount)
            results['areas'] = area_score
            print(f"  Areas Recognition:   {area_score:6.1f}% pass rate")

        # Overall score
        if results:
            overall = sum(results.values()) / len(results)
            results['overall'] = overall

            print(f"\n  Overall Test Score:  {overall:6.1f}%")

        print("="*80 + "\n")

        return results


def evaluate_model(
    nlp: spacy.Language = None,
    model_path: str = None,
    eval_data: List[Example] = None,
    run_custom_tests: bool = True,
    test_data: Dict[str, List[str]] = None
) -> Tuple[Dict, Dict]:
    """
    Complete evaluation pipeline

    Args:
        nlp: Loaded SpaCy model (optional)
        model_path: Path to model (optional)
        eval_data: Evaluation examples (optional)
        run_custom_tests: Whether to run custom test scenarios
        test_data: Dictionary with test data (names, streets, areas)

    Returns:
        Tuple of (evaluation_scores, test_results)
    """
    print("\n" + "="*80)
    print(" MODEL EVALUATION PIPELINE")
    print("="*80)

    evaluator = ModelEvaluator(nlp, model_path)

    # Evaluate with test data if provided
    eval_scores = {}
    if eval_data:
        print("\n Evaluating on prepared test set...")
        eval_scores = evaluator.evaluate_with_test_data(eval_data, verbose=True)

    # Run custom tests if requested
    test_results = {}
    if run_custom_tests and test_data:
        test_results = evaluator.run_comprehensive_tests(
            names=test_data.get('names'),
            streets=test_data.get('streets'),
            areas=test_data.get('areas'),
            amount=test_data.get('amount', 50)
        )

    return eval_scores, test_results


def prepare_evaluation_data(nlp: spacy.Language) -> List[Example]:
    """
    Prepare evaluation data from predefined test sentences

    Args:
        nlp: SpaCy model for creating examples

    Returns:
        List of evaluation examples
    """
    from training_data import (
        EVALUATION_SENTENCES,
        EVALUATION_VALUES,
        EVALUATION_LABELS
    )
    from data_preparation import SentenceGenerator

    eval_data = []

    for i in range(len(EVALUATION_SENTENCES)):
        # Select sentence, value, and label from same position, eg. "{s} on katu Helsingissä." "Liisankatu" "LOC"
        sentence = EVALUATION_SENTENCES[i]
        value = EVALUATION_VALUES[i]
        label = EVALUATION_LABELS[i]

        if value:
            sentence, start, end = SentenceGenerator.generate_evaluation_sentence(value.lower(), sentence)
            doc = nlp(sentence)
            entities = [[start, end, label]]
            example = Example.from_dict(doc, {"entities": entities})
        else:
            example = Example.from_dict(nlp(sentence), {"entities": []})

        eval_data.append(example)

    return eval_data


if __name__ == "__main__":
    # Example usage
    from data_preparation import DataLoader

    # Load test data
    data_loader = DataLoader()
    data_loader.load_all()

    # Prepare test data
    test_data = {
        'names': data_loader.first_names[:100] + data_loader.last_names[:100],
        'streets': data_loader.streets[:100],
        'areas': data_loader.areas[:50],
        'amount': 50
    }

    # Evaluate model
    eval_scores, test_results = evaluate_model(
        model_path=f"../custom_spacy_model/{FINETUNED_MODEL_VERSION}",
        run_custom_tests=True,
        test_data=test_data
    )

    print("\n Evaluation complete!")
    if test_results.get('overall'):
        print(f"Overall Test Score: {test_results['overall']:.2f}%")

