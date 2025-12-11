#!/usr/bin/env python
"""
Test script for custom regex pattern loading with profiles.
Verifies that:
1. Profile-specific regex patterns are loaded correctly
2. Regex recognizers are added to profile registries
3. Pattern matching works as expected
4. NLP engine is reused (memory efficient)
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

import logging

# Configure logging to see debug messages
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

from text_anonymizer import TextAnonymizer


def test_profile_regex_patterns():
    """Test that profile regex patterns are loaded and work correctly."""
    print("\n" + "="*70)
    print("TEST: Profile Regex Pattern Loading and Detection")
    print("="*70)

    # Initialize anonymizer
    anonymizer = TextAnonymizer(debug_mode=False)

    # Test cases
    test_cases = [
        ("ABCxyz123", "Simple pattern without context"),
        ("Please process ABCxyz123 immediately.", "Pattern in context"),
        ("ABCxyz123 department", "Pattern with trailing word"),
        ("XYZabc321", "Alternative pattern"),
        ("XYZabc321 unit", "Alternative pattern with trailing word"),
    ]

    results = []
    for text, description in test_cases:
        print(f"\nTest: {description}")
        print(f"Input: '{text}'")

        try:
            result = anonymizer.anonymize(text, profile="example")

            print(f"Anonymized: '{result.anonymized_text}'")
            print(f"Details: {result.details}")

            # Verify EXAMPLE entity was detected
            if "EXAMPLE" in result.details:
                entities = result.details["EXAMPLE"]
                print(f"EXAMPLE Entities Detected: {entities}")
                results.append((text, True, entities))
            else:
                print("EXAMPLE: No entities detected")
                results.append((text, False, []))

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append((text, False, str(e)))

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    passed = sum(1 for _, detected, _ in results if detected)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    for text, detected, entities in results:
        status = "PASS" if detected else "FAIL"
        print(f"  [{status}] '{text}' -> {entities}")

    return passed == total


def test_nlp_engine_reuse():
    """Verify that NLP engine is reused across profile analyzers."""
    print("\n" + "="*70)
    print("TEST: NLP Engine Reuse (Memory Efficiency)")
    print("="*70)

    anonymizer = TextAnonymizer(debug_mode=False)

    # Get default analyzer
    default_analyzer = anonymizer.analyzer_engine
    default_nlp = anonymizer.nlp_engine

    print(f"\nDefault NLP Engine ID: {id(default_nlp)}")
    print(f"Default Analyzer ID: {id(default_analyzer)}")

    # Get profile-specific analyzer
    profile_analyzer = anonymizer._get_analyzer_for_profile("example")
    profile_nlp = profile_analyzer.nlp_engine

    print(f"\nProfile NLP Engine ID: {id(profile_nlp)}")
    print(f"Profile Analyzer ID: {id(profile_analyzer)}")

    # Verify NLP engine is the same object
    if default_nlp is profile_nlp:
        print("\n✓ SUCCESS: NLP engine is reused (same object)")
        print("  Memory efficient: Only one SpaCy model loaded in memory")
        return True
    else:
        print("\n✗ FAILURE: NLP engine instances are different")
        print("  Memory inefficient: Multiple SpaCy models loaded")
        return False


def test_profile_caching():
    """Verify that profile analyzers are cached."""
    print("\n" + "="*70)
    print("TEST: Profile Analyzer Caching")
    print("="*70)

    anonymizer = TextAnonymizer(debug_mode=False)

    # First call to profile
    analyzer1 = anonymizer._get_analyzer_for_profile("example")
    print(f"\nFirst call - Analyzer ID: {id(analyzer1)}")

    # Second call to same profile
    analyzer2 = anonymizer._get_analyzer_for_profile("example")
    print(f"Second call - Analyzer ID: {id(analyzer2)}")

    # Verify they're the same object
    caching_ok = analyzer1 is analyzer2
    if caching_ok:
        print("\n✓ SUCCESS: Profile analyzers are cached (same object)")
    else:
        print("\n✗ FAILURE: Profile analyzers are not cached (different objects)")
        return False

    # Verify there's a regex recognizer in the profile registry with EXAMPLE entity_type
    print("\n" + "-"*70)
    print("Verifying EXAMPLE regex recognizer in profile registry...")
    print("-"*70)

    profile_registry = analyzer1.registry
    print(f"\nRegistry type: {type(profile_registry)}")
    print(f"Registry supported languages: {profile_registry.supported_languages}")

    # Get all recognizers from registry
    all_recognizers = profile_registry.recognizers
    print(f"\nTotal recognizers in registry: {len(all_recognizers)}")

    # Look for EXAMPLE entity type recognizers
    # Handle both 'entity_type' and 'entity' attributes (different recognizer types use different names)
    example_recognizers = []
    for r in all_recognizers:
        entity_attr = getattr(r, 'entity_type', None) or getattr(r, 'entity', None)
        if entity_attr == "EXAMPLE":
            example_recognizers.append(r)

    print(f"Recognizers with EXAMPLE entity_type: {len(example_recognizers)}")

    if example_recognizers:
        for idx, recognizer in enumerate(example_recognizers, 1):
            print(f"\n  [{idx}] Recognizer Details:")
            print(f"      Name: {recognizer.name}")
            entity_attr = getattr(recognizer, 'entity_type', None) or getattr(recognizer, 'entity', None)
            print(f"      Entity Type: {entity_attr}")
            print(f"      Type: {type(recognizer).__name__}")
            if hasattr(recognizer, 'patterns'):
                print(f"      Patterns: {recognizer.patterns}")
            supported_lang = getattr(recognizer, 'supported_language', getattr(recognizer, 'languages', 'N/A'))
            print(f"      Supported Language: {supported_lang}")
            confidence = getattr(recognizer, 'confidence_score', 'N/A')
            print(f"      Score: {confidence}")
    else:
        print("\n✗ WARNING: No EXAMPLE recognizers found in registry")
        print("\nAll recognizers in registry by entity type:")
        entity_types = {}
        for recognizer in all_recognizers:
            entity_attr = getattr(recognizer, 'entity_type', None) or getattr(recognizer, 'entity', None)
            if entity_attr:
                if entity_attr not in entity_types:
                    entity_types[entity_attr] = []
                entity_types[entity_attr].append(recognizer.name)

        for entity_type in sorted(entity_types.keys()):
            print(f"  {entity_type}: {entity_types[entity_type]}")

    return caching_ok



if __name__ == "__main__":
    print("\n")
    print("*" * 70)
    print("CUSTOM REGEX PROFILE INTEGRATION TESTS")
    print("*" * 70)

    test_results = []

    # Run tests
    test_results.append(("Profile Regex Patterns", test_profile_regex_patterns()))
    test_results.append(("NLP Engine Reuse", test_nlp_engine_reuse()))
    test_results.append(("Profile Caching", test_profile_caching()))

    # Final summary
    print("\n" + "*" * 70)
    print("FINAL RESULTS")
    print("*" * 70)
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    print(f"\nTotal: {passed}/{total} test suites passed\n")

    for test_name, result in test_results:
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"  {symbol} {test_name}: {status}")

    sys.exit(0 if passed == total else 1)

