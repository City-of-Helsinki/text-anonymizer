"""
Helper script for visual inspection of training and evaluation data.
Exports training and evaluation data to text files for review.

This script:
1. Prepares training data
2. Prepares evaluation data
3. Writes each example's text to separate files (one per line)
"""

from data_preparation import prepare_training_data
from evaluation_pipeline import prepare_evaluation_data


def export_examples_to_file(examples, filename):
    """
    Export training/eval examples to a text file

    Args:
        examples: List of SpaCy Example objects
        filename: Output filename
    """
    print(f"Writing {len(examples)} examples to {filename}...")

    with open(filename, 'w', encoding='utf-8') as f:
        for i, example in enumerate(examples, 1):
            # Get the text from the example
            text = example.reference.text
            # Write one example per line
            f.write(f"{text}\n")

    print(f"✓ Wrote {len(examples)} examples to {filename}")


def main():
    print("="*80)
    print("EXPORTING TRAINING AND EVALUATION DATA FOR REVIEW")
    print("="*80 + "\n")

    # Step 1: Prepare training data
    print("Step 1: Preparing training data...")
    train_data, nlp = prepare_training_data(base_model="fi_core_news_lg")
    print(f"✓ Prepared {len(train_data)} training examples\n")

    # Step 2: Prepare evaluation data (with 10% of training data)
    print("Step 2: Preparing evaluation data...")
    eval_data = prepare_evaluation_data(nlp)
    print(f"✓ Prepared {len(eval_data)} evaluation examples\n")

    # Step 3: Export to files
    print("Step 3: Exporting to text files...")
    export_examples_to_file(train_data, "training_data.txt")
    export_examples_to_file(eval_data, "evaluation_data.txt")

    print("\n" + "="*80)
    print("EXPORT COMPLETE!")
    print("="*80)
    print(f"\nFiles created:")
    print(f"  • training_data.txt ({len(train_data)} examples)")
    print(f"  • evaluation_data.txt ({len(eval_data)} examples)")
    print(f"\nYou can now review these files to check if sentences are sane.")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

