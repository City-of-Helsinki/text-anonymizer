"""
Main orchestrator script for SpaCy NER fine-tuning

This script coordinates:
1. ETL and data preparation
2. Model training
3. Model evaluation

Usage:
    python main_train.py [--base-model MODEL_NAME] [--iterations N] [--no-save] [--disable-gpu]

Environment Variables:
    DISABLE_GPU=1    Disable GPU/Metal acceleration (alternative to --disable-gpu flag)
"""

import argparse
import datetime
import os
import sys

from data_preparation import prepare_training_data, DataLoader
from training_pipeline import train_model_pipeline, TRAINING_CONFIG
from evaluation_pipeline import evaluate_model, prepare_evaluation_data
from model_version import FINETUNED_MODEL_VERSION


def main(
    base_model: str = "fi_core_news_lg",
    iterations: int = None,
    save_model: bool = True,
    use_entity_ruler: bool = True,
    seed: int = None
):
    """
    Main training pipeline orchestrator

    Args:
        base_model: Base SpaCy model name
        iterations: Number of training iterations (None = use config default)
        save_model: Whether to save the trained model
        use_entity_ruler: Whether to use entity ruler
        seed: Random seed for reproducibility (None = use default from data_preparation)
    """
    timestamp = datetime.datetime.now().strftime('%Y.%m.%d %H:%M')

    print("\n" + "="*80)
    print("SPACY NER MODEL FINE TUNING - FINNISH TEXT ANONYMIZER")
    print("="*80)
    print(f"Base Model: {base_model}")
    print(f"Target Model Version: {FINETUNED_MODEL_VERSION}")
    print(f"Timestamp: {timestamp}")
    print("="*80 + "\n")

    # ========================================================================
    # STEP 1: ETL & DATA PREPARATION
    # ========================================================================
    print("STEP 1/3: ETL & DATA PREPARATION")
    print("-" * 80)

    train_data, nlp = prepare_training_data(base_model=base_model, seed=seed)

    print(f" Data preparation complete: {len(train_data)} training examples")

    # Prepare evaluation data
    eval_data = prepare_evaluation_data(nlp)
    print(f" Evaluation data prepared: {len(eval_data)} test examples\n")

    # ========================================================================
    # STEP 2: MODEL TRAINING
    # ========================================================================
    print("STEP 2/3: MODEL TRAINING")
    print("-" * 80)

    # Update config if custom iterations provided
    config = TRAINING_CONFIG.copy()
    if iterations:
        config['iterations'] = iterations
        print(f"Using custom iterations: {iterations}\n")

    # Prepare entity ruler patterns if needed
    patterns_data = None
    if use_entity_ruler:
        data_loader = DataLoader()
        data_loader.load_all()

        # NOTE: Do NOT add PERSON patterns - individual first/last names break
        # NER's ability to detect full names like "Matti Korhonen"
        # The NER component handles person names better through context
        patterns_data = {
            'PRODUCT': data_loader.products,
            'LOC': data_loader.streets,
            'GPE': data_loader.areas,
            'ORG': data_loader.organizations,
        }

    # Train the model
    save_path = f"../custom_spacy_model/{FINETUNED_MODEL_VERSION}" if save_model else None
    best_f1, metrics = train_model_pipeline(
        train_data=train_data,
        eval_data=eval_data,
        nlp=nlp,
        use_entity_ruler=use_entity_ruler,
        patterns_data=patterns_data,
        save_model_path=save_path,
        config=config
    )

    print(f" Training complete: Best F1 = {best_f1*100:.2f}%\n")

    # ========================================================================
    # STEP 3: MODEL EVALUATION
    # ========================================================================
    print("STEP 3/3: MODEL EVALUATION")
    print("-" * 80)

    # Prepare test data for custom tests
    data_loader = DataLoader()
    data_loader.load_all()

    from data_preparation import SentenceGenerator
    test_names = SentenceGenerator.generate_full_names(
        data_loader.first_names,
        data_loader.last_names,
        amount=100
    )

    test_data = {
        'names': test_names,
        'streets': data_loader.streets[:100],
        'areas': data_loader.areas[:50],
        'amount': 100
    }

    # Run comprehensive evaluation
    eval_scores, test_results = evaluate_model(
        nlp=nlp,
        eval_data=eval_data,
        run_custom_tests=True,
        test_data=test_data
    )

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print(" TRAINING SESSION SUMMARY")
    print("="*80)
    print(f"\n Configuration:")
    print(f"  • Base Model: {base_model}")
    print(f"  • Training Examples: {len(train_data)}")
    print(f"  • Iterations: {config['iterations']}")
    print(f"  • Learning Rate: {config['learn_rate']}")
    print(f"  • Dropout: {config['dropout']}")

    print(f"\n Training Results:")
    print(f"  • Final F1 Score: {metrics['final_f1']*100:.2f}%")
    print(f"  • Improvement: {metrics['improvement']:+.2f}%")
    print(f"  • Final Precision: {metrics['final_precision']*100:.2f}%")
    print(f"  • Final Recall: {metrics['final_recall']*100:.2f}%")

    if test_results:
        print(f"\nTest Results:")
        if 'names' in test_results:
            print(f"  • Names Recognition: {test_results['names']:.1f}%")
        if 'streets' in test_results:
            print(f"  • Streets Recognition: {test_results['streets']:.1f}%")
        if 'areas' in test_results:
            print(f"  • Areas Recognition: {test_results['areas']:.1f}%")
        if 'overall' in test_results:
            print(f"  • Overall Test Score: {test_results['overall']:.1f}%")

    if save_model:
        print(f"\n Model saved to: ../custom_spacy_model/{FINETUNED_MODEL_VERSION}")

    print("\n" + "="*80)
    print(" TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80 + "\n")

    # Log results to file
    log_results(timestamp, base_model, config, metrics, test_results, eval_scores, seed)

    return nlp, metrics, test_results


def log_results(
    timestamp: str,
    base_model: str,
    config: dict,
    metrics: dict,
    test_results: dict,
    eval_scores: dict,
    seed: int = None
):
    """Log training results to file"""
    log_filename = f"logs/training_{timestamp.replace(':', '.')}.txt"

    with open(log_filename, "w") as f:
        f.write("="*80 + "\n")
        f.write("SPACY NER TRAINING SESSION RESULTS\n")
        f.write("="*80 + "\n\n")

        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Base Model: {base_model}\n")
        f.write(f"Target Version: {FINETUNED_MODEL_VERSION}\n\n")

        f.write("CONFIGURATION:\n")
        f.write("-"*80 + "\n")
        for key, value in config.items():
            f.write(f"  {key}: {value}\n")

        f.write("\nTRAINING METRICS:\n")
        f.write("-"*80 + "\n")
        for key, value in metrics.items():
            if isinstance(value, float):
                f.write(f"  {key}: {value*100:.2f}%\n")
            else:
                f.write(f"  {key}: {value}\n")

        if test_results:
            f.write("\nTEST RESULTS:\n")
            f.write("-"*80 + "\n")
            for key, value in test_results.items():
                f.write(f"  {key}: {value:.2f}%\n")

        if eval_scores:
            f.write("\nEVALUATION SCORES:\n")
            f.write("-"*80 + "\n")
            for key, value in eval_scores.items():
                if isinstance(value, (int, float)):
                    f.write(f"  {key}: {value}\n")

        f.write("\n" + "="*80 + "\n")

    print(f" Results logged to: {log_filename}")

    # Also append to training.log for history
    with open("training.log", "a") as f:
        overall_score = test_results.get('overall', 0) if test_results else 0
        seed_str = f"Seed: {seed}, " if seed is not None else ""
        f.write(
            f"{timestamp}: {seed_str}Training with {config['iterations']} iterations. "
            f"F1: {metrics['final_f1']*100:.2f}%, Test Score: {overall_score:.2f}%. "
            f"Model: {base_model}\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train SpaCy NER model for Finnish text anonymization"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="fi_core_news_lg",
        help="Base SpaCy model to fine-tune (default: fi_core_news_lg)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Number of training iterations (default: use config value)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save the trained model"
    )
    parser.add_argument(
        "--no-ruler",
        action="store_true",
        help="Do not use entity ruler"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: use fixed seed from data_preparation)"
    )
    parser.add_argument(
        "--disable-gpu",
        action="store_true",
        help="Disable GPU/Metal acceleration (use CPU only, for Windows/Linux/containers)"
    )

    args = parser.parse_args()

    # Set environment variable before importing training_pipeline if GPU should be disabled
    if args.disable_gpu:
        os.environ['DISABLE_GPU'] = '1'

    try:
        nlp, metrics, test_results = main(
            base_model=args.base_model,
            iterations=args.iterations,
            save_model=not args.no_save,
            use_entity_ruler=not args.no_ruler,
            seed=args.seed
        )
    except KeyboardInterrupt:
        print("\n\n   Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n   Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

