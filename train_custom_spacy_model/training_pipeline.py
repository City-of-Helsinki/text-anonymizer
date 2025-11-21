"""
Training Pipeline Module for SpaCy NER Fine-tuning

This module handles:
- Model training with the prepared data
- Early stopping and learning rate configuration
- Training progress monitoring
- Model saving
"""

import datetime
import random
from typing import Dict, List, Tuple

import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding

from model_version import FINETUNED_MODEL_VERSION

# Training configuration
TRAINING_CONFIG = {
    'iterations': 10,
    'dropout': 0.3,            # Higher regularization
    'learn_rate': 0.001,       # Keep higher rate
    'patience': 3,             # Stop faster
    'min_improvement': 0.01,   # 1% improvement required
    'train_split': 0.6,
}


class SpacyNERTrainer:
    """Handles the training of SpaCy NER model"""

    def __init__(
        self,
        nlp: spacy.Language,
        train_data: List[Example],
        config: Dict = None
    ):
        self.nlp = nlp
        self.train_data = train_data
        self.config = config or TRAINING_CONFIG
        self.metrics_history = {
            'loss': [],
            'val_f1': [],
            'val_precision': [],
            'val_recall': []
        }

        # Split data into train and validation
        random.shuffle(self.train_data)
        train_split_idx = int(self.config['train_split'] * len(self.train_data))
        self.train_examples = self.train_data[:train_split_idx]
        self.dev_examples = self.train_data[train_split_idx:]

        print(f"Data Split: {len(self.train_examples)} training, {len(self.dev_examples)} validation examples\n")

    def evaluate(self, data: List[Example], verbose: bool = False) -> Dict:
        """Evaluate model on given data"""
        scores = self.nlp.evaluate(data)

        if verbose and "ents_p" in scores:
            precision = scores["ents_p"]
            recall = scores["ents_r"]
            f1_score = scores["ents_f"]
            print(f"  Overall Metrics:")
            print(f"     Precision: {precision*100:6.2f}%  (How many detected entities are correct)")
            print(f"     Recall:    {recall*100:6.2f}%  (How many actual entities were found)")
            print(f"     F1 Score:  {f1_score*100:6.2f}%  (Balanced measure of both)")

        if verbose and "ents_per_type" in scores:
            per_type = scores["ents_per_type"]
            print(f"  Per Entity Type:")
            for entity_type, metrics in per_type.items():
                print(f"     {entity_type:8s}: P={metrics['p']*100:5.1f}%  R={metrics['r']*100:5.1f}%  F1={metrics['f']*100:5.1f}%")
        elif verbose:
            print("  No entity types found")

        return scores

    def train(self, verbose: bool = True) -> Tuple[float, Dict]:
        """
        Train the NER model with early stopping

        Returns:
            Tuple of (best_f1_score, final_metrics)
        """
        print("\n" + "="*80)
        print("STARTING SPACY NER MODEL TRAINING")
        print("="*80)
        print(f"Training Configuration:")
        print(f"  • Max Iterations: {self.config['iterations']}")
        print(f"  • Learning Rate: {self.config['learn_rate']}")
        print(f"  • Dropout: {self.config['dropout']}")
        print(f"  • Early Stopping Patience: {self.config['patience']}")
        print(f"  • Min Improvement Threshold: {self.config['min_improvement']}")
        print(f"  • Total training examples: {len(self.train_examples)}")
        print(f"  • Validation examples: {len(self.dev_examples)}")
        print("="*80 + "\n")

        # Baseline evaluation
        print("BASELINE EVALUATION (Before Training)")
        print("-" * 80)
        baseline_scores = self.evaluate(self.dev_examples, verbose=True)
        baseline_f1 = baseline_scores.get("ents_f", 0.0)
        baseline_precision = baseline_scores.get("ents_p", 0.0)
        baseline_recall = baseline_scores.get("ents_r", 0.0)
        print(f"\nStarting training to improve from baseline F1: {baseline_f1*100:.2f}%")
        print("-" * 80 + "\n")

        # Training loop
        best_f1 = 0
        patience_counter = 0

        # Get only NER pipe
        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != 'ner']

        with self.nlp.disable_pipes(*other_pipes):
            optimizer = self.nlp.resume_training()

            # Configure optimizer
            optimizer.learn_rate = self.config['learn_rate']
            optimizer.L2 = 1e-6  # L2 regularization

            print("TRAINING IN PROGRESS")
            print("-" * 80)
            print(f"Learning Rate: {self.config['learn_rate']}, Dropout: {self.config['dropout']}")
            print(f"{'Iter':>6} | {'Loss':>8} | {'F1':>7} | {'Precision':>9} | {'Recall':>7} | {'Status':>20}")
            print("-" * 80)

            for i in range(self.config['iterations']):
                # Shuffle training data each iteration
                random.shuffle(self.train_examples)

                losses = {}
                # Update model with training examples
                batches = minibatch(self.train_examples, size=compounding(4.0, 32.0, 1.001))
                for batch in batches:
                    self.nlp.update(
                        batch,
                        drop=self.config['dropout'],
                        losses=losses,
                        sgd=optimizer
                    )

                # Evaluate on validation set
                val_scores = self.evaluate(self.dev_examples, verbose=False)
                current_f1 = val_scores.get("ents_f") or 0.0
                current_precision = val_scores.get("ents_p") or 0.0
                current_recall = val_scores.get("ents_r") or 0.0

                # Track metrics
                loss_value = losses.get('ner', 0.0)
                self.metrics_history['loss'].append(loss_value)
                self.metrics_history['val_f1'].append(current_f1)
                self.metrics_history['val_precision'].append(current_precision)
                self.metrics_history['val_recall'].append(current_recall)

                # Determine status
                status = ""
                improvement = current_f1 - best_f1
                if improvement > self.config['min_improvement']:
                    status = f"+{improvement*100:.2f}% IMPROVED!"
                    best_f1 = current_f1
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter == 1:
                        status = f"No significant improvement"
                    else:
                        status = f"No improvement ({patience_counter}/{self.config['patience']})"

                # Print progress
                if verbose or i % 1 == 0 or i == self.config['iterations'] - 1:
                    print(f"{i+1:6d} | {loss_value:8.4f} | {current_f1*100:6.2f}% | {current_precision*100:8.2f}% | {current_recall*100:6.2f}% | {status}")

                # Early stopping check
                if patience_counter >= self.config['patience']:
                    print("-" * 80)
                    print(f"Early stopping at iteration {i+1} (no improvement for {self.config['patience']} iterations)")
                    break

            print("-" * 80)

        # Re-enable other pipes
        for p in other_pipes:
            self.nlp.enable_pipe(p)

        # Calculate improvements
        final_f1 = self.metrics_history['val_f1'][-1]
        final_precision = self.metrics_history['val_precision'][-1]
        final_recall = self.metrics_history['val_recall'][-1]

        f1_improvement = (best_f1 - baseline_f1) * 100
        precision_improvement = (final_precision - baseline_precision) * 100
        recall_improvement = (final_recall - baseline_recall) * 100

        # Print summary
        print("\n" + "="*80)
        print("TRAINING RESULTS SUMMARY")
        print("="*80)
        print(f"\n{'Metric':<15} | {'Before':>10} | {'After':>10} | {'Change':>15}")
        print("-" * 80)
        print(f"{'F1 Score':<15} | {baseline_f1*100:9.2f}% | {best_f1*100:9.2f}% | {f1_improvement:+14.2f}%")
        print(f"{'Precision':<15} | {baseline_precision*100:9.2f}% | {final_precision*100:9.2f}% | {precision_improvement:+14.2f}%")
        print(f"{'Recall':<15} | {baseline_recall*100:9.2f}% | {final_recall*100:9.2f}% | {recall_improvement:+14.2f}%")
        print("-" * 80)

        # Overall verdict
        if f1_improvement > 5:
            print(f"EXCELLENT: Training significantly improved the model! (+{f1_improvement:.1f}% F1)")
        elif f1_improvement > 1:
            print(f"GOOD: Training improved the model moderately. (+{f1_improvement:.1f}% F1)")
        elif f1_improvement > 0:
            print(f"MINOR: Training showed slight improvement. (+{f1_improvement:.1f}% F1)")
        else:
            print(f"NO EFFECT: Training did not improve the model. ({f1_improvement:.1f}% F1)")

        print("="*80 + "\n")

        return best_f1, {
            'baseline_f1': baseline_f1,
            'final_f1': best_f1,
            'improvement': f1_improvement,
            'final_precision': final_precision,
            'final_recall': final_recall
        }

    def add_entity_ruler(self, patterns_data: Dict[str, List[str]]):
        """Add entity ruler with patterns"""
        if "entity_ruler" not in self.nlp.pipe_names:
            ruler = self.nlp.add_pipe(
                "entity_ruler",
                after="ner",
                config={"phrase_matcher_attr": "LOWER", "overwrite_ents": True}
            )
        else:
            ruler = self.nlp.get_pipe("entity_ruler")

        print("Adding patterns to Entity Ruler...")
        for label, data_list in patterns_data.items():
            patterns = [{'pattern': item, 'label': label} for item in data_list]
            ruler.add_patterns(patterns)
            print(f"  • Added {len(patterns)} patterns for {label}")

        print("Entity Ruler patterns added\n")


def save_model(nlp: spacy.Language, target_path: str = None) -> bool:
    """
    Save the trained model to disk

    Args:
        nlp: Trained SpaCy model
        target_path: Path to save model (default from FINETUNED_MODEL_VERSION)

    Returns:
        True if saved successfully, False otherwise
    """
    if target_path is None:
        target_path = f"../custom_spacy_model/{FINETUNED_MODEL_VERSION}"

    print("\n" + "MODEL SAVING")
    print("-" * 80)
    print(f"Saving model to: {target_path}")

    try:
        nlp.to_disk(target_path)
        print("Model saved successfully!")
        print("-" * 80)
        return True
    except Exception as e:
        print(f"Error saving model: {e}")
        print("-" * 80)
        return False


def train_model_pipeline(
    train_data: List[Example],
    nlp: spacy.Language,
    use_entity_ruler: bool = True,
    patterns_data: Dict[str, List[str]] = None,
    save_model_path: str = None,
    config: Dict = None
) -> Tuple[float, Dict]:
    """
    Complete training pipeline

    Args:
        train_data: List of training examples
        nlp: SpaCy model to train
        use_entity_ruler: Whether to add entity ruler
        patterns_data: Dictionary of patterns for entity ruler
        save_model_path: Path to save trained model
        config: Training configuration dictionary

    Returns:
        Tuple of (best_f1_score, metrics_dict)
    """
    print("\n" + "="*80)
    print("SPACY NER MODEL TRAINING PIPELINE")
    print("="*80)
    print(f"Base Model: {nlp.meta['name']}")
    print(f"Training Examples: {len(train_data)}")
    print(f"Timestamp: {datetime.datetime.now().strftime('%Y.%m.%d %H:%M')}")
    print("="*80 + "\n")

    # Initialize trainer
    trainer = SpacyNERTrainer(nlp, train_data, config)

    # Add entity ruler if requested
    if use_entity_ruler and patterns_data:
        trainer.add_entity_ruler(patterns_data)

    # Train model
    best_f1, metrics = trainer.train(verbose=True)

    # Save model if path provided
    if save_model_path:
        save_model(nlp, save_model_path)

    return best_f1, metrics


if __name__ == "__main__":
    # Example usage
    from data_preparation import prepare_training_data

    print("Preparing training data...")
    train_data, nlp = prepare_training_data(base_model="fi_core_news_lg")

    print("\nStarting training pipeline...")
    best_f1, metrics = train_model_pipeline(
        train_data=train_data,
        nlp=nlp,
        use_entity_ruler=True,
        save_model_path=f"../custom_spacy_model/{FINETUNED_MODEL_VERSION}"
    )

    print(f"\nTraining completed with F1 score: {best_f1*100:.2f}%")

