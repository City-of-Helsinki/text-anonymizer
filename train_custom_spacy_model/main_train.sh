#!/bin/sh

# Main orchestration script for training the refactored SpaCy NER model
# This script:
# 1. Runs the complete training pipeline (ETL -> Training -> Evaluation)
# 2. Copies template config files to the trained model directory
# 3. Installs the model as a package
# 4. Runs final evaluation

set -e  # Exit on any error

echo "========================================================================"
echo "SpaCy NER Fine-Tuning Pipeline - Helsinki "
echo "========================================================================"
echo ""

# Get model version from Python
FINETUNED_MODEL_VERSION=$(python -c "from model_version import FINETUNED_MODEL_VERSION; print(FINETUNED_MODEL_VERSION)")
echo "Model Version: $FINETUNED_MODEL_VERSION"
echo ""

# Parse command line arguments
BASE_MODEL="fi_core_news_lg"
ITERATIONS=""
NO_SAVE=""
NO_RULER=""

while [ $# -gt 0 ]; do
  case "$1" in
    --base-model)
      BASE_MODEL="$2"
      shift 2
      ;;
    --iterations)
      ITERATIONS="--iterations $2"
      shift 2
      ;;
    --no-save)
      NO_SAVE="--no-save"
      shift
      ;;
    --no-ruler)
      NO_RULER="--no-ruler"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--base-model MODEL] [--iterations N] [--no-save] [--no-ruler]"
      exit 1
      ;;
  esac
done

echo " Configuration:"
echo "   Base Model: $BASE_MODEL"
[ -n "$ITERATIONS" ] && echo "   Iterations: $(echo $ITERATIONS | cut -d' ' -f2)"
echo ""

# ========================================================================
# STEP 1: Run the complete training pipeline
# ========================================================================
echo "========================================================================"
echo "STEP 1/4: Running Training Pipeline (ETL + Training + Evaluation)"
echo "========================================================================"
echo ""

python ./main_train.py --base-model "$BASE_MODEL" $ITERATIONS $NO_SAVE $NO_RULER

if [ $? -ne 0 ]; then
    echo ""
    echo " Training pipeline failed!"
    exit 1
fi

echo ""
echo " Training pipeline completed successfully!"
echo ""

# Check if model was saved (skip next steps if --no-save was used)
if [ -n "$NO_SAVE" ]; then
    echo "️  Model was not saved (--no-save flag used)"
    echo "   Skipping installation and final evaluation"
    echo ""
    echo "========================================================================"
    echo " Pipeline completed (model not saved)"
    echo "========================================================================"
    exit 0
fi

# ========================================================================
# STEP 2: Copy template files to model directory
# ========================================================================
echo "========================================================================"
echo "STEP 2/4: Copying Template Configuration Files"
echo "========================================================================"
echo ""

MODEL_PATH="../custom_spacy_model/$FINETUNED_MODEL_VERSION"

if [ ! -d "$MODEL_PATH" ]; then
    echo " Model directory not found: $MODEL_PATH"
    exit 1
fi

echo " Copying meta.json template..."
if [ -f "template_meta_spacy_fi_lg.json" ]; then
    cp template_meta_spacy_fi_lg.json "$MODEL_PATH/meta.json"
    echo " meta.json copied"
else
    echo "️  template_meta_spacy_fi_lg.json not found, skipping"
fi

echo " Copying config.cfg template..."
if [ -f "template_config_spacy_fi_lg.cfg" ]; then
    cp template_config_spacy_fi_lg.cfg "$MODEL_PATH/config.cfg"
    echo " config.cfg copied"
else
    echo "️  template_config_spacy_fi_lg.cfg not found, skipping"
fi

echo ""

# ========================================================================
# STEP 3: Install model as a package
# ========================================================================
echo "========================================================================"
echo "STEP 3/4: Installing Model as Python Package"
echo "========================================================================"
echo ""

cd ..
echo " Installing custom_spacy_model in editable mode..."
pip install -e custom_spacy_model

if [ $? -ne 0 ]; then
    echo ""
    echo " Model installation failed!"
    cd train_custom_spacy_model
    exit 1
fi

echo ""
echo " Model installed successfully!"
cd train_custom_spacy_model
echo ""

# ========================================================================
# STEP 4: Run final comprehensive evaluation
# ========================================================================
echo "========================================================================"
echo "STEP 4/4: Running Final Comprehensive Evaluation"
echo "========================================================================"
echo ""

echo "Running evaluation.py for detailed model assessment..."
echo ""

python ./evaluation.py

if [ $? -ne 0 ]; then
    echo ""
    echo "️  Final evaluation had issues, but model is trained and installed"
    exit 0
fi

echo ""
echo " Final evaluation completed!"
echo ""

# ========================================================================
# COMPLETION SUMMARY
# ========================================================================
echo "========================================================================"
echo " COMPLETE PIPELINE FINISHED SUCCESSFULLY!"
echo "========================================================================"
echo ""
echo " Summary:"
echo "   - Model trained and evaluated"
echo "   - Configuration files copied"
echo "   - Model installed as package"
echo "   - Final evaluation completed"
echo ""
echo " Model Location: $MODEL_PATH"
echo "- Version: $FINETUNED_MODEL_VERSION"
echo ""
echo "========================================================================"

