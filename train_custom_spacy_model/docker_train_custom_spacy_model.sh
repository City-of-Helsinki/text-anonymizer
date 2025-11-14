#!/bin/sh

# Model name is in model_version.py in variable FINETUNED_MODEL_VERSION

# Extract the model version from the Python file
FINETUNED_MODEL_VERSION=$(python -c "from model_version import FINETUNED_MODEL_VERSION; print(FINETUNED_MODEL_VERSION)")

if [ ! -d "/app/custom_spacy_model/$FINETUNED_MODEL_VERSION" ]; then
    echo "Model not found, training..."
    python ./train_custom_spacy_model.py
  cp template_meta.json ../custom_spacy_model/$FINETUNED_MODEL_VERSION/meta.json
  cp template_config.cfg ../custom_spacy_model/$FINETUNED_MODEL_VERSION/config.cfg
fi


