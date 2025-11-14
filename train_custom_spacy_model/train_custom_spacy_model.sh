#!/bin/sh

FINETUNED_MODEL_VERSION=$(python -c "from model_version import FINETUNED_MODEL_VERSION; print(FINETUNED_MODEL_VERSION)")

python ./train_custom_spacy_model.py
cp template_meta_spacy_fi_lg.json ../custom_spacy_model/$FINETUNED_MODEL_VERSION/meta.json
cp template_config_spacy_fi_lg.cfg ../custom_spacy_model/$FINETUNED_MODEL_VERSION/config.cfg
cd ..
pip install -e custom_spacy_model
cd train_custom_spacy_model
python ./evaluation.py
