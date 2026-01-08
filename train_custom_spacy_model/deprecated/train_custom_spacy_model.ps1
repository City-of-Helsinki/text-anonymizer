# Extract the model version from the Python file
$FINETUNED_MODEL_VERSION = python -c "from model_version import FINETUNED_MODEL_VERSION; print(FINETUNED_MODEL_VERSION)"

python .\main_train.py
Copy-Item -Path ".\template_meta_spacy_fi_lg.json" -Destination "..\custom_spacy_model\$FINETUNED_MODEL_VERSION\meta.json"
Copy-Item -Path ".\template_config_spacy_fi_lg.cfg" -Destination "..\custom_spacy_model\$FINETUNED_MODEL_VERSION\config.cfg"
Set-Location ".."
pip install -e .\custom_spacy_model
Set-Location ".\train_custom_spacy_model"
python .\evaluation.py