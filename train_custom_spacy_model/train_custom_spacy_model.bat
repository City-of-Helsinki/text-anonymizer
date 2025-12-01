@echo off

REM Extract the model version from the Python file
for /f %%i in ('python -c "from model_version import FINETUNED_MODEL_VERSION; print(FINETUNED_MODEL_VERSION)"') do set FINETUNED_MODEL_VERSION=%%i

python main_train.py
copy template_meta_spacy_fi_lg.json ..\custom_spacy_model\%FINETUNED_MODEL_VERSION%\meta.json
copy template_config_spacy_fi_lg.cfg ..\custom_spacy_model\%FINETUNED_MODEL_VERSION%\config.cfg
cd ..
pip install -e custom_spacy_model
cd train_custom_spacy_model
python evaluation.py