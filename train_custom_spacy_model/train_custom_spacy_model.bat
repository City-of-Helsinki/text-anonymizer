@echo off
python train_custom_spacy_model.py
copy template_meta_spacy_fi_lg.json ..\custom_spacy_model\fi_datahel_spacy-0.0.2\meta.json
copy template_config_spacy_fi_lg.cfg ..\custom_spacy_model\fi_datahel_spacy-0.0.2\config.cfg
cd ..
pip install -e custom_spacy_model
cd train_custom_spacy_model
python evaluation.py