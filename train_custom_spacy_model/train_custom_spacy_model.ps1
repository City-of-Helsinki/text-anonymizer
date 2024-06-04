python .\train_custom_spacy_model.py
Copy-Item -Path ".\template_meta_spacy_fi_lg.json" -Destination "..\custom_spacy_model\fi_datahel_spacy-0.0.2\meta.json"
Copy-Item -Path ".\template_config_spacy_fi_lg.cfg" -Destination "..\custom_spacy_model\fi_datahel_spacy-0.0.2\config.cfg"
Set-Location ".."
pip install -e .\custom_spacy_model
Set-Location ".\train_custom_spacy_model"
python .\evaluation.py