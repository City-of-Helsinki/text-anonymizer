# Set the virtual environment name
$venv_path = "venv"
python -m venv $venv_path
# Activate the virtual environment
& $venv_path\Scripts\Activate.ps1

# Install project
Set-Location ".."
python -m pip install -r requirements.in
python -m pip install -r requirements-server.txt
Set-Location "train_custom_spacy_model"
.\train_custom_spacy_model.ps1
Set-Location ".."
# Smoke test
echo "Test anonymizer: Antti Aalto" | python anonymize.py

Write-Host "Setup complete."
