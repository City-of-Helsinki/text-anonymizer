#!/bin/bash
MODE=${MODE:-dev}

# Helper: ensure a command exists, else fail fast with a clear message
require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Missing required command: $1" >&2
        echo "Container dependencies appear incomplete. Ensure requirements were installed during image build." >&2
        echo "Tip: Rebuild the image: docker-compose up --build -d (or docker build ...)" >&2
        exit 1
    fi
}

if [[ $MODE = dev ]]
then
    echo "Run container in dev mode"
    tail -f /dev/null
elif [[ $MODE = api ]]
then
    echo "Run container in api mode"
    require_cmd uvicorn
    python -c "import presidio_analyzer, spacy" 2>/dev/null || {
        echo "Required Python modules not available. Rebuild the image to install requirements." >&2
        exit 1
    }
    # Target actual FastAPI app object defined in anonymizer_api_app.py
    uvicorn anonymizer_api_app:anonymizer_api --host 0.0.0.0 --port 8000
elif [[ $MODE = web ]]
then
    echo "Run container in web mode"
    require_cmd gunicorn
    python -c "import flask" 2>/dev/null || {
        echo "Flask not available. Rebuild the image to install requirements." >&2
        exit 1
    }
    gunicorn -w 4 -b 0.0.0.0:8000 --timeout 600 anonymizer_flask_app:app
elif [[ $MODE = webapi ]]
then
    echo "Run container in web/api mode"
    require_cmd uvicorn
    python -c "import presidio_analyzer, spacy" 2>/dev/null || {
        echo "Required Python modules not available. Rebuild the image to install requirements." >&2
        exit 1
    }
    uvicorn anonymizer_api_webapp:main_app --host 0.0.0.0 --port 8000

else
    echo "unknown mode: "$MODE", use 'dev', 'api', 'web', 'webapi' or leave empty (defaults to 'dev')"
fi