#!/bin/bash
MODE=${MODE:-dev}

if [[ $MODE = dev ]]
then
    echo "Run container in dev mode"
    tail -f /dev/null
elif [[ $MODE = api ]]
then
    echo "Run container in api mode"
    uvicorn anonymizer_api_app:anonymizer
elif [[ $MODE = streamlit ]]
then
    echo "Run container in streamlit mode"
    streamlit run anonymizer_web_app.py --server.port 8000
elif [[ $MODE = web ]]
then
    echo "Run container in web mode"
    gunicorn -w 4 -b 0.0.0.0:8000 --timeout 600 anonymizer_flask_app:app
elif [[ $MODE = webapi ]]
then
    echo "Run container in web/api mode"
    uvicorn anonymizer_api_webapp:main_app  --host 0.0.0.0

else
    echo "unknown mode: "$MODE", use 'dev', 'api', 'web' or leave empty (defaults to 'dev')"
fi