FROM ubuntu:22.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Base system setup
RUN apt-get update --fix-missing && \
    apt-get -y upgrade && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    sudo curl build-essential software-properties-common \
    python-is-python3 python3-pip python3-venv python3-dev git vim less && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create and activate a dedicated virtual environment for the app
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv ${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# Setup project directory
RUN echo "Setup project dir"
RUN mkdir -p /app
WORKDIR /app

# Copy only files needed to resolve and install dependencies first for better layer caching
COPY ./requirements.in /app/
COPY ./requirements-server.txt /app/
COPY ./setup.py /app/

# Generate requirements.txt from requirements.in inside the container (no hashes for cleaner logs)
RUN python -m pip install -U pip setuptools wheel pip-tools && \
    pip-compile --upgrade --output-file requirements.txt requirements.in

# Install application dependencies into the venv
RUN python -m pip install -r /app/requirements.txt && \
    python -m pip install -r /app/requirements-server.txt

# Copy application source after dependency installation for better caching
COPY ./train_custom_spacy_model/ /app/train_custom_spacy_model/
COPY ./custom_spacy_model/ /app/custom_spacy_model/
COPY ./text_anonymizer/ /app/text_anonymizer/
COPY ./examples/ /app/examples
COPY ./*.py /app/
COPY ./entrypoint.sh /app/
COPY ./flask/ /app/flask

# Install the package in editable mode after deps are installed
RUN python -m pip install -e /app/

# Trigger tldextract public suffix list download in build phase
RUN python -c "import tldextract; tldextract.extract('example.com')"

# Include test data (used by entrypoint/tests)
COPY /test/data/ /app/test/data/

# Ensure train script is executable (if used later by entrypoint)
RUN chmod +x /app/train_custom_spacy_model/docker_train_custom_spacy_model.sh

# Install custom spaCy model (editable) after base package
RUN (cd /app ; python -m pip install -e custom_spacy_model/)

# Disable statistics
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Determine container mode in entrypoint
ENTRYPOINT ["/bin/bash", "./entrypoint.sh"]