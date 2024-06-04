FROM ubuntu:22.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update --fix-missing && \
apt-get -y upgrade && \
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends sudo curl build-essential software-properties-common python-is-python3 python3-pip python3-dev git vim less && \
apt-get clean && \
rm -rf /var/lib/apt/lists/*

# Setup project directory & venv
RUN echo "Setup project dir"
RUN mkdir -p /app
WORKDIR /app

COPY ./train_custom_spacy_model/ /app/train_custom_spacy_model/
COPY ./custom_spacy_model/ /app/custom_spacy_model/
COPY ./text_anonymizer/ /app/text_anonymizer/
COPY ./examples/ /app/examples
COPY ./setup.py /app/
COPY ./requirements.in /app/
COPY ./requirements.txt /app/
COPY ./requirements-server.txt /app/
COPY ./*.py /app/
COPY ./entrypoint.sh /app/
COPY ./flask/ /app/flask


# Install project dependencies
RUN pip3 install -e /app/

# Install anonymizer dependencies
RUN pip3 install -r /app/requirements.in
# Install server dependencies
RUN pip3 install -r /app/requirements-server.txt


# Train custom spacy model and install it
COPY /test/data/ /app/test/data/

# if custom model is missing, train it
RUN chmod +x /app/train_custom_spacy_model/docker_train_custom_spacy_model.sh

# Install model
RUN (cd /app ; pip3 install -e custom_spacy_model/)

# Disable statistics
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Determine container mode in entrypoint
ENTRYPOINT ["/bin/bash", "./entrypoint.sh"]